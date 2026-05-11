import os
import torch
import soundfile as sf
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from typing import List, Dict
import gc
import librosa
from rapidfuzz import fuzz
import re
from ctc_segmentation import ctc_segmentation, CtcSegmentationParameters

class AlignmentEngine:
    def __init__(self, whisper_path: str, wav2vec2_path: str):
        self.whisper_path = whisper_path
        self.wav2vec2_path = wav2vec2_path
        self.whisper_model = None
        self.whisper_processor = None
        self.wav2vec2_model = None
        self.wav2vec2_processor = None
        self.whisperx_model = None # Lazy load
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_models(self):
        print(f"Loading Whisper model from {self.whisper_path}...")
        self.whisper_processor = WhisperProcessor.from_pretrained(self.whisper_path)
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
            self.whisper_path, attn_implementation="eager"
        ).to(self.device)

        print(f"Loading Wav2Vec2 model from {self.wav2vec2_path}...")
        self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained(self.wav2vec2_path)
        self.wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(self.wav2vec2_path).to(self.device)

    # ─── WhisperX Implementation ─────────────────────────────────────────────

    def align_whisperx(self, audio_path: str, reference_text: str) -> List[Dict]:
        """
        State-of-the-art alignment using WhisperX (Whisper + Phoneme Alignment).
        """
        import whisperx
        import torch

        print("[WhisperX] Starting alignment...")
        # 1. Load Model (Lazy)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        # Load whisper model for transcription if not already loaded as whisperx
        model = whisperx.load_model("large-v2", device, compute_type=compute_type, language="ar")

        # 2. Transcribe with VAD
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=16)
        
        # 3. Align with language-specific model (Arabic)
        model_a, metadata = whisperx.load_align_model(language_code="ar", device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        
        # 4. Map to reference text
        # WhisperX gives us word-level timestamps directly. 
        # We need to map these to the provided reference_text using fuzzy matching.
        
        extracted_words = []
        for segment in result["segments"]:
            if "words" in segment:
                for w in segment["words"]:
                    if "start" in w and "end" in w:
                        extracted_words.append({
                            "word": w["word"],
                            "start": w["start"],
                            "end": w["end"],
                            "confidence": w.get("score", 0.9)
                        })

        # Fuzzy mapping to reference words to maintain the exact reference text structure
        ref_words = self._normalize_arabic(reference_text).split()
        mapped_alignments = []
        
        # Simple greedy mapping for now
        last_found_idx = 0
        for ref_w in ref_words:
            found = False
            # Search in a window of the extracted words
            search_window = extracted_words[max(0, last_found_idx - 5) : last_found_idx + 20]
            for i, ext_w in enumerate(search_window):
                if fuzz.ratio(self._normalize_arabic(ref_w), self._normalize_arabic(ext_w["word"])) > 75:
                    mapped_alignments.append({
                        "word": ref_w,
                        "start": ext_w["start"],
                        "end": ext_w["end"],
                        "confidence": ext_w["confidence"]
                    })
                    last_found_idx = max(0, last_found_idx - 5) + i + 1
                    found = True
                    break
            
            if not found:
                # Interpolate or use fallback for missing words
                prev_end = mapped_alignments[-1]["end"] if mapped_alignments else 0
                mapped_alignments.append({
                    "word": ref_w,
                    "start": prev_end,
                    "end": prev_end + 0.2,
                    "confidence": 0.0
                })

        # Cleanup memory
        del model
        del model_a
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        return mapped_alignments

    # ─── CTC Segmentation (Previous) ──────────────────────────────────────────

    def align_ctc(self, audio_data: np.ndarray, sr: int, words: List[str], offset: float = 0.0) -> List[Dict]:
        """
        Internal CTC alignment for a given audio numpy array and word list.
        """
        if not self.wav2vec2_model:
            self.load_models()

        if sr != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
            sr = 16000

        duration = len(audio_data) / sr
        audio_tensor = torch.from_numpy(audio_data).to(self.device)

        chunk_size = 30 * 16000
        all_log_probs = []

        with torch.inference_mode():
            for i in range(0, len(audio_tensor), chunk_size):
                chunk = audio_tensor[i : i + chunk_size]
                if len(chunk) < 400: continue
                logits = self.wav2vec2_model(chunk.unsqueeze(0)).logits
                log_probs = torch.log_softmax(logits, dim=-1).cpu()
                all_log_probs.append(log_probs)
            
            if not all_log_probs:
                return []
            combined_log_probs = torch.cat(all_log_probs, dim=1)[0].numpy()

        vocab = self.wav2vec2_processor.tokenizer.get_vocab()
        inv_vocab = {v: k for k, v in vocab.items()}
        char_list = [inv_vocab[i] for i in range(len(inv_vocab))]

        config = CtcSegmentationParameters()
        config.char_list = char_list
        config.index_duration = duration / combined_log_probs.shape[0]
        
        try:
            results = ctc_segmentation(config, combined_log_probs, words)
        except Exception as e:
            print(f"[CTC-Seg] Local Error: {e}")
            return []

        word_alignments = []
        for i, segment in enumerate(results):
            word_alignments.append({
                "word":       words[i],
                "start":      round(float(segment[0]) + offset, 3),
                "end":        round(float(segment[1]) + offset, 3),
                "confidence": round(min(1.0, float(np.exp(segment[2]))), 2),
            })
        return word_alignments

    def align(self, audio_path: str, reference_text: str) -> List[Dict]:
        speech, sr = sf.read(audio_path, dtype='float32')
        if len(speech.shape) > 1: speech = speech.mean(axis=1)
        reference_text = self._normalize_arabic(reference_text)
        words = reference_text.split()
        return self.align_ctc(speech, sr, words)

    def align_hybrid(self, audio_path: str, reference_text: str) -> List[Dict]:
        """
        Hybrid Pro v2: Anchor-to-Anchor segmented CTC refinement.
        """
        print("[Hybrid-Pro] Starting alignment v2...")
        
        # 1. WhisperX Global Pass
        w_alignments = self.align_whisperx(audio_path, reference_text)
        
        # 2. Anchor Identification
        # An anchor is a word with very high confidence.
        anchors = []
        for i, entry in enumerate(w_alignments):
            if entry["confidence"] > 0.92: # More strict for v2
                anchors.append(i)
        
        if not anchors:
            print("[Hybrid-Pro] No strong anchors found, falling back to WhisperX.")
            return w_alignments

        # 3. Local CTC Refinement
        speech, sr = sf.read(audio_path, dtype='float32')
        if len(speech.shape) > 1: speech = speech.mean(axis=1)
        duration_total = len(speech) / sr
        
        refined_alignments = []
        last_anchor_idx = -1
        last_anchor_end_time = 0.0
        
        # Reference words
        ref_words = [w["word"] for w in w_alignments]

        # Add a virtual end anchor
        anchor_indices = anchors + [len(w_alignments)]

        for current_anchor_idx in anchor_indices:
            gap_start_idx = last_anchor_idx + 1
            gap_end_idx = current_anchor_idx
            
            # Determine window end time
            if current_anchor_idx < len(w_alignments):
                current_anchor_start_time = w_alignments[current_anchor_idx]["start"]
            else:
                current_anchor_start_time = duration_total

            if gap_start_idx < gap_end_idx:
                # We have words between anchors (or before the first anchor)
                # WINDOW: From the end of the last anchor to the start of the current anchor
                t_start = max(0, last_anchor_end_time - 0.3)
                t_end = min(duration_total, current_anchor_start_time + 0.3)
                
                start_sample = int(t_start * sr)
                end_sample = int(t_end * sr)
                sub_audio = speech[start_sample:end_sample]
                sub_words = ref_words[gap_start_idx:gap_end_idx]
                
                print(f"[Hybrid-Pro] Refining gap {gap_start_idx}-{gap_end_idx-1} in window [{t_start:.2f}s, {t_end:.2f}s]")
                
                ctc_results = self.align_ctc(sub_audio, sr, sub_words, offset=t_start)
                
                if len(ctc_results) == len(sub_words):
                    refined_alignments.extend(ctc_results)
                else:
                    print(f"[Hybrid-Pro] CTC failed for gap {gap_start_idx}, fallback to WhisperX.")
                    refined_alignments.extend(w_alignments[gap_start_idx:gap_end_idx])
            
            # Add the anchor word itself
            if current_anchor_idx < len(w_alignments):
                anchor_word = w_alignments[current_anchor_idx]
                refined_alignments.append(anchor_word)
                last_anchor_end_time = anchor_word["end"]
                last_anchor_idx = current_anchor_idx

        # Final cleanup: ensure timestamps are non-decreasing
        for i in range(1, len(refined_alignments)):
            if refined_alignments[i]["start"] < refined_alignments[i-1]["end"]:
                refined_alignments[i]["start"] = refined_alignments[i-1]["end"]
            if refined_alignments[i]["end"] < refined_alignments[i]["start"]:
                refined_alignments[i]["end"] = refined_alignments[i]["start"] + 0.1

        return refined_alignments

    def align_smart(self, audio_path: str, reference_text: str) -> List[Dict]:
        return self.align(audio_path, reference_text)

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _normalize_arabic(self, text: str) -> str:
        pattern = re.compile(r'[\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]')
        return pattern.sub('', text)

    def _linear_fallback(self, words: List[str], duration: float) -> List[Dict]:
        step = duration / max(len(words), 1)
        return [{"word": w, "start": round(i * step, 3), "end": round((i + 1) * step, 3), "confidence": 0.1} for i, w in enumerate(words)]

    def transcribe(self, audio_path: str) -> str:
        if not self.whisper_model: self.load_models()
        speech, sr = sf.read(audio_path, dtype='float32')
        if len(speech.shape) > 1: speech = speech.mean(axis=1)
        if sr != 16000: speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
        input_features = self.whisper_processor(speech, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
        predicted_ids = self.whisper_model.generate(input_features)
        return self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]