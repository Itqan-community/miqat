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
        Hybrid Pro v3 (Fusion-Pro): WhisperX Backbone + CTC Micro-Refinement.
        """
        print("[Hybrid-Pro] Starting Fusion-Pro v3...")
        
        # 1. WhisperX Base Pass (The reliable sequence)
        w_alignments = self.align_whisperx(audio_path, reference_text)
        
        # 2. CTC Base Pass (The precise candidate)
        # We run CTC on the whole audio to get a full candidate set
        c_alignments = self.align(audio_path, reference_text)
        
        # 3. Intelligent Fusion
        final_alignments = []
        
        # Create a searchable map of CTC results by word text
        # (Using a sliding window or index search because words repeat)
        c_idx = 0
        
        for w_entry in w_alignments:
            word_text = self._normalize_arabic(w_entry["word"])
            best_ctc = None
            min_dist = 0.6 # Max allowed distance to trust CTC (seconds)
            
            # Look for a matching word in CTC results within a time window
            # Search nearby indices in c_alignments to handle repetitions correctly
            search_start = max(0, c_idx - 5)
            search_end = min(len(c_alignments), c_idx + 10)
            
            for i in range(search_start, search_end):
                c_entry = c_alignments[i]
                c_text = self._normalize_arabic(c_entry["word"])
                
                if word_text == c_text:
                    # Check temporal distance between centers
                    dist = abs(((w_entry["start"] + w_entry["end"])/2) - ((c_entry["start"] + c_entry["end"])/2))
                    if dist < min_dist:
                        best_ctc = c_entry
                        min_dist = dist
                        c_idx = i + 1 # Move CTC pointer forward
                        break
            
            if best_ctc and best_ctc["confidence"] > 0.4:
                # Use CTC for precision, but clamp to a reasonable window around WhisperX
                # to prevent radical jumps
                refined_start = best_ctc["start"]
                refined_end = best_ctc["end"]
                
                # Boundary check: don't drift more than 0.5s from WhisperX
                if abs(refined_start - w_entry["start"]) > 0.5:
                    refined_start = w_entry["start"]
                if abs(refined_end - w_entry["end"]) > 0.5:
                    refined_end = w_entry["end"]

                final_alignments.append({
                    "word": w_entry["word"],
                    "start": round(refined_start, 3),
                    "end": round(refined_end, 3),
                    "confidence": max(w_entry["confidence"], best_ctc["confidence"])
                })
            else:
                # Fallback to WhisperX backbone (Robust)
                final_alignments.append(w_entry)

        # Final pass: Sanitize timestamps (non-decreasing)
        for i in range(1, len(final_alignments)):
            # Ensure no negative duration and no overlap
            if final_alignments[i]["start"] < final_alignments[i-1]["end"]:
                # If they overlap, find a middle ground or nudge
                overlap = final_alignments[i-1]["end"] - final_alignments[i]["start"]
                if overlap < 0.2:
                    final_alignments[i]["start"] = final_alignments[i-1]["end"]
                else:
                    # Significant overlap, keep WhisperX's original start if possible
                    pass 
            
            if final_alignments[i]["end"] <= final_alignments[i]["start"]:
                final_alignments[i]["end"] = final_alignments[i]["start"] + 0.1

        print(f"[Hybrid-Pro] Fusion complete. Refined {len([a for a in final_alignments if a.get('refined', False)])} words.")
        return final_alignments

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