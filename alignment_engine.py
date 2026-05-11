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

        ref_words = self._normalize_arabic(reference_text).split()
        audio_duration = len(audio) / 16000
        
        # Robust DP mapping to reference words
        # This ensures that repetitive patterns (Mutashabihat) are correctly aligned
        # by finding the globally optimal mapping instead of a local greedy one.
        
        ext_words_norm = [self._normalize_arabic(w["word"]) for w in extracted_words]
        ref_words_norm = [self._normalize_arabic(w) for w in ref_words]
        
        n, m = len(ref_words_norm), len(ext_words_norm)
        if n == 0: return []
        if m == 0: 
            return self._linear_fallback(ref_words, audio_duration) 

        # DP table for alignment score calculation
        dp = np.zeros((n + 1, m + 1))
        
        # Filling the DP table
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                ratio = fuzz.ratio(ref_words_norm[i-1], ext_words_norm[j-1]) / 100.0
                match_score = (ratio * 2.0) - 1.0
                dp[i][j] = max(
                    dp[i-1][j-1] + match_score, 
                    dp[i-1][j] - 0.5, 
                    dp[i][j-1] - 0.5
                )
        
        # Backtrack to find the optimal mapping
        mapping = {}
        i, j = n, m
        while i > 0 and j > 0:
            ratio = fuzz.ratio(ref_words_norm[i-1], ext_words_norm[j-1]) / 100.0
            match_score = (ratio * 2.0) - 1.0
            
            if dp[i][j] >= dp[i-1][j-1] + match_score - 1e-5 and ratio > 0.4:
                mapping[i-1] = j-1
                i -= 1
                j -= 1
            elif dp[i][j] >= dp[i-1][j] - 0.5 - 1e-5:
                i -= 1
            else:
                j -= 1
        
        # Construct the final mapped alignments with smart proportional interpolation
        mapped_alignments = []
        last_ext_end = 0
        
        for k in range(n):
            if k in mapping:
                ext_w = extracted_words[mapping[k]]
                mapped_alignments.append({
                    "word": ref_words[k],
                    "start": ext_w["start"],
                    "end": ext_w["end"],
                    "confidence": ext_w["confidence"]
                })
                last_ext_end = ext_w["end"]
            else:
                # Proportional Interpolation: distribute remaining time
                next_start = audio_duration
                gap_count = 1
                for next_k in range(k + 1, n):
                    if next_k in mapping:
                        next_start = extracted_words[mapping[next_k]]["start"]
                        break
                    gap_count += 1
                
                available_time = max(0, next_start - last_ext_end)
                time_per_word = available_time / gap_count
                
                start = last_ext_end
                end = start + time_per_word
                
                mapped_alignments.append({
                    "word": ref_words[k],
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "confidence": 0.0
                })
                last_ext_end = end

        # Cleanup memory
        del model
        del model_a
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        return mapped_alignments

    # ─── CTC Segmentation (Previous) ──────────────────────────────────────────

    def align(self, audio_path: str, reference_text: str) -> List[Dict]:
        if not self.wav2vec2_model:
            self.load_models()

        speech, sr = sf.read(audio_path, dtype='float32')
        if len(speech.shape) > 1: speech = speech.mean(axis=1)
        if sr != 16000:
            speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
            sr = 16000

        duration = len(speech) / sr
        audio_tensor = torch.from_numpy(speech).to(self.device)

        chunk_size = 30 * 16000
        all_log_probs = []

        with torch.inference_mode():
            for i in range(0, len(audio_tensor), chunk_size):
                chunk = audio_tensor[i : i + chunk_size]
                if len(chunk) < 400: continue
                logits = self.wav2vec2_model(chunk.unsqueeze(0)).logits
                log_probs = torch.log_softmax(logits, dim=-1).cpu()
                all_log_probs.append(log_probs)
            
            combined_log_probs = torch.cat(all_log_probs, dim=1)[0].numpy()

        reference_text = self._normalize_arabic(reference_text)
        words = reference_text.split()
        vocab = self.wav2vec2_processor.tokenizer.get_vocab()
        inv_vocab = {v: k for k, v in vocab.items()}
        char_list = [inv_vocab[i] for i in range(len(inv_vocab))]

        config = CtcSegmentationParameters()
        config.char_list = char_list
        config.index_duration = duration / combined_log_probs.shape[0]
        
        try:
            results = ctc_segmentation(config, combined_log_probs, words)
        except Exception as e:
            print(f"[CTC-Seg] Error: {e}")
            return self._linear_fallback(words, duration)

        word_alignments = []
        for i, segment in enumerate(results):
            word_alignments.append({
                "word":       words[i],
                "start":      round(float(segment[0]), 3),
                "end":        round(float(segment[1]), 3),
                "confidence": round(min(1.0, float(np.exp(segment[2]))), 2),
            })
        return word_alignments

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