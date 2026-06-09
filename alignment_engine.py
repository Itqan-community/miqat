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
        print(f"Loading models on device: {self.device}")
        if self.device == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")

        print(f"Loading Whisper model from {self.whisper_path}...")
        self.whisper_processor = WhisperProcessor.from_pretrained(self.whisper_path)
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
            self.whisper_path, attn_implementation="eager"
        ).to(self.device)
        # Verify model is actually on GPU
        first_param = next(self.whisper_model.parameters())
        print(f"  Whisper model device: {first_param.device} (expected: {self.device})")

        print(f"Loading Wav2Vec2 model from {self.wav2vec2_path}...")
        self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained(self.wav2vec2_path)
        self.wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(self.wav2vec2_path).to(self.device)
        # Verify model is actually on GPU
        first_param = next(self.wav2vec2_model.parameters())
        print(f"  Wav2Vec2 model device: {first_param.device} (expected: {self.device})")

    # ─── WhisperX Implementation ─────────────────────────────────────────────

    def _normalize_arabic(self, text: str, for_ctc: bool = False) -> str:
        """
        Enhanced normalization for Arabic. 
        for_ctc=True removes all special Quranic marks that Wav2Vec2 usually doesn't recognize.
        """
        # Remove diacritics
        text = re.sub(r'[\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]', '', text)
        
        if for_ctc:
            # Normalize Alefs
            text = re.sub(r'[أإآٱ]', 'ا', text)
            # Normalize Hamza on Ya and Waw (optional, but safer for some models)
            # text = re.sub(r'[ؤئ]', 'ء', text)
            # Remove any non-arabic letters except space
            text = re.sub(r'[^\u0621-\u064A\s]', '', text)
            
        return text.strip()

    def align_whisperx(self, audio_path: str, reference_text: str) -> List[Dict]:
        """
        State-of-the-art alignment using WhisperX + Robust DP Mapping.
        """
        import whisperx
        import torch

        print("[WhisperX] Starting alignment...")
        print(f"[WhisperX] CUDA available: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        print(f"[WhisperX] Using device={device}, compute_type={compute_type}")

        # Use local CTranslate2 model to avoid network download hang
        local_model_path = os.path.join(os.path.dirname(__file__), "model_local", "whisperx")
        if os.path.isdir(local_model_path) and os.listdir(local_model_path):
            print(f"[WhisperX] Using local model: {local_model_path}")
            model_name = local_model_path
        else:
            print("[WhisperX] No local model found, downloading from HF...")
            model_name = "large-v2"

        print("[WhisperX] Loading model (this may take a minute)...")
        model = whisperx.load_model(model_name, device, compute_type=compute_type, language="ar")
        print(f"[WhisperX] Model loaded successfully on {device}")
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=16)
        
        model_a, metadata = whisperx.load_align_model(language_code="ar", device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        
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

        # --- Robust DP Mapping (Longest Common Subsequence style) ---
        ref_words = reference_text.split()
        n = len(ref_words)
        m = len(extracted_words)
        
        # Scoring matrix: dp[i][j] = max score matching ref_words[:i] with extracted_words[:j]
        dp = np.zeros((n + 1, m + 1))
        
        # Fill DP table
        for i in range(1, n + 1):
            rw = self._normalize_arabic(ref_words[i-1], for_ctc=True)
            for j in range(1, m + 1):
                ew = self._normalize_arabic(extracted_words[j-1]["word"], for_ctc=True)
                
                # Match score (fuzzy ratio)
                match_score = fuzz.ratio(rw, ew) / 100.0
                if match_score < 0.6: match_score = -1.0 # Penalty for bad matches
                
                # We want to match as many as possible correctly
                dp[i][j] = max(
                    dp[i-1][j],      # Skip ref word
                    dp[i][j-1],      # Skip extracted word
                    dp[i-1][j-1] + match_score # Match
                )
        
        # Backtrack to find optimal mapping
        mapped_alignments = [None] * n
        i, j = n, m
        while i > 0 and j > 0:
            rw = self._normalize_arabic(ref_words[i-1], for_ctc=True)
            ew = self._normalize_arabic(extracted_words[j-1]["word"], for_ctc=True)
            match_score = fuzz.ratio(rw, ew) / 100.0
            
            if match_score >= 0.6 and dp[i][j] == dp[i-1][j-1] + match_score:
                mapped_alignments[i-1] = extracted_words[j-1]
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i-1][j]:
                i -= 1
            else:
                j -= 1
        
        # Finalize results with interpolation for missing words
        final_results = []
        for k in range(n):
            if mapped_alignments[k]:
                final_results.append({
                    "word": ref_words[k],
                    "start": mapped_alignments[k]["start"],
                    "end": mapped_alignments[k]["end"],
                    "confidence": mapped_alignments[k]["confidence"]
                })
            else:
                # Interpolate missing word timing
                prev_end = final_results[-1]["end"] if final_results else 0
                final_results.append({
                    "word": ref_words[k],
                    "start": prev_end,
                    "end": prev_end + 0.1,
                    "confidence": 0.0
                })

        # Cleanup memory
        del model
        del model_a
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        return final_results

    def transcribe(self, audio_path: str) -> str:
        if not self.whisper_model: self.load_models()
        speech, sr = sf.read(audio_path, dtype='float32')
        if len(speech.shape) > 1: speech = speech.mean(axis=1)
        if sr != 16000: speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
        input_features = self.whisper_processor(speech, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
        predicted_ids = self.whisper_model.generate(input_features)
        return self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    # ─── CTC Segmentation ─────────────────────────────────────────────────────

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
        
        # CLEAN WORDS FOR CTC: This is crucial to avoid "invalid literal" errors
        clean_words = [self._normalize_arabic(w, for_ctc=True) for w in words]
        clean_words = [w for w in clean_words if w]
        
        if not clean_words:
            return []

        try:
            results = ctc_segmentation(config, combined_log_probs, clean_words)
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
        words = reference_text.split()
        return self.align_ctc(speech, sr, words)

    def align_hybrid(self, audio_path: str, reference_text: str) -> List[Dict]:
        """
        Hybrid Pro v3 (Fusion-Pro): WhisperX Backbone + CTC Micro-Refinement.
        """
        print("[Hybrid-Pro] Starting Fusion-Pro v3 (DP Optimized)...")
        w_alignments = self.align_whisperx(audio_path, reference_text)
        c_alignments = self.align(audio_path, reference_text)
        
        final_alignments = []
        c_idx = 0
        
        for w_entry in w_alignments:
            word_text = self._normalize_arabic(w_entry["word"], for_ctc=True)
            best_ctc = None
            min_dist = 0.6 
            
            search_start = max(0, c_idx - 5)
            search_end = min(len(c_alignments), c_idx + 10)
            
            for i in range(search_start, search_end):
                c_entry = c_alignments[i]
                c_text = self._normalize_arabic(c_entry["word"], for_ctc=True)
                
                if word_text == c_text:
                    dist = abs(((w_entry["start"] + w_entry["end"])/2) - ((c_entry["start"] + c_entry["end"])/2))
                    if dist < min_dist:
                        best_ctc = c_entry
                        min_dist = dist
                        c_idx = i + 1 
                        break
            
            if best_ctc and best_ctc["confidence"] > 0.4:
                refined_start = best_ctc["start"]
                refined_end = best_ctc["end"]
                
                if abs(refined_start - w_entry["start"]) > 0.5:
                    refined_start = w_entry["start"]
                if abs(refined_end - w_entry["end"]) > 0.5:
                    refined_end = w_entry["end"]

                final_alignments.append({
                    "word": w_entry["word"],
                    "start": round(refined_start, 3),
                    "end": round(refined_end, 3),
                    "confidence": max(w_entry["confidence"], best_ctc["confidence"]),
                    "refined": True
                })
            else:
                final_alignments.append(w_entry)

        # Get total duration for the final word extension
        try:
            total_duration = sf.info(audio_path).duration
        except:
            total_duration = final_alignments[-1]["end"] + 2.0

        # Final pass: Sanitize timestamps and apply Tail Expansion
        for i in range(len(final_alignments)):
            # 1. Non-decreasing check
            if i > 0:
                if final_alignments[i]["start"] < final_alignments[i-1]["end"]:
                    final_alignments[i]["start"] = final_alignments[i-1]["end"]
            
            # 2. Tail Expansion (The "Madd" and "Breath" fix)
            if i < len(final_alignments) - 1:
                next_start = final_alignments[i+1]["start"]
                current_end = final_alignments[i]["end"]
                gap = next_start - current_end
                if gap > 0.1:
                    final_alignments[i]["end"] = round(next_start - 0.1, 3)
            else:
                # Last word of the entire recording: extend until the actual end of audio
                final_alignments[i]["end"] = round(total_duration, 3)

            # 3. Final safety check (ensure end > start)
            if final_alignments[i]["end"] <= final_alignments[i]["start"]:
                final_alignments[i]["end"] = round(final_alignments[i]["start"] + 0.1, 3)

        print(f"[Hybrid-Pro] Fusion complete. Tail expansion applied to end of audio ({total_duration:.2f}s).")
        return final_alignments

    def align_smart(self, audio_path: str, reference_text: str) -> List[Dict]:
        return self.align_hybrid(audio_path, reference_text)

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _linear_fallback(self, words: List[str], duration: float) -> List[Dict]:
        step = duration / max(len(words), 1)
        return [{"word": w, "start": round(i * step, 3), "end": round((i + 1) * step, 3), "confidence": 0.1} for i, w in enumerate(words)]