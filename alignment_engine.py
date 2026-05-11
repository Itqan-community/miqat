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

    # ─── Public API ───────────────────────────────────────────────────────────

    def align(self, audio_path: str, reference_text: str) -> List[Dict]:
        """
        Robust Forced Alignment using the ctc-segmentation library.
        Handles long audio files by processing emissions in chunks and then
        running the global segmentation algorithm.
        """
        if not self.wav2vec2_model:
            self.load_models()

        # 1. Load and prepare audio
        speech, sr = sf.read(audio_path, dtype='float32')
        if len(speech.shape) > 1:
            speech = speech.mean(axis=1)
        if sr != 16000:
            speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
            sr = 16000

        duration = len(speech) / sr
        audio_tensor = torch.from_numpy(speech).to(self.device)

        # 2. Get CTC Emissions (Log Probs)
        # Process in large chunks to avoid OOM, but concatenate into a single matrix
        chunk_size = 30 * 16000  # 30 seconds
        all_log_probs = []

        with torch.inference_mode():
            for i in range(0, len(audio_tensor), chunk_size):
                chunk = audio_tensor[i : i + chunk_size]
                if len(chunk) < 400: continue # Skip tiny fragments
                
                logits = self.wav2vec2_model(chunk.unsqueeze(0)).logits
                log_probs = torch.log_softmax(logits, dim=-1).cpu()
                all_log_probs.append(log_probs)
                
            combined_log_probs = torch.cat(all_log_probs, dim=1)[0].numpy()

        # 3. Prepare Text and Vocabulary
        reference_text = self._normalize_arabic(reference_text)
        words = reference_text.split()
        
        # Get character list from tokenizer
        vocab = self.wav2vec2_processor.tokenizer.get_vocab()
        # Sort vocab by ID
        inv_vocab = {v: k for k, v in vocab.items()}
        char_list = [inv_vocab[i] for i in range(len(inv_vocab))]

        # 4. CTC Segmentation Configuration
        config = CtcSegmentationParameters()
        config.char_list = char_list
        config.index_duration = duration / combined_log_probs.shape[0]
        
        # We pass words as separate segments to get word-level timing
        # The library finds the optimal split points for these segments
        try:
            results = ctc_segmentation(config, combined_log_probs, words)
        except Exception as e:
            print(f"[CTC-Seg] Error: {e}. Falling back to basic linear distribution.")
            return self._linear_fallback(words, duration)

        # 5. Format Results
        word_alignments = []
        for i, segment in enumerate(results):
            # segment: (start_time, end_time, score)
            word_alignments.append({
                "word":       words[i],
                "start":      round(float(segment[0]), 3),
                "end":        round(float(segment[1]), 3),
                "confidence": round(min(1.0, float(np.exp(segment[2]))), 2),
            })

        print(f"[CTC-Seg] Generated {len(word_alignments)} word timestamps using ctc-segmentation library.")
        return word_alignments

    def align_smart(self, audio_path: str, reference_text: str) -> List[Dict]:
        """
        With ctc-segmentation, 'smart' alignment is the default 'align' 
        because the library is designed for long files.
        """
        return self.align(audio_path, reference_text)

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _normalize_arabic(self, text: str) -> str:
        pattern = re.compile(r'[\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]')
        return pattern.sub('', text)

    def _linear_fallback(self, words: List[str], duration: float) -> List[Dict]:
        step = duration / max(len(words), 1)
        return [{
            "word": w,
            "start": round(i * step, 3),
            "end": round((i + 1) * step, 3),
            "confidence": 0.1
        } for i, w in enumerate(words)]

    def transcribe(self, audio_path: str) -> str:
        if not self.whisper_model: self.load_models()
        speech, sr = sf.read(audio_path, dtype='float32')
        if len(speech.shape) > 1: speech = speech.mean(axis=1)
        if sr != 16000: speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
        
        inputs = self.whisper_processor(speech, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)
        predicted_ids = self.whisper_model.generate(input_features)
        return self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]