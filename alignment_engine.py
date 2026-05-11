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
        whisper_bin = os.path.join(self.whisper_path, "pytorch_model.bin")
        wav2vec2_bin = os.path.join(self.wav2vec2_path, "pytorch_model.bin")

        if not os.path.exists(whisper_bin):
            raise FileNotFoundError(
                f"Whisper model weights not found at {whisper_bin}. "
                "Please run model_downloader.py."
            )
        if not os.path.exists(wav2vec2_bin):
            raise FileNotFoundError(
                f"Wav2Vec2 model weights not found at {wav2vec2_bin}. "
                "Please run model_downloader.py."
            )

        print(f"Loading Whisper model from {self.whisper_path}...")
        try:
            self.whisper_processor = WhisperProcessor.from_pretrained(self.whisper_path)
            self.whisper_model = (
                WhisperForConditionalGeneration
                .from_pretrained(self.whisper_path, attn_implementation="eager")
                .to(self.device)
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {e}")

        print(f"Loading Wav2Vec2 model from {self.wav2vec2_path}...")
        try:
            self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained(self.wav2vec2_path)
            self.wav2vec2_model = (
                Wav2Vec2ForCTC
                .from_pretrained(self.wav2vec2_path)
                .to(self.device)
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Wav2Vec2 model: {e}")

    # ─── Public API ───────────────────────────────────────────────────────────

    def transcribe(self, audio_path: str) -> str:
        if not self.whisper_model:
            self.load_models()

        speech, sr = sf.read(audio_path, dtype='float32')
        if len(speech.shape) > 1:
            speech = speech.mean(axis=1)
        if sr != 16000:
            speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
            sr = 16000

        input_features = self.whisper_processor(
            speech, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(self.device)

        predicted_ids = self.whisper_model.generate(input_features)
        transcription = self.whisper_processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]
        return transcription

    def align(self, audio_path: str, reference_text: str) -> List[Dict]:
        """Forced alignment using CTC (Wav2Vec2). Handles audio of any length."""
        if not self.wav2vec2_model:
            self.load_models()

        speech, sr = sf.read(audio_path, dtype='float32')
        if len(speech.shape) > 1:
            speech = speech.mean(axis=1)
        if sr != 16000:
            speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
            sr = 16000

        audio_tensor = torch.from_numpy(speech).to(self.device)

        # ── Process in 30-second chunks with overlap to avoid CUDA OOM ─────────
        chunk_size_seconds = 30
        overlap_seconds    = 2        # 2s overlap to avoid cutting madd letters
        chunk_length       = chunk_size_seconds * 16000
        overlap_length     = overlap_seconds * 16000
        step_length        = chunk_length - overlap_length
        all_logits         = []

        with torch.inference_mode():
            pos = 0
            while pos < len(audio_tensor):
                chunk = audio_tensor[pos: pos + chunk_length]
                if len(chunk) < 400:
                    chunk = torch.nn.functional.pad(chunk, (0, 400 - len(chunk)))

                chunk_logits = self.wav2vec2_model(chunk.unsqueeze(0)).logits

                # Drop overlapping frames from previous chunk (except first)
                if pos > 0:
                    overlap_frames = int(
                        chunk_logits.shape[1] * overlap_length / len(chunk)
                    )
                    chunk_logits = chunk_logits[:, overlap_frames:, :]

                all_logits.append(chunk_logits.cpu())

                del chunk_logits
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if pos + chunk_length >= len(audio_tensor):
                    break
                pos += step_length

            logits = torch.cat(all_logits, dim=1)
            emissions = torch.log_softmax(logits, dim=-1)

        emission = emissions[0].detach()

        # ── Tokenise reference text ───────────────────────────────────────────
        reference_text = self._normalize_arabic(reference_text)
        words = reference_text.split()
        vocab = self.wav2vec2_processor.tokenizer.get_vocab()
        special_ids = [
            vocab.get('[PAD]', -1), vocab.get('[UNK]', -1),
            vocab.get('<s>', -1),   vocab.get('</s>', -1),
        ]
        word_delimiter = getattr(
            self.wav2vec2_processor.tokenizer, 'word_delimiter_token', None
        )
        word_delimiter_id = vocab.get(word_delimiter, -1) if word_delimiter else -1

        # Strategy 1: encode full text to preserve context
        full_tokens = self.wav2vec2_processor.tokenizer.encode(reference_text)
        full_tokens = [t for t in full_tokens if t not in special_ids]

        word_token_spans = []
        if word_delimiter_id != -1:
            start_idx = 0
            for idx, t in enumerate(full_tokens):
                if t == word_delimiter_id:
                    if start_idx < idx:
                        word_token_spans.append((start_idx, idx - 1))
                    start_idx = idx + 1
            if start_idx < len(full_tokens):
                word_token_spans.append((start_idx, len(full_tokens) - 1))

        # Strategy 2: word-by-word fallback if counts don't match
        if len(word_token_spans) != len(words):
            full_tokens = []
            word_token_spans = []
            for i, word in enumerate(words):
                word_tokens = self.wav2vec2_processor.tokenizer.encode(word)
                word_tokens_filtered = [
                    t for t in word_tokens
                    if t not in special_ids and t != word_delimiter_id
                ]
                if not word_tokens_filtered:
                    word_tokens_filtered = [vocab.get('[UNK]', 3)]

                start_idx = len(full_tokens)
                full_tokens.extend(word_tokens_filtered)
                end_idx = len(full_tokens) - 1
                word_token_spans.append((start_idx, end_idx))

                if i < len(words) - 1 and word_delimiter_id != -1:
                    full_tokens.append(word_delimiter_id)

        tokens = full_tokens
        if not tokens:
            return []

        # ── CTC alignment ─────────────────────────────────────────────────────
        trellis = self._get_trellis(emission, tokens)
        path = self._backtrack(trellis, emission, tokens)

        transition_frames = [p for p in path if p.get('is_changed')]
        is_linear_fallback = False
        
        if len(transition_frames) < len(tokens):
            is_linear_fallback = True
            step = emission.shape[0] / max(len(tokens), 1)
            transition_frames = [
                {'time_index': int(i * step), 'token_index': i}
                for i in range(len(tokens))
            ]

        ratio = len(audio_tensor) / 16000 / emission.shape[0]
        total_duration = len(audio_tensor) / 16000
        word_alignments = []
        
        # Calculate a basic confidence score: 1.0 if CTC found a path, 0.2 if it failed and used linear fallback
        score = 0.95 if not is_linear_fallback else 0.2

        for i, word in enumerate(words):
            start_idx, end_idx = word_token_spans[i]
            start_frame = transition_frames[start_idx]['time_index']
            start_time = round(start_frame * ratio, 3)

            if end_idx + 1 < len(transition_frames):
                end_frame = transition_frames[end_idx + 1]['time_index']
                end_time = round(end_frame * ratio, 3)

                if i < len(words) - 1:
                    next_start_idx = word_token_spans[i + 1][0]
                    next_start_time = round(
                        transition_frames[next_start_idx]['time_index'] * ratio, 3
                    )
                    gap = next_start_time - end_time
                    end_time += min(gap, 0.05) if gap > 0 else -0.03
                else:
                    end_time = total_duration
            else:
                if i == len(words) - 1:
                    end_time = total_duration
                else:
                    end_frame = transition_frames[end_idx]['time_index']
                    end_time = round((end_frame + 10) * ratio, 3)

            if i == len(words) - 1:
                end_time = total_duration
            else:
                end_time = min(end_time, total_duration)
            if end_time <= start_time:
                end_time = start_time + 0.1

            word_alignments.append({
                "word":       word,
                "start":      round(start_time, 3),
                "end":        round(end_time, 3),
                "confidence": score,
            })

        return word_alignments, score

    # ─── Whisper alignment ────────────────────────────────────────────────────

    def align_whisper(self, audio_path: str, reference_text: str) -> List[Dict]:
        """
        Uses Whisper's word-level timestamps.
        Automatically chunks audio > 28 seconds to handle long files.
        """
        if not self.whisper_model:
            self.load_models()

        speech, sr = sf.read(audio_path, dtype='float32')
        if len(speech.shape) > 1:
            speech = speech.mean(axis=1)
        if sr != 16000:
            speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
            sr = 16000

        duration = len(speech) / sr

        # ── For long audio: chunk into 28s windows and combine ────────────────
        WHISPER_MAX_SEC = 28        # safe margin below Whisper's 30s window
        OVERLAP_SEC     = 1.0       # overlap to catch words near chunk edges

        if duration > WHISPER_MAX_SEC:
            print(f"[Whisper] Audio is {duration:.1f}s → chunking into {WHISPER_MAX_SEC}s windows")
            return self._align_whisper_chunked(
                speech, sr, reference_text, WHISPER_MAX_SEC, OVERLAP_SEC
            )

        # ── Short audio: process in one shot ──────────────────────────────────
        return self._align_whisper_single(speech, sr, reference_text, time_offset=0.0)

    def _align_whisper_chunked(
        self,
        speech: np.ndarray,
        sr: int,
        reference_text: str,
        chunk_sec: float,
        overlap_sec: float,
    ) -> List[Dict]:
        """Tiles Whisper over long audio and stitches word timestamps."""
        chunk_samples   = int(chunk_sec * sr)
        overlap_samples = int(overlap_sec * sr)
        step_samples    = chunk_samples - overlap_samples

        all_whisper_words = []
        offset_sec = 0.0
        pos = 0

        while pos < len(speech):
            chunk = speech[pos: pos + chunk_samples]
            chunk_duration = len(chunk) / sr

            print(f"[Whisper] Chunk at {offset_sec:.1f}s ({chunk_duration:.1f}s)")

            words = self._align_whisper_single(chunk, sr, reference_text="", time_offset=offset_sec)
            # Only keep words whose start falls in the non-overlapping region
            # (except for the very first chunk)
            cutoff = offset_sec + overlap_sec if pos > 0 else 0.0
            for w in words:
                if w["start"] >= cutoff:
                    all_whisper_words.append(w)

            pos += step_samples
            offset_sec = pos / sr

            torch.cuda.empty_cache()
            gc.collect()

        if not all_whisper_words:
            print("[Whisper] No words extracted from chunks")
            return []

        # Map stitched Whisper words to reference text
        ref_words = reference_text.split()
        return self._map_to_reference(all_whisper_words, ref_words)

    def _align_whisper_single(
        self,
        speech: np.ndarray,
        sr: int,
        reference_text: str,
        time_offset: float = 0.0,
    ) -> List[Dict]:
        """Runs Whisper on a single ≤30s segment. Returns raw word list."""
        inputs = self.whisper_processor(
            speech, sampling_rate=16000, return_tensors="pt", return_attention_mask=True
        )
        input_features = inputs.input_features.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device) if "attention_mask" in inputs else None

        # Ensure alignment_heads are set (needed for token timestamps)
        if (not hasattr(self.whisper_model.generation_config, 'alignment_heads')
                or self.whisper_model.generation_config.alignment_heads is None):
            self.whisper_model.generation_config.alignment_heads = [
                [3, 1], [4, 2], [4, 7], [5, 1], [5, 2], [5, 4]
            ]
            print("[Whisper] Set alignment_heads from whisper-base config")

        try:
            generate_kwargs = {
                "return_token_timestamps":  True,
                "return_dict_in_generate":  True,
            }
            if attention_mask is not None:
                generate_kwargs["attention_mask"] = attention_mask

            result = self.whisper_model.generate(input_features, **generate_kwargs)
        except Exception as e:
            print(f"[Whisper] Generation failed: {e}")
            return []

        # ── Extract token IDs ─────────────────────────────────────────────────
        if hasattr(result, 'sequences'):
            token_ids = result.sequences[0]
        elif isinstance(result, torch.Tensor):
            token_ids = result[0]
        else:
            for key in ['sequences', 'output_token_ids']:
                if key in result:
                    token_ids = result[key][0]
                    break
            else:
                print("[Whisper] Cannot find token IDs")
                return []

        # ── Extract timestamps ────────────────────────────────────────────────
        token_timestamps = None
        if hasattr(result, 'token_timestamps') and result.token_timestamps is not None:
            token_timestamps = result.token_timestamps[0]
        elif isinstance(result, dict) and 'token_timestamps' in result:
            token_timestamps = result['token_timestamps'][0]

        if token_timestamps is None:
            print("[Whisper] token_timestamps not available")
            return []

        transcription = self.whisper_processor.decode(token_ids, skip_special_tokens=True)
        print(f"[Whisper] Transcribed: {transcription}")

        tokenizer = self.whisper_processor.tokenizer
        whisper_words = []
        current_word_tokens = []
        current_word_start = None
        current_word_end = None

        for tid, ts in zip(token_ids, token_timestamps):
            if tid.item() in tokenizer.all_special_ids:
                continue
            token_text = tokenizer.decode([tid.item()])
            ts_val = ts.item() + time_offset  # apply chunk offset

            if token_text.startswith("<|") and token_text.endswith("|>"):
                continue

            if token_text.startswith(" ") or token_text.startswith("Ġ"):
                if current_word_tokens:
                    word_text = tokenizer.decode(current_word_tokens).strip()
                    if word_text:
                        whisper_words.append({
                            "text":  word_text,
                            "start": current_word_start,
                            "end":   current_word_end,
                        })
                current_word_tokens = [tid.item()]
                current_word_start = ts_val
                current_word_end = ts_val
            elif not current_word_tokens:
                current_word_tokens = [tid.item()]
                current_word_start = ts_val
                current_word_end = ts_val
            else:
                current_word_tokens.append(tid.item())
                current_word_end = ts_val

        if current_word_tokens:
            word_text = tokenizer.decode(current_word_tokens).strip()
            if word_text:
                whisper_words.append({
                    "text":  word_text,
                    "start": current_word_start,
                    "end":   current_word_end,
                })

        return whisper_words

    # ─── Smart alignment for long audio ──────────────────────────────────────

    def align_smart(self, audio_path: str, reference_text: str) -> List[Dict]:
        """
        For long audio:
          Uses a Sliding Window approach purely on CPU to split audio into 5-minute chunks
          with overlap. It passes enough text to each chunk to ensure valid alignments,
          dropping squished words at the end, and advancing the text pointer.
          Avoids Whisper entirely to prevent CUDA OOM on long files.
        """
        if not self.wav2vec2_model:
            self.load_models()

        print(f"[SmartAlign] Loading audio: {audio_path}")
        speech, sr = sf.read(audio_path, dtype='float32')
        if len(speech.shape) > 1:
            speech = speech.mean(axis=1)
        if sr != 16000:
            speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
            sr = 16000

        duration = len(speech) / sr
        print(f"[SmartAlign] Duration: {duration:.2f}s")

        if duration < 60 * 7: # If less than 7 minutes, direct CTC is safe
            print("[SmartAlign] File is short enough for direct CTC alignment.")
            return self.align(audio_path, reference_text)

        # Sliding window parameters
        chunk_sec   = 300.0   # 5 minutes per chunk
        overlap_sec = 30.0    # 30 seconds overlap between consecutive chunks

        reference_text = self._normalize_arabic(reference_text)
        ref_words   = reference_text.split()
        total_words = len(ref_words)
        words_per_sec = total_words / duration

        # KEY FIX: Use 1.5x the expected words per chunk (not 4x).
        # With 4x, large Surahs had words_per_chunk > total_words, so chunk-0 got
        # ALL words and CTC squished the entire surah into the first 300 seconds.
        expected_words_per_chunk = chunk_sec * words_per_sec
        words_per_chunk = max(int(expected_words_per_chunk * 1.5), 50)

        # Only keep CTC words whose chunk-relative end falls before this cutoff,
        # to avoid the squished tail CTC produces when it runs out of tokens.
        safe_zone_sec = chunk_sec - (overlap_sec / 2.0)

        final_alignments = []
        current_word_idx = 0
        pos_sec = 0.0

        while pos_sec < duration and current_word_idx < total_words:
            end_sec       = min(pos_sec + chunk_sec, duration)
            is_last_chunk = (end_sec >= duration - 0.5)

            chunk_start_sample = int(pos_sec * sr)
            chunk_end_sample   = int(end_sec * sr)
            chunk_speech       = speech[chunk_start_sample:chunk_end_sample]

            # Slice the reference words for this chunk
            if is_last_chunk:
                chunk_words_end = total_words
            else:
                chunk_words_end = min(current_word_idx + words_per_chunk, total_words)

            chunk_ref_words = ref_words[current_word_idx:chunk_words_end]
            chunk_text      = " ".join(chunk_ref_words)

            print(
                f"[SmartAlign] Chunk {pos_sec:.1f}s - {end_sec:.1f}s | "
                f"Words {current_word_idx} to {chunk_words_end} ({len(chunk_ref_words)} words)"
            )

            chunk_tmp = f"{audio_path}_temp.wav"
            sf.write(chunk_tmp, chunk_speech, sr)

            try:
                ctc_result, score = self.align(chunk_tmp, chunk_text)
                
                # --- RESYNC LOGIC ---
                # If score is very low, CTC is faking it. Try to find the true anchor using Whisper.
                if score < 0.4 and not is_last_chunk:
                    print(f"[SmartAlign] Low confidence ({score}). Attempting Whisper Resync...")
                    # Run a 10s whisper transcription from the middle of the chunk
                    whisper_sample = chunk_speech[sr*5 : sr*15] if len(chunk_speech) > sr*15 else chunk_speech
                    if len(whisper_sample) > 16000:
                        raw_transcription = self.transcribe_chunk(whisper_sample)
                        norm_trans = self._normalize_arabic(raw_transcription)
                        
                        # Find where this transcription fits in the reference text (search radius: 500 words)
                        search_start = max(0, current_word_idx - 100)
                        search_end   = min(total_words, current_word_idx + 500)
                        
                        found_idx = self._find_best_word_match(norm_trans, ref_words[search_start:search_end])
                        if found_idx != -1:
                            global_found_idx = search_start + found_idx
                            print(f"[SmartAlign] Resync successful! Anchor found at index {global_found_idx}")
                            
                            # Adjust index and retry aligning with a fresh start
                            current_word_idx = global_found_idx
                            pos_sec += 5.0 # Skip the confusing part
                            continue 

                valid_alignments = []
                words_consumed   = 0

                for i, wa in enumerate(ctc_result):
                    if not is_last_chunk and wa['end'] > safe_zone_sec:
                        break

                    wa['start'] = round(wa['start'] + pos_sec, 3)
                    wa['end']   = round(wa['end']   + pos_sec, 3)
                    valid_alignments.append(wa)
                    words_consumed = i + 1

                if not valid_alignments:
                    print("[SmartAlign] WARNING: No valid words found. Advancing audio.")
                    pos_sec += chunk_sec / 2.0
                    continue

                final_alignments.extend(valid_alignments)
                current_word_idx += words_consumed

                if is_last_chunk or current_word_idx >= total_words:
                    break

                last_end = valid_alignments[-1]['end']
                next_start_sec = max(last_end - (overlap_sec / 4.0), pos_sec + 2.0)
                pos_sec = next_start_sec

            except Exception as e:
                print(f"[SmartAlign] Error: {e}")
                pos_sec += 30.0

            finally:
                if os.path.exists(chunk_tmp):
                    os.remove(chunk_tmp)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        return final_alignments

    def transcribe_chunk(self, speech_np: np.ndarray) -> str:
        """Helper to quickly transcribe a small audio segment."""
        if not self.whisper_model: self.load_models()
        input_features = self.whisper_processor(speech_np, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
        predicted_ids = self.whisper_model.generate(input_features)
        return self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    def _find_best_word_match(self, snippet: str, search_words: List[str]) -> int:
        """Finds the starting index of 'snippet' within 'search_words' using fuzzy matching."""
        snippet_words = snippet.split()
        if not snippet_words: return -1
        
        best_score = 0
        best_idx = -1
        
        # Look for the first 3 words of the snippet
        anchor = " ".join(snippet_words[:3])
        for i in range(len(search_words) - 3):
            candidate = " ".join(search_words[i:i+3])
            score = fuzz.ratio(anchor, candidate)
            if score > 85 and score > best_score:
                best_score = score
                best_idx = i
        
        return best_idx

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _normalize_arabic(self, text: str) -> str:
        """Removes Arabic diacritics and Uthmani symbols to improve CTC alignment."""
        # Matches basic tashkeel, small high letters, and Quranic annotation signs
        pattern = re.compile(r'[\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]')
        return pattern.sub('', text)

    def _map_to_reference(
        self,
        whisper_words: List[Dict],
        ref_words: List[str],
    ) -> List[Dict]:
        """
        Maps a flat list of Whisper word dicts to the reference word list,
        adjusting timing so every reference word has a timestamp.
        """
        if not whisper_words:
            return []

        total_start = whisper_words[0]["start"]
        total_end   = whisper_words[-1]["end"]

        if len(whisper_words) == len(ref_words):
            word_alignments = [
                {
                    "word":       ref_words[i],
                    "start":      round(w["start"], 3),
                    "end":        round(w["end"],   3),
                    "confidence": 0.95,
                }
                for i, w in enumerate(whisper_words)
            ]
        elif len(whisper_words) > len(ref_words):
            chunks_per_word = len(whisper_words) / len(ref_words)
            word_alignments = []
            for i in range(len(ref_words)):
                s = min(int(round(i * chunks_per_word)),       len(whisper_words) - 1)
                e = min(int(round((i + 1) * chunks_per_word)) - 1, len(whisper_words) - 1)
                word_alignments.append({
                    "word":       ref_words[i],
                    "start":      round(whisper_words[s]["start"], 3),
                    "end":        round(whisper_words[e]["end"],   3),
                    "confidence": 0.8,
                })
        else:
            # Fewer Whisper words → distribute time proportionally
            char_lengths   = [max(len(w), 1) for w in ref_words]
            total_chars    = sum(char_lengths)
            total_duration = total_end - total_start
            word_alignments = []
            current_time = total_start
            for i, word in enumerate(ref_words):
                word_dur = total_duration * (char_lengths[i] / total_chars)
                word_alignments.append({
                    "word":       word,
                    "start":      round(current_time, 3),
                    "end":        round(current_time + word_dur, 3),
                    "confidence": 0.7,
                })
                current_time += word_dur

        # Fix overlaps / micro-gaps
        for i in range(len(word_alignments) - 1):
            gap = word_alignments[i + 1]["start"] - word_alignments[i]["end"]
            if gap > 0:
                word_alignments[i]["end"] += min(gap, 0.05)
            elif gap < 0:
                word_alignments[i]["end"] = word_alignments[i + 1]["start"] - 0.02
            else:
                word_alignments[i]["end"] -= 0.03
            if word_alignments[i]["end"] <= word_alignments[i]["start"]:
                word_alignments[i]["end"] = word_alignments[i]["start"] + 0.05
            word_alignments[i]["end"] = round(word_alignments[i]["end"], 3)

        print(f"[Whisper] Generated {len(word_alignments)} word timestamps:")
        for wa in word_alignments:
            print(f"  {wa['word']}: {wa['start']}s - {wa['end']}s (conf: {wa['confidence']})")

        return word_alignments

    # Kept for backward-compatibility (old align_whisper called this)
    def _match_whisper_to_reference(
        self, whisper_chunks: List[Dict], ref_words: List[str]
    ) -> List[Dict]:
        words = [{"text": c["timestamp"][0], "start": c["timestamp"][0], "end": c["timestamp"][1]}
                 for c in whisper_chunks]
        return self._map_to_reference(words, ref_words)

    # ─── CTC internals ────────────────────────────────────────────────────────

    def _get_trellis(self, emission, tokens, blank_id=0):
        num_frame  = emission.size(0)
        num_tokens = len(tokens)
        trellis    = torch.zeros((num_frame, num_tokens + 1))
        trellis[1:, 0]  = torch.cumsum(emission[1:, blank_id], 0)
        trellis[0,  1:] = -float('inf')
        trellis[-num_tokens:, 0] = float('inf')

        for t in range(num_frame - 1):
            trellis[t + 1, 1:] = torch.maximum(
                trellis[t, 1:] + emission[t, blank_id],
                trellis[t, :-1] + emission[t, tokens],
            )
        return trellis

    def _backtrack(self, trellis, emission, tokens, blank_id=0):
        t, j = trellis.size(0) - 1, trellis.size(1) - 1
        path = []
        while j > 0 and t > 0:
            stayed  = trellis[t - 1, j]     + emission[t - 1, blank_id]
            changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]
            path.append({
                'time_index':  t - 1,
                'token_index': j - 1,
                'is_changed':  changed > stayed,
            })
            t -= 1
            if changed > stayed:
                j -= 1
        return path[::-1]


if __name__ == "__main__":
    pass