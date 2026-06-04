"""
Audio encoding for CosyVoice3 inference.

Extracts three kinds of features from a reference audio clip:
  - Speech tokens  (speech_tokenizer_v3.onnx)  — 16 kHz / Whisper mel path
  - Speaker embedding (campplus.onnx)           — 16 kHz / Kaldi fbank path
  - Prompt mel features (matcha mel_spectrogram) — 24 kHz path

For CosyVoice2/3, prompt_feat and prompt_token are aligned to 2:1 ratio
(token_mel_ratio = 2) as required by the flow matching model.
"""

import logging
import os
import sys
from functools import partial
from typing import Dict, Optional

import numpy as np
import onnxruntime
import soundfile as sf
import torch

from serve.paths import (
    CAMPPLUS_ONNX,
    SPEECH_TOKENIZER_ONNX,
    CHECKPOINTS_DIR,
)

# Ensure CosyVoice + Matcha-TTS are importable for hyperpyyaml / matcha deps
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_COSYVOICE_DIR = os.path.join(_REPO_ROOT, 'CosyVoice')
_MATCHA_DIR = os.path.join(_COSYVOICE_DIR, 'third_party', 'Matcha-TTS')
for _p in [_REPO_ROOT, _COSYVOICE_DIR, _MATCHA_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = logging.getLogger(__name__)


def _load_wav(wav_path: str, target_sr: int) -> torch.Tensor:
    """Load audio file and resample to target_sr. Returns (1, num_samples)."""
    import torchaudio.transforms

    wav_raw, sr_raw = sf.read(wav_path, dtype='float32')
    wav = torch.tensor(wav_raw, dtype=torch.float32)
    if wav.dim() > 1:
        wav = wav.mean(dim=1)
    wav = wav.unsqueeze(0)  # (1, num_samples)
    if sr_raw != target_sr:
        wav = torchaudio.transforms.Resample(sr_raw, target_sr)(wav)
    return wav


def extract_speech_token(
    wav_path: str,
    onnx_path: str = SPEECH_TOKENIZER_ONNX,
    provider: str = 'CUDAExecutionProvider',
) -> Dict[str, object]:
    """Extract speech tokens from audio via speech_tokenizer_v3.onnx.

    Pipeline: load at 16kHz -> Whisper log-mel (128 bins) -> ONNX -> token IDs.

    Args:
        wav_path: Path to audio file.
        onnx_path: Path to speech_tokenizer_v3.onnx.
        provider: ONNX Runtime execution provider.

    Returns:
        Dict with 'speech_token' (list[int]) and 'speech_token_len' (int).
    """
    import whisper

    wav_16k = _load_wav(wav_path, 16000).squeeze(0)  # (num_samples,)

    # Whisper log-mel spectrogram: (n_mels=128, T)
    feat = whisper.log_mel_spectrogram(wav_16k, n_mels=128)

    providers = [provider] if provider != 'CUDAExecutionProvider' else (
        ['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    session = onnxruntime.InferenceSession(
        onnx_path,
        sess_options=onnxruntime.SessionOptions(),
        providers=providers,
    )

    # ONNX inputs: feats (1, 128, T), feats_length (1,)
    mel_batch = feat.unsqueeze(0).detach().cpu().numpy()  # (1, 128, T)
    mel_len = np.array([feat.shape[1]], dtype=np.int32)

    speech_token = session.run(
        None,
        {session.get_inputs()[0].name: mel_batch,
         session.get_inputs()[1].name: mel_len},
    )[0].flatten().tolist()

    logger.info(f'Speech tokens: {len(speech_token)} from {wav_path}')
    return {
        'speech_token': speech_token,
        'speech_token_len': len(speech_token),
    }


def extract_speaker_embedding(
    wav_path: str,
    onnx_path: str = CAMPPLUS_ONNX,
) -> Dict[str, object]:
    """Extract speaker embedding from audio via campplus.onnx.

    Pipeline: load at 16kHz -> Kaldi fbank (80 bins) -> CMVN -> ONNX -> 192-dim embedding.

    Args:
        wav_path: Path to audio file.
        onnx_path: Path to campplus.onnx.

    Returns:
        Dict with 'embedding' (list[float], dim=192).
    """
    import torchaudio.compliance.kaldi as kaldi

    wav_16k = _load_wav(wav_path, 16000)  # (1, num_samples)

    # Kaldi fbank: (num_frames, 80)
    feat = kaldi.fbank(wav_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
    # CMVN (mean subtraction)
    feat = feat - feat.mean(dim=0, keepdim=True)

    session = onnxruntime.InferenceSession(
        onnx_path,
        sess_options=onnxruntime.SessionOptions(),
        providers=['CPUExecutionProvider'],
    )

    # ONNX input: (1, num_frames, 80) -> output: (1, 192)
    embedding = session.run(
        None,
        {session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()},
    )[0].flatten().tolist()

    logger.info(f'Speaker embedding: dim={len(embedding)} from {wav_path}')
    return {
        'embedding': embedding,
    }


def extract_prompt_feat(
    wav_path: str,
    n_fft: int = 1920,
    num_mels: int = 80,
    sampling_rate: int = 24000,
    hop_size: int = 480,
    win_size: int = 1920,
    fmin: int = 0,
    fmax: Optional[int] = None,
) -> Dict[str, object]:
    """Extract prompt mel features from audio via Matcha-TTS mel_spectrogram.

    Pipeline: load at 24kHz -> matcha mel_spectrogram (80 bins, hop=480) -> log -> transpose.

    Args:
        wav_path: Path to audio file.
        n_fft: FFT size (CosyVoice3 default: 1920).
        num_mels: Number of mel bins (80).
        sampling_rate: Target sample rate (CosyVoice3: 24000).
        hop_size: Hop size in samples (CosyVoice3: 480).
        win_size: Window size (CosyVoice3: 1920).
        fmin: Minimum frequency.
        fmax: Maximum frequency (None = Nyquist).

    Returns:
        Dict with 'prompt_feat' (list[list[float]], shape (T, 80)) and 'prompt_feat_len' (int).
    """
    from matcha.utils.audio import mel_spectrogram as matcha_mel_spectrogram

    wav_24k = _load_wav(wav_path, sampling_rate)  # (1, num_samples)

    mel_fn = partial(
        matcha_mel_spectrogram,
        n_fft=n_fft, num_mels=num_mels, sampling_rate=sampling_rate,
        hop_size=hop_size, win_size=win_size, fmin=fmin, fmax=fmax,
        center=False,
    )
    mel = mel_fn(wav_24k)       # (1, 80, T)
    mel = mel.squeeze(0).T      # (T, 80)

    prompt_feat = mel.tolist()
    logger.info(f'Prompt feat: ({len(prompt_feat)}, {len(prompt_feat[0]) if prompt_feat else 0}) from {wav_path}')
    return {
        'prompt_feat': prompt_feat,
        'prompt_feat_len': len(prompt_feat),
    }


def extract_all(
    wav_path: str,
    token_mel_ratio: int = 2,
    speech_tokenizer_onnx: str = SPEECH_TOKENIZER_ONNX,
    campplus_onnx: str = CAMPPLUS_ONNX,
) -> Dict[str, object]:
    """Extract all features from a reference audio and align them.

    For CosyVoice2/3 (token_mel_ratio=2), prompt_feat and speech_token
    are truncated to maintain a strict 2:1 length ratio:
        token_len = min(prompt_feat_len // 2, speech_token_len)
        prompt_feat   -> first 2 * token_len frames
        speech_token  -> first token_len tokens

    Also returns full_speech_token (unaligned) for LLM conditioning.

    Args:
        wav_path: Path to reference audio file.
        token_mel_ratio: Mel-to-token ratio (CosyVoice2/3: 2, CosyVoice1: not enforced).
        speech_tokenizer_onnx: Path to speech tokenizer ONNX model.
        campplus_onnx: Path to campplus ONNX model.

    Returns:
        Dict with: embedding, prompt_token, prompt_feat,
                   prompt_token_len, prompt_feat_len,
                   full_speech_token (unaligned, for LLM).
    """
    speech_token_result = extract_speech_token(wav_path, onnx_path=speech_tokenizer_onnx)
    embedding_result = extract_speaker_embedding(wav_path, onnx_path=campplus_onnx)
    prompt_feat_result = extract_prompt_feat(wav_path)

    full_speech_token = speech_token_result['speech_token']
    prompt_feat = prompt_feat_result['prompt_feat']
    feat_len = prompt_feat_result['prompt_feat_len']

    if token_mel_ratio > 0:
        # CosyVoice2/3: enforce token_mel_ratio alignment
        token_len = min(feat_len // token_mel_ratio, len(full_speech_token))
        speech_token = full_speech_token[:token_len]
        prompt_feat = prompt_feat[:token_mel_ratio * token_len]
        feat_len = token_mel_ratio * token_len
        logger.info(f'Aligned: token_len={token_len}, feat_len={feat_len} (ratio={token_mel_ratio})')
    else:
        speech_token = full_speech_token

    return {
        'embedding': embedding_result['embedding'],
        'prompt_token': speech_token,
        'prompt_feat': prompt_feat,
        'prompt_token_len': len(speech_token),
        'prompt_feat_len': feat_len,
        'full_speech_token': full_speech_token,
    }
