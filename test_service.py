"""
CosyVoice Full Pipeline Smoke Test.

Tests the end-to-end inference: text -> LLM -> FM -> Vocoder -> WAV.

Uses zero-shot (few-shot) mode for voice cloning:
  - LLM receives prompt speech tokens + transcription from reference audio
  - FM/Vocoder receive speaker embedding + prompt features from reference audio

Prerequisites:
    - All three services running (or use --direct mode for in-process test):
        python serve/setup_server.py

    - Reference audio file available (default: Audio-Data/finetune-yuki-clara-voice)
    - Reference audio transcription in a .txt file next to the .wav file

Usage:
    # In-process test (no HTTP servers needed):
    PYTHONPATH=. python test_service.py --direct

    # HTTP test (requires running servers):
    PYTHONPATH=. python test_service.py

    # Custom text and output:
    PYTHONPATH=. python test_service.py --direct --text "Hello, welcome to the voice synthesis service." --output test_output.wav
"""

import argparse
import os
import sys
import time

# Ensure project root + CosyVoice + Matcha-TTS are importable
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_COSYVOICE_DIR = os.path.join(_REPO_ROOT, 'CosyVoice')
_MATCHA_DIR = os.path.join(_COSYVOICE_DIR, 'third_party', 'Matcha-TTS')
for p in [_REPO_ROOT, _COSYVOICE_DIR, _MATCHA_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from serve.paths import CHECKPOINTS_DIR


def load_prompt_transcription(prompt_wav_path: str) -> str:
    """Load transcription text from a .txt file next to the audio file.

    Falls back to empty string if no .txt file is found.
    Strips special markers like {F#...}, {M#...}, {NICKNAME} from game-voice datasets.
    """
    txt_path = os.path.splitext(prompt_wav_path)[0] + '.txt'
    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        # Strip game-voice markers like {F#big sister}{M#big brother} {NICKNAME}
        import re
        text = re.sub(r'\{[FM]#[^}]*\}', '', text)
        text = re.sub(r'\{NICKNAME\}', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ''


def extract_prompt_features(prompt_wav_path: str):
    """Extract speaker embedding, speech tokens, and mel features from a reference audio.

    Returns dict with: embedding, prompt_token, prompt_feat, full_speech_token, prompt_text
    - prompt_token/prompt_feat: aligned (2:1 ratio) for flow matching
    - full_speech_token: unaligned, full speech tokens for LLM conditioning
    - prompt_text: transcription text for LLM input
    """
    import numpy as np
    import onnxruntime
    import soundfile as sf
    import torch
    import torchaudio.compliance.kaldi as kaldi
    import torchaudio.transforms
    import whisper

    # Load reference audio via soundfile (avoids torchaudio/torchcodec dependency)
    wav_raw, sr_raw = sf.read(prompt_wav_path, dtype='float32')
    wav_tensor = torch.tensor(wav_raw, dtype=torch.float32)
    if wav_tensor.dim() > 1:
        wav_tensor = wav_tensor.mean(dim=1)

    # Resample to 16kHz for speech token + speaker embedding
    if sr_raw != 16000:
        wav_16k = torchaudio.transforms.Resample(sr_raw, 16000)(wav_tensor.unsqueeze(0)).squeeze(0)
    else:
        wav_16k = wav_tensor

    # --- Speaker embedding via campplus ---
    spk_feat = kaldi.fbank(wav_16k.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
    spk_feat = spk_feat - spk_feat.mean(dim=0, keepdim=True)

    campplus_path = os.path.join(CHECKPOINTS_DIR, 'campplus.onnx')
    campplus_session = onnxruntime.InferenceSession(
        campplus_path,
        sess_options=onnxruntime.SessionOptions(),
        providers=['CPUExecutionProvider'],
    )
    embedding = campplus_session.run(
        None,
        {campplus_session.get_inputs()[0].name: spk_feat.unsqueeze(dim=0).cpu().numpy()},
    )[0].flatten().tolist()
    print(f'  Speaker embedding: dim={len(embedding)}')

    # --- Speech tokens via speech_tokenizer_v3 ONNX ---
    mel_128 = whisper.log_mel_spectrogram(wav_16k, n_mels=128)  # (n_mels, T)
    mel_128_batch = mel_128.unsqueeze(0).detach().cpu().numpy()  # (1, n_mels, T)

    speech_tokenizer_path = os.path.join(CHECKPOINTS_DIR, 'speech_tokenizer_v3.onnx')
    speech_session = onnxruntime.InferenceSession(
        speech_tokenizer_path,
        sess_options=onnxruntime.SessionOptions(),
        providers=['CPUExecutionProvider'],
    )
    speech_token = speech_session.run(
        None,
        {
            speech_session.get_inputs()[0].name: mel_128_batch,
            speech_session.get_inputs()[1].name: np.array([mel_128.shape[1]], dtype=np.int32),
        },
    )[0].flatten().tolist()
    print(f'  Speech tokens: count={len(speech_token)}')

    # --- Prompt mel features (24kHz) for flow matching ---
    if sr_raw != 24000:
        wav_24k = torchaudio.transforms.Resample(sr_raw, 24000)(wav_tensor.unsqueeze(0))
    else:
        wav_24k = wav_tensor.unsqueeze(0)

    from functools import partial
    from matcha.utils.audio import mel_spectrogram as matcha_mel_spectrogram
    mel_fn = partial(matcha_mel_spectrogram,
                     n_fft=1920, num_mels=80, sampling_rate=24000,
                     hop_size=480, win_size=1920, fmin=0, fmax=None, center=False)
    prompt_mel = mel_fn(wav_24k)  # (1, 80, T)
    prompt_mel = prompt_mel.squeeze(0).T  # (T, 80)

    # Align prompt_feat and prompt_token to 2:1 ratio (CosyVoice3 requirement)
    token_len = min(int(prompt_mel.shape[0] / 2), len(speech_token))
    prompt_token = speech_token[:token_len]
    prompt_feat = prompt_mel[:2 * token_len].tolist()

    # Load transcription text for LLM input
    prompt_text = load_prompt_transcription(prompt_wav_path)
    print(f'  Prompt text: "{prompt_text[:60]}..."' if len(prompt_text) > 60 else f'  Prompt text: "{prompt_text}"')
    print(f'  Prompt feat: shape=({len(prompt_feat)}, {len(prompt_feat[0]) if prompt_feat else 0})')
    print(f'  Aligned: token_len={token_len}, feat_len={2 * token_len}')
    print(f'  Full speech tokens (for LLM): {len(speech_token)}')

    return {
        'embedding': embedding,
        'prompt_token': prompt_token,
        'prompt_feat': prompt_feat,
        'full_speech_token': speech_token,
        'prompt_text': prompt_text,
    }


def run_direct_test(text: str, prompt_wav_path: str, output_path: str):
    """Run full pipeline in-process (no HTTP)."""
    import numpy as np
    import soundfile as sf

    from serve.server.llm_server import CosyVoiceLLMServer
    from serve.server.fm_server import CosyVoiceFMServer
    from serve.server.vocoder_server import CosyVoiceVocoderServer

    engine_dir = os.path.join(CHECKPOINTS_DIR, 'trt_engines', 'trt_engines_bfloat16')
    tokenizer_dir = os.path.join(CHECKPOINTS_DIR, 'trt_engines', 'hf_merged_bfloat16')

    # Step 0: Extract prompt features for FM/Vocoder
    print('\n[Step 0] Extracting prompt features from reference audio ...')
    features = extract_prompt_features(prompt_wav_path)
    embedding = features['embedding']
    prompt_token = features['prompt_token']
    prompt_feat = features['prompt_feat']
    full_speech_token = features['full_speech_token']
    prompt_text = features['prompt_text']

    # Step 1: LLM inference (zero-shot mode: prompt speech tokens + transcription)
    print('\n[Step 1] Loading LLM server (TRT-LLM) ...')
    llm_server = CosyVoiceLLMServer(
        engine_dir=engine_dir,
        tokenizer_dir=tokenizer_dir,
        max_batch_size=1,
        max_output_len=2048,
    )

    # CosyVoice3 zero-shot format:
    #   User: <|sos|>You are a helpful assistant.<|endofprompt|>{prompt_text}{tts_text}<|task_id|>
    #   Assistant prefix: <|s_0|><|s_1|>... (prompt speech tokens from reference audio)
    #   Model continues generating speech tokens
    llm_text = f'You are a helpful assistant.<|endofprompt|>{prompt_text}{text}'
    input_ids = llm_server.prepare_input(llm_text, prompt_speech_tokens=full_speech_token)

    print(f'[Step 1] Running LLM inference (zero-shot mode with {len(full_speech_token)} prompt speech tokens) ...')
    t0 = time.time()
    results = llm_server.generate([input_ids])
    speech_tokens = results[0]
    llm_latency = (time.time() - t0) * 1000
    print(f'  Generated {len(speech_tokens)} speech tokens in {llm_latency:.0f}ms')

    if not speech_tokens:
        print('ERROR: LLM generated no speech tokens!')
        return False

    # Step 2: Flow matching inference
    print('\n[Step 2] Loading FM server ...')
    fm_server = CosyVoiceFMServer(model_dir=CHECKPOINTS_DIR)

    print('[Step 2] Running flow matching inference ...')
    t0 = time.time()
    fm_result = fm_server.generate(
        token=speech_tokens,
        prompt_token=prompt_token,
        prompt_feat=prompt_feat,
        embedding=embedding,
    )
    mel = fm_result['mel']
    fm_latency = fm_result['latency_ms']
    print(f'  Generated mel spectrogram: ({len(mel)}, {len(mel[0])}) in {fm_latency:.0f}ms')

    # Step 3: Vocoder inference
    print('\n[Step 3] Loading Vocoder server ...')
    vocoder_server = CosyVoiceVocoderServer(model_dir=CHECKPOINTS_DIR)

    print('[Step 3] Running vocoder inference ...')
    t0 = time.time()
    voc_result = vocoder_server.generate(mel=mel)
    waveform = voc_result['waveform']
    voc_latency = voc_result['latency_ms']
    print(f'  Generated waveform: {len(waveform)} samples in {voc_latency:.0f}ms')

    # Step 4: Save output
    wav_array = np.array(waveform, dtype=np.float32)
    if np.abs(wav_array).max() > 0:
        wav_array = wav_array / max(np.abs(wav_array).max(), 1.0) * 0.95
    sf.write(output_path, wav_array, 24000)

    duration = len(waveform) / 24000
    print(f'\n[DONE] Saved to {output_path} ({duration:.2f}s, {len(waveform)} samples)')
    print(f'  LLM: {llm_latency:.0f}ms | FM: {fm_latency:.0f}ms | Vocoder: {voc_latency:.0f}ms')

    return True


def run_http_test(text: str, prompt_wav_path: str, output_path: str,
                  llm_url: str, fm_url: str, vocoder_url: str):
    """Run full pipeline through HTTP APIs."""
    import requests
    import numpy as np
    import soundfile as sf

    # Step 0: Extract prompt features (still done locally since it's preprocessing)
    print('\n[Step 0] Extracting prompt features from reference audio ...')
    features = extract_prompt_features(prompt_wav_path)

    # Step 1: LLM inference (zero-shot mode with prompt speech tokens + transcription)
    llm_text = f'You are a helpful assistant.<|endofprompt|>{features["prompt_text"]}{text}'
    print(f'\n[Step 1] Calling LLM server ({llm_url}) with zero-shot mode ...')
    t0 = time.time()
    resp = requests.post(f'{llm_url}/v1/generate', json={
        'text': llm_text,
        'prompt_speech_tokens': features['full_speech_token'],
    })
    resp.raise_for_status()
    llm_result = resp.json()
    speech_tokens = llm_result['speech_tokens']
    llm_latency = (time.time() - t0) * 1000
    print(f'  Generated {len(speech_tokens)} speech tokens in {llm_latency:.0f}ms')

    if not speech_tokens:
        print('ERROR: LLM generated no speech tokens!')
        return False

    # Step 2: Flow matching inference
    print(f'\n[Step 2] Calling FM server ({fm_url}) ...')
    t0 = time.time()
    resp = requests.post(f'{fm_url}/v1/generate', json={
        'token': speech_tokens,
        'prompt_token': features['prompt_token'],
        'prompt_feat': features['prompt_feat'],
        'embedding': features['embedding'],
    })
    resp.raise_for_status()
    fm_result = resp.json()
    mel = fm_result['mel']
    fm_latency = (time.time() - t0) * 1000
    print(f'  Generated mel spectrogram: ({len(mel)}, {len(mel[0])}) in {fm_latency:.0f}ms')

    # Step 3: Vocoder inference
    print(f'\n[Step 3] Calling Vocoder server ({vocoder_url}) ...')
    t0 = time.time()
    resp = requests.post(f'{vocoder_url}/v1/generate', json={
        'mel': mel,
    })
    resp.raise_for_status()
    voc_result = resp.json()
    waveform = voc_result['waveform']
    voc_latency = (time.time() - t0) * 1000
    print(f'  Generated waveform: {len(waveform)} samples in {voc_latency:.0f}ms')

    # Step 4: Save output
    wav_array = np.array(waveform, dtype=np.float32)
    if np.abs(wav_array).max() > 0:
        wav_array = wav_array / max(np.abs(wav_array).max(), 1.0) * 0.95
    sf.write(output_path, wav_array, 24000)

    duration = len(waveform) / 24000
    print(f'\n[DONE] Saved to {output_path} ({duration:.2f}s, {len(waveform)} samples)')
    print(f'  LLM: {llm_latency:.0f}ms | FM: {fm_latency:.0f}ms | Vocoder: {voc_latency:.0f}ms')

    return True


def check_health(base_url: str, name: str) -> bool:
    """Check if a service is healthy."""
    import requests
    try:
        resp = requests.get(f'{base_url}/health', timeout=5)
        if resp.status_code == 200:
            print(f'  {name}: OK ({resp.json()})')
            return True
    except Exception as e:
        print(f'  {name}: FAILED ({e})')
    return False


def main():
    parser = argparse.ArgumentParser(description='CosyVoice Full Pipeline Smoke Test')
    parser.add_argument('--direct', action='store_true',
                        help='Run in-process test (no HTTP servers needed)')
    parser.add_argument('--text', type=str, default='你好，欢迎使用语音合成服务。',
                        help='Text to synthesize')
    parser.add_argument('--prompt-wav', type=str,
                        default='/root/Audio-Data/finetune-yuki-clara-voice/wavs/en_clara_00034.wav',
                        help='Reference audio for speaker voice')
    parser.add_argument('--output', type=str, default='smoke_test_output.wav',
                        help='Output WAV file path')

    # HTTP server URLs (for non-direct mode)
    parser.add_argument('--llm-url', type=str, default='http://localhost:50000')
    parser.add_argument('--fm-url', type=str, default='http://localhost:50001')
    parser.add_argument('--vocoder-url', type=str, default='http://localhost:50002')

    args = parser.parse_args()

    # Validate reference audio
    if not os.path.exists(args.prompt_wav):
        print(f'ERROR: Reference audio not found: {args.prompt_wav}')
        print('  Download CosyVoice assets or specify --prompt-wav')
        return 1

    print(f'CosyVoice Smoke Test')
    print(f'  Text: "{args.text}"')
    print(f'  Reference audio: {args.prompt_wav}')
    print(f'  Output: {args.output}')
    print(f'  Mode: {"direct (in-process)" if args.direct else "HTTP"}')

    try:
        if args.direct:
            success = run_direct_test(args.text, args.prompt_wav, args.output)
        else:
            # Check service health first
            print('\nChecking service health ...')
            all_healthy = True
            all_healthy &= check_health(args.llm_url, 'LLM')
            all_healthy &= check_health(args.fm_url, 'FM')
            all_healthy &= check_health(args.vocoder_url, 'Vocoder')

            if not all_healthy:
                print('\nSome services are not healthy. Start them with:')
                print('  python serve/setup_server.py')
                return 1

            success = run_http_test(
                args.text, args.prompt_wav, args.output,
                args.llm_url, args.fm_url, args.vocoder_url,
            )

        return 0 if success else 1
    except Exception as e:
        print(f'\nERROR: {e}')
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
