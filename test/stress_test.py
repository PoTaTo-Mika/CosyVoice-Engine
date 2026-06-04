"""
CosyVoice stress test with dynamic batching.

Full pipeline: request -> LLM (batcher) -> FM (direct) -> Vocoder (direct) -> save WAV.
Runs for 60s with random(1,16) concurrent requests.
16 slot IDs to cap disk usage (files overwrite).

Architecture:
    - LLM on GPU 0, accessed through dynamic batcher (inflight batching)
    - FM on GPU 1 (fp16), direct access (pipelining beats batching)
    - Vocoder on GPU 2, direct access (overlaps with FM on different GPU)

Prerequisites:
    - LLM server:    python serve/setup_server.py --service llm
    - FM server:     python serve/setup_server.py --service fm --fp16
    - Vocoder server: python serve/setup_server.py --service vocoder --vocoder-gpu 2
    - LLM batcher:   python -m serve.tool_func.dynamic_batch --backend-url http://localhost:50000 --port 60000
    - Place prompt.wav and prompt.txt (same basename, paired audio+transcript) in this directory

Usage:
    PYTHONPATH=. python test/stress_test.py
"""

import os
import random
import re
import sys
import time
import traceback

import numpy as np
import requests
import soundfile as sf

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COSYVOICE_DIR = os.path.join(_REPO_ROOT, 'CosyVoice')
_MATCHA_DIR = os.path.join(_COSYVOICE_DIR, 'third_party', 'Matcha-TTS')
for p in [_REPO_ROOT, _COSYVOICE_DIR, _MATCHA_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from serve.paths import CHECKPOINTS_DIR
from serve.server.inference.audio_encode import extract_all

# --- Config ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_WAV = os.path.join(SCRIPT_DIR, 'prompt.wav')
PROMPT_TXT = os.path.join(SCRIPT_DIR, 'prompt.txt')
TEST_TXT = os.path.join(SCRIPT_DIR, 'test.txt')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')

LLM_URL = 'http://localhost:60000'       # through dynamic batcher (inflight batching)
FM_URL = 'http://localhost:50001'        # direct — pipelining beats batching on FM
VOCODER_URL = 'http://localhost:50002'   # direct — Vocoder on GPU 2, overlaps with FM

DURATION_SEC = 60
MAX_SLOT = 16


def load_prompt_features():
    with open(PROMPT_TXT, 'r', encoding='utf-8') as f:
        prompt_text = f.read().strip()
    prompt_text = re.sub(r'\{[FM]#[^}]*\}', '', prompt_text)
    prompt_text = re.sub(r'\{NICKNAME\}', '', prompt_text)
    prompt_text = re.sub(r'\s+', ' ', prompt_text).strip()

    print('Extracting prompt features ...')
    features = extract_all(PROMPT_WAV)
    features['prompt_text'] = prompt_text
    print(f'  embedding: dim={len(features["embedding"])}')
    print(f'  prompt_token: {len(features["prompt_token"])}')
    print(f'  prompt_feat: ({len(features["prompt_feat"])}, {len(features["prompt_feat"][0]) if features["prompt_feat"] else 0})')
    print(f'  prompt_text: "{prompt_text}"')
    return features


def run_single_request(slot_id: int, tts_text: str, features: dict):
    start = time.time()
    try:
        # LLM
        llm_text = f'You are a helpful assistant.<|endofprompt|>{features["prompt_text"]}{tts_text}'
        t0 = time.time()
        resp = requests.post(f'{LLM_URL}/v1/generate', json={
            'text': llm_text,
            'prompt_speech_tokens': features['full_speech_token'],
        }, timeout=120)
        resp.raise_for_status()
        speech_tokens = resp.json()['speech_tokens']
        llm_ms = (time.time() - t0) * 1000

        if not speech_tokens:
            print(f'  [slot {slot_id}] LLM returned empty tokens')
            return None

        # FM
        t0 = time.time()
        resp = requests.post(f'{FM_URL}/v1/generate', json={
            'token': speech_tokens,
            'prompt_token': features['prompt_token'],
            'prompt_feat': features['prompt_feat'],
            'embedding': features['embedding'],
        }, timeout=120)
        resp.raise_for_status()
        mel = resp.json()['mel']
        fm_ms = (time.time() - t0) * 1000

        # Vocoder
        t0 = time.time()
        resp = requests.post(f'{VOCODER_URL}/v1/generate', json={
            'mel': mel,
        }, timeout=120)
        resp.raise_for_status()
        waveform = resp.json()['waveform']
        voc_ms = (time.time() - t0) * 1000

        # Save
        wav_array = np.array(waveform, dtype=np.float32)
        if np.abs(wav_array).max() > 0:
            wav_array = wav_array / max(np.abs(wav_array).max(), 1.0) * 0.95
        out_path = os.path.join(OUTPUT_DIR, f'slot_{slot_id:02d}.wav')
        sf.write(out_path, wav_array, 24000)

        total_ms = (time.time() - start) * 1000
        audio_dur = len(waveform) / 24000
        print(f'  [slot {slot_id}] done: LLM={llm_ms:.0f}ms FM={fm_ms:.0f}ms Voc={voc_ms:.0f}ms '
              f'audio={audio_dur:.1f}s total={total_ms:.0f}ms')
        return total_ms

    except Exception as e:
        print(f'  [slot {slot_id}] FAILED: {e}')
        traceback.print_exc()
        return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(TEST_TXT, 'r', encoding='utf-8') as f:
        tts_text = f.read().strip()

    print(f'=== CosyVoice Stress Test ===')
    print(f'  Text length: {len(tts_text)} chars')
    print(f'  Duration: {DURATION_SEC}s')
    print(f'  Output: {OUTPUT_DIR}/slot_00~{MAX_SLOT-1:02d}.wav')

    # Check services
    for name, url in [('LLM+Batcher', LLM_URL), ('FM', FM_URL), ('Vocoder', VOCODER_URL)]:
        try:
            r = requests.get(f'{url}/health', timeout=3)
            print(f'  {name}: OK')
        except Exception as e:
            print(f'  {name}: DOWN ({e})')
            return 1

    features = load_prompt_features()

    from concurrent.futures import ThreadPoolExecutor, as_completed

    end_time = time.time() + DURATION_SEC
    completed = 0
    errors = 0
    latencies = []
    slot_counter = 0

    print(f'\nStress test started, running for {DURATION_SEC}s ...\n')

    while time.time() < end_time:
        concurrency = random.randint(1, 16)
        remaining = end_time - time.time()
        if remaining <= 0:
            break

        print(f'[{time.strftime("%H:%M:%S")}] launching {concurrency} concurrent requests '
              f'({remaining:.1f}s remaining)')

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {}
            for _ in range(concurrency):
                slot_id = slot_counter % MAX_SLOT
                slot_counter += 1
                f = pool.submit(run_single_request, slot_id, tts_text, features)
                futures[f] = slot_id

            for f in as_completed(futures):
                result = f.result()
                if result is not None:
                    completed += 1
                    latencies.append(result)
                else:
                    errors += 1

    print(f'\n=== Stress Test Results ===')
    print(f'  Completed: {completed}')
    print(f'  Errors:    {errors}')
    if latencies:
        print(f'  Latency:   avg={np.mean(latencies):.0f}ms '
              f'p50={np.percentile(latencies, 50):.0f}ms '
              f'p95={np.percentile(latencies, 95):.0f}ms '
              f'max={np.max(latencies):.0f}ms')


if __name__ == '__main__':
    sys.exit(main())
