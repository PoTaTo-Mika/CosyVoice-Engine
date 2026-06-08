"""
CosyVoice Gateway Server.

Single-entry point for text-to-speech synthesis.
Internally chains LLM → FM → Vocoder via local DynamicBatchers,
so intermediate data (speech_tokens, mel) never leaves the machine.

Two ways to provide speaker reference:
  1. prompt_wav_path: Gateway extracts features from audio file
  2. Pre-extracted features: embedding + prompt_token + prompt_feat + full_speech_token
     (preferred for production — extract once, reuse for same character voice)

Usage:
    python -m serve.server.gateway \\
        --llm-batcher-url http://localhost:50100 \\
        --fm-batcher-url http://localhost:50101 \\
        --vocoder-batcher-url http://localhost:50102 \\
        --port 50200
"""

import argparse
import asyncio
import io
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

# Ensure CosyVoice + Matcha-TTS are importable for audio_encode
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_COSYVOICE_DIR = os.path.join(_REPO_ROOT, 'CosyVoice')
_MATCHA_DIR = os.path.join(_COSYVOICE_DIR, 'third_party', 'Matcha-TTS')
for _p in [_REPO_ROOT, _COSYVOICE_DIR, _MATCHA_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from serve.paths import CHECKPOINTS_DIR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wav_bytes_from_waveform(waveform: list, sample_rate: int = 24000) -> bytes:
    """Convert raw float32 waveform list to in-memory WAV bytes."""
    wav_array = np.array(waveform, dtype=np.float32)
    if np.abs(wav_array).max() > 0:
        wav_array = wav_array / max(np.abs(wav_array).max(), 1.0) * 0.95
    buf = io.BytesIO()
    sf.write(buf, wav_array, sample_rate, format='WAV')
    return buf.getvalue()


def load_prompt_transcription(prompt_wav_path: str) -> str:
    """Load transcription text from a .txt file next to the audio file.

    Falls back to empty string if no .txt file is found.
    Strips special markers like {F#...}, {M#...}, {NICKNAME}.
    """
    import re
    txt_path = os.path.splitext(prompt_wav_path)[0] + '.txt'
    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        text = re.sub(r'\{[FM]#[^}]*\}', '', text)
        text = re.sub(r'\{NICKNAME\}', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ''


# ---------------------------------------------------------------------------
# Gateway core
# ---------------------------------------------------------------------------

class GatewayServer:
    """Orchestrates the LLM → FM → Vocoder pipeline via local batchers.

    All three backend calls go over localhost, so intermediate data
    (speech_tokens, mel spectrograms) stays on-machine.
    """

    def __init__(
        self,
        llm_batcher_url: str,
        fm_batcher_url: str,
        vocoder_batcher_url: str,
        request_timeout: float = 60.0,
    ):
        self.llm_batcher_url = llm_batcher_url.rstrip('/')
        self.fm_batcher_url = fm_batcher_url.rstrip('/')
        self.vocoder_batcher_url = vocoder_batcher_url.rstrip('/')
        self.request_timeout = request_timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def start(self):
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.request_timeout))
        logger.info(
            f'Gateway started: '
            f'llm={self.llm_batcher_url}, '
            f'fm={self.fm_batcher_url}, '
            f'vocoder={self.vocoder_batcher_url}'
        )

    async def stop(self):
        if self._client:
            await self._client.aclose()

    async def _post(self, url: str, json_body: dict) -> dict:
        """POST to a backend service and return JSON."""
        resp = await self._client.post(url, json=json_body)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Feature extraction (optional, runs in thread pool)
    # ------------------------------------------------------------------

    async def extract_features(self, prompt_wav_path: str,
                               prompt_text: str = '') -> dict:
        """Extract prompt features from an audio file.

        Runs in a thread pool to avoid blocking the event loop.
        """
        from serve.server.inference.audio_encode import extract_all
        loop = asyncio.get_running_loop()
        features = await loop.run_in_executor(None, extract_all, prompt_wav_path)
        # If no prompt_text provided, try loading from companion .txt
        if not prompt_text:
            prompt_text = load_prompt_transcription(prompt_wav_path)
        features['prompt_text'] = prompt_text
        return features

    # ------------------------------------------------------------------
    # Full synthesis pipeline
    # ------------------------------------------------------------------

    async def synthesize(
        self,
        text: str,
        embedding: List[float],
        prompt_token: List[int],
        prompt_feat: List[List[float]],
        full_speech_token: List[int],
        prompt_text: str = '',
        streaming: bool = False,
        finalize: bool = True,
    ) -> Dict[str, Any]:
        """Run the full LLM → FM → Vocoder pipeline.

        Returns dict with 'waveform' (list[float]) and latency breakdown.
        """
        # Step 1: LLM — text → speech tokens
        llm_text = f'You are a helpful assistant.<|endofprompt|>{prompt_text}{text}'
        t0 = time.time()
        llm_result = await self._post(
            f'{self.llm_batcher_url}/v1/generate',
            {'text': llm_text, 'prompt_speech_tokens': full_speech_token},
        )
        speech_tokens = llm_result['speech_tokens']
        llm_latency = (time.time() - t0) * 1000

        if not speech_tokens:
            raise RuntimeError('LLM generated no speech tokens')

        logger.info(f'[Gateway] LLM: {len(speech_tokens)} speech tokens in {llm_latency:.0f}ms')

        # Step 2: FM — speech tokens → mel
        t0 = time.time()
        fm_result = await self._post(
            f'{self.fm_batcher_url}/v1/generate',
            {
                'token': speech_tokens,
                'prompt_token': prompt_token,
                'prompt_feat': prompt_feat,
                'embedding': embedding,
                'streaming': streaming,
                'finalize': finalize,
            },
        )
        mel = fm_result['mel']
        fm_latency = (time.time() - t0) * 1000

        logger.info(f'[Gateway] FM: mel ({len(mel)}, {len(mel[0]) if mel else 0}) in {fm_latency:.0f}ms')

        # Step 3: Vocoder — mel → waveform
        t0 = time.time()
        voc_result = await self._post(
            f'{self.vocoder_batcher_url}/v1/generate',
            {
                'mel': mel,
                'finalize': finalize,
            },
        )
        waveform = voc_result['waveform']
        voc_latency = (time.time() - t0) * 1000

        logger.info(f'[Gateway] Vocoder: {len(waveform)} samples in {voc_latency:.0f}ms')

        return {
            'waveform': waveform,
            'latency': {
                'llm_ms': llm_latency,
                'fm_ms': fm_latency,
                'vocoder_ms': voc_latency,
                'total_ms': llm_latency + fm_latency + voc_latency,
            },
            'speech_tokens_count': len(speech_tokens),
        }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def create_app(gateway: GatewayServer) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        await gateway.start()
        yield
        await gateway.stop()

    app = FastAPI(title='CosyVoice Gateway', lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'], allow_methods=['*'], allow_headers=['*'],
    )

    @app.post('/v1/synthesize')
    async def synthesize(request: Request):
        """Synthesize speech from text.

        Request body (JSON):
            text: str (required)
                Text to synthesize.

            --- Option A: reference audio file (Gateway extracts features) ---
            prompt_wav_path: str (optional)
                Path to reference audio for voice cloning.

            --- Option B: pre-extracted features (preferred for production) ---
            embedding: list[float] (optional, dim=192)
                Speaker embedding from campplus.
            prompt_token: list[int] (optional)
                Aligned prompt speech token IDs for flow matching.
            prompt_feat: list[list[float]] (optional, shape [T, 80])
                Aligned prompt mel features for flow matching.
            full_speech_token: list[int] (optional)
                Full (unaligned) speech tokens for LLM conditioning.

            --- Common ---
            prompt_text: str (optional)
                Transcription of reference audio (improves zero-shot quality).
            streaming: bool (optional, default False)
            finalize: bool (optional, default True)
            format: str (optional)
                "wav" (default): returns audio/wav binary.
                "json": returns JSON with waveform array + metadata.

        Response:
            format=wav → audio/wav binary with latency info in X-Latency-* headers.
            format=json → {"waveform": [...], "sample_rate": 24000, "latency": {...}}
        """
        body = await request.json()
        text = body['text']
        streaming = body.get('streaming', False)
        finalize = body.get('finalize', True)
        fmt = body.get('format', 'wav')

        # --- Resolve prompt features ---
        embedding = body.get('embedding')
        prompt_token = body.get('prompt_token')
        prompt_feat = body.get('prompt_feat')
        full_speech_token = body.get('full_speech_token')
        prompt_text = body.get('prompt_text', '')

        if embedding is None:
            # No pre-extracted features — extract from audio file
            prompt_wav_path = body.get('prompt_wav_path')
            if not prompt_wav_path:
                return {
                    'error': 'Must provide either prompt_wav_path or pre-extracted '
                             'features (embedding, prompt_token, prompt_feat, '
                             'full_speech_token)',
                }
            features = await gateway.extract_features(prompt_wav_path, prompt_text)
            embedding = features['embedding']
            prompt_token = features['prompt_token']
            prompt_feat = features['prompt_feat']
            full_speech_token = features['full_speech_token']
            prompt_text = features.get('prompt_text', prompt_text)

        # --- Run pipeline ---
        result = await gateway.synthesize(
            text=text,
            embedding=embedding,
            prompt_token=prompt_token,
            prompt_feat=prompt_feat,
            full_speech_token=full_speech_token,
            prompt_text=prompt_text,
            streaming=streaming,
            finalize=finalize,
        )

        # --- Format response ---
        if fmt == 'json':
            return {
                'waveform': result['waveform'],
                'sample_rate': 24000,
                'latency': result['latency'],
                'speech_tokens_count': result['speech_tokens_count'],
            }

        # Default: WAV binary
        wav_data = wav_bytes_from_waveform(result['waveform'], sample_rate=24000)
        return Response(
            content=wav_data,
            media_type='audio/wav',
            headers={
                'X-Latency-LLM-Ms': str(int(result['latency']['llm_ms'])),
                'X-Latency-FM-Ms': str(int(result['latency']['fm_ms'])),
                'X-Latency-Vocoder-Ms': str(int(result['latency']['vocoder_ms'])),
                'X-Latency-Total-Ms': str(int(result['latency']['total_ms'])),
                'X-Speech-Tokens-Count': str(result['speech_tokens_count']),
            },
        )

    @app.get('/health')
    async def health():
        """Check gateway and all backend health."""
        checks = {}
        all_ok = True
        for name, url in [('llm', gateway.llm_batcher_url),
                          ('fm', gateway.fm_batcher_url),
                          ('vocoder', gateway.vocoder_batcher_url)]:
            try:
                resp = await gateway._client.get(f'{url}/health', timeout=3.0)
                ok = resp.status_code == 200
            except Exception:
                ok = False
            checks[name] = 'ok' if ok else 'degraded'
            all_ok = all_ok and ok
        return {
            'status': 'ok' if all_ok else 'degraded',
            'backends': checks,
        }

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='CosyVoice Gateway Server')
    parser.add_argument('--llm-batcher-url', type=str, default='http://localhost:50100')
    parser.add_argument('--fm-batcher-url', type=str, default='http://localhost:50101')
    parser.add_argument('--vocoder-batcher-url', type=str, default='http://localhost:50102')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=50200)
    parser.add_argument('--request-timeout', type=float, default=60.0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
    )

    gateway = GatewayServer(
        llm_batcher_url=args.llm_batcher_url,
        fm_batcher_url=args.fm_batcher_url,
        vocoder_batcher_url=args.vocoder_batcher_url,
        request_timeout=args.request_timeout,
    )
    app = create_app(gateway)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
