"""
Flow Matching Server for CosyVoice3.

Provides an HTTP API for flow matching inference (speech tokens + speaker embedding -> mel spectrogram).
Uses the FlowMatchingEngine from serve.server.inference.fm_infer.

Usage:
    python -m serve.server.fm_server \
        --model-dir ./pretrained_models/CosyVoice2-0.5B \
        --port 50001
"""

import time
from typing import Optional

import torch

from serve.paths import CHECKPOINTS_DIR
from serve.server.inference.fm_infer import FlowMatchingEngine


class CosyVoiceFMServer:
    """Flow matching server wrapping FlowMatchingEngine."""

    def __init__(self, model_dir: str = CHECKPOINTS_DIR, fp16: bool = False,
                 device: Optional[str] = None, n_timesteps: int = 10,
                 inference_cfg_rate: Optional[float] = None):
        self.engine = FlowMatchingEngine(
            model_dir=model_dir,
            fp16=fp16,
            device=device,
            n_timesteps=n_timesteps,
            inference_cfg_rate=inference_cfg_rate,
        )
        self.token_mel_ratio = self.engine.token_mel_ratio
        self.pre_lookahead_len = self.engine.pre_lookahead_len
        self.sample_rate = self.engine.sample_rate

    def generate(self, token: list, prompt_token: list,
                 prompt_feat: list, embedding: list,
                 streaming: bool = False, finalize: bool = True):
        """Run single flow matching inference.

        Args:
            token: Speech token ids, list of int.
            prompt_token: Prompt speech token ids, list of int.
            prompt_feat: Prompt mel features, list of list of float (Tm, 80).
            embedding: Speaker embedding, list of float (192,).

        Returns:
            dict with 'mel' (list of list of float) and 'latency_ms'.
        """
        token_t = torch.tensor([token], dtype=torch.int32)
        prompt_token_t = torch.tensor([prompt_token], dtype=torch.int32)
        prompt_feat_t = torch.tensor([prompt_feat], dtype=torch.float32)
        embedding_t = torch.tensor([embedding], dtype=torch.float32)

        start = time.time()
        mel, _ = self.engine.inference(
            token=token_t,
            prompt_token=prompt_token_t,
            prompt_feat=prompt_feat_t,
            embedding=embedding_t,
            streaming=streaming,
            finalize=finalize,
        )
        latency = (time.time() - start) * 1000

        # mel: (1, 80, T_mel) -> (T_mel, 80) list
        mel_list = mel[0].T.cpu().tolist()
        return {'mel': mel_list, 'latency_ms': latency}

    def generate_batch(self, items: list, streaming: bool = False,
                       finalize: bool = True):
        """Run batched flow matching inference.

        Args:
            items: List of dicts, each with token, prompt_token, prompt_feat, embedding.
            streaming: Enable streaming mode.
            finalize: Whether this is the final chunk.

        Returns:
            dict with 'results', 'total_latency_ms', 'batch_size'.
        """
        input_ids_list = []
        for item in items:
            token_t = torch.tensor([item['token']], dtype=torch.int32)
            prompt_token_t = torch.tensor([item['prompt_token']], dtype=torch.int32)
            prompt_feat_t = torch.tensor([item['prompt_feat']], dtype=torch.float32)
            embedding_t = torch.tensor([item['embedding']], dtype=torch.float32)
            input_ids_list.append({
                'token': token_t,
                'prompt_token': prompt_token_t,
                'prompt_feat': prompt_feat_t,
                'embedding': embedding_t,
            })

        start = time.time()
        results = self.engine.inference_batch(input_ids_list, streaming=streaming, finalize=finalize)
        total_latency = (time.time() - start) * 1000

        response_results = []
        for mel, _ in results:
            mel_list = mel[0].T.cpu().tolist()
            response_results.append({'mel': mel_list})

        return {
            'results': response_results,
            'total_latency_ms': total_latency,
            'batch_size': len(items),
        }


def create_app(server: CosyVoiceFMServer):
    import asyncio
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title='CosyVoice Flow Matching Server')
    app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

    @app.post('/v1/generate')
    async def generate(request: dict):
        return await asyncio.to_thread(
            server.generate,
            token=request['token'],
            prompt_token=request['prompt_token'],
            prompt_feat=request['prompt_feat'],
            embedding=request['embedding'],
            streaming=request.get('streaming', False),
            finalize=request.get('finalize', True),
        )

    @app.post('/v1/generate_batch')
    async def generate_batch(request: dict):
        return await asyncio.to_thread(
            server.generate_batch,
            items=request['items'],
            streaming=request.get('streaming', False),
            finalize=request.get('finalize', True),
        )

    @app.get('/health')
    async def health():
        return {
            'status': 'ok',
            'sample_rate': server.sample_rate,
            'token_mel_ratio': server.token_mel_ratio,
            'pre_lookahead_len': server.pre_lookahead_len,
        }

    return app
