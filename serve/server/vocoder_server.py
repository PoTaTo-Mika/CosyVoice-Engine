"""
Vocoder Server for CosyVoice3.

Converts mel spectrogram -> waveform via CausalHiFTGenerator.
Self-contained: model loading and inference are in this file.

Usage:
    python -m serve.server.vocoder_server \
        --model-dir ./pretrained_models/CosyVoice2-0.5B \
        --port 50002
"""

import os
import time
from typing import Optional

import torch

from hyperpyyaml import load_hyperpyyaml


class CosyVoiceVocoderServer:
    """Vocoder server wrapping CausalHiFTGenerator."""

    def __init__(self, model_dir: str, fp16: bool = False,
                 device: Optional[str] = None):
        self.model_dir = model_dir
        self.fp16 = fp16
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

        self._load_model()
        self.sample_rate = self.hift.sampling_rate

    def _load_model(self):
        yaml_path = os.path.join(self.model_dir, 'cosyvoice3.yaml')
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f'cosyvoice3.yaml not found in {self.model_dir}')

        qwen_path = os.path.join(self.model_dir, 'CosyVoice-BlankEN')
        overrides = {'qwen_pretrain_path': qwen_path}

        with open(yaml_path, 'r') as f:
            configs = load_hyperpyyaml(f, overrides=overrides)

        self.hift = configs['hift']

        # hift.pt weights have 'generator.' prefix from HiFiGan wrapper
        hift_pt_path = os.path.join(self.model_dir, 'hift.pt')
        if not os.path.exists(hift_pt_path):
            raise FileNotFoundError(f'hift.pt not found in {self.model_dir}')

        state_dict = {
            k.replace('generator.', ''): v
            for k, v in torch.load(hift_pt_path, map_location=self.device, weights_only=True).items()
        }
        self.hift.load_state_dict(state_dict, strict=True)
        self.hift.to(self.device).eval()

        del configs

    @torch.inference_mode()
    def generate(self, mel: list, finalize: bool = True):
        """Run vocoder inference: mel spectrogram -> waveform.

        Args:
            mel: Mel spectrogram, list of list of float (T_mel, 80).
            finalize: Whether this is the final chunk (affects causal padding).

        Returns:
            dict with 'waveform' (list of float) and 'latency_ms'.
        """
        # (T_mel, 80) -> (1, 80, T_mel)
        mel_t = torch.tensor([mel], dtype=torch.float32, device=self.device).transpose(1, 2)

        start = time.time()
        with torch.cuda.amp.autocast(self.fp16):
            speech, _ = self.hift.inference(speech_feat=mel_t, finalize=finalize)
        latency = (time.time() - start) * 1000

        # speech: (1, S) or (1, 1, S)
        waveform = speech.squeeze().cpu().tolist()
        return {'waveform': waveform, 'latency_ms': latency}

    @torch.inference_mode()
    def generate_batch(self, items: list, finalize: bool = True):
        """Run batched vocoder inference with padding.

        Items with varying mel lengths are padded to the same size
        within the batch, then run in a single forward pass.

        Args:
            items: List of dicts, each with 'mel' (list of list of float, T_mel_i, 80).
            finalize: Whether this is the final chunk.

        Returns:
            dict with 'results', 'total_latency_ms', 'batch_size'.
        """
        if len(items) == 0:
            return {'results': [], 'total_latency_ms': 0, 'batch_size': 0}

        # Convert to tensors and find max mel length
        mel_tensors = []
        for item in items:
            # (T_mel, 80) -> (1, 80, T_mel)
            mel_t = torch.tensor([item['mel']], dtype=torch.float32, device=self.device).transpose(1, 2)
            mel_tensors.append(mel_t)

        max_mel_len = max(t.shape[2] for t in mel_tensors)

        # Pad to same length along time axis
        padded = []
        mel_len_list = []
        for mel_t in mel_tensors:
            T = mel_t.shape[2]
            mel_len_list.append(T)
            if T < max_mel_len:
                pad = torch.zeros(1, 80, max_mel_len - T, dtype=torch.float32, device=self.device)
                padded.append(torch.cat([mel_t, pad], dim=2))
            else:
                padded.append(mel_t)

        # Stack into (B, 80, T_mel)
        mel_batch = torch.cat(padded, dim=0)

        start = time.time()
        with torch.cuda.amp.autocast(self.fp16):
            speech, _ = self.hift.inference(speech_feat=mel_batch, finalize=finalize)
        latency = (time.time() - start) * 1000

        # Un-batch: trim each waveform to its actual length
        # CausalHiFTGenerator output: (B, S) where S = T_mel * upsample_total
        # upsample_total = prod(upsample_rates) * hop_len = 8*5*3*4 = 480
        upsample_factor = int(self.hift.upsample_rates[0])
        for r in self.hift.upsample_rates[1:]:
            upsample_factor *= r
        upsample_factor *= self.hift.istft_params['hop_len']

        results = []
        for i in range(len(items)):
            expected_len = mel_len_list[i] * upsample_factor
            waveform = speech[i, :expected_len].cpu().tolist()
            results.append({'waveform': waveform})

        return {
            'results': results,
            'total_latency_ms': latency,
            'batch_size': len(items),
        }


def create_app(server: CosyVoiceVocoderServer):
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title='CosyVoice Vocoder Server')
    app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

    @app.post('/v1/generate')
    async def generate(request: dict):
        return server.generate(
            mel=request['mel'],
            finalize=request.get('finalize', True),
        )

    @app.post('/v1/generate_batch')
    async def generate_batch(request: dict):
        return server.generate_batch(
            items=request['items'],
            finalize=request.get('finalize', True),
        )

    @app.get('/health')
    async def health():
        return {
            'status': 'ok',
            'sample_rate': server.sample_rate,
        }

    return app


if __name__ == '__main__':
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description='CosyVoice Vocoder Server')
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Path to pretrained model directory containing cosyvoice3.yaml and hift.pt')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=50002)
    parser.add_argument('--fp16', action='store_true')
    args = parser.parse_args()

    server = CosyVoiceVocoderServer(model_dir=args.model_dir, fp16=args.fp16)
    app = create_app(server)
    uvicorn.run(app, host=args.host, port=args.port)
