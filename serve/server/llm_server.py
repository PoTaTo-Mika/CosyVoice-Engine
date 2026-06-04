"""
TensorRT-LLM based LLM Server for CosyVoice3.

Provides an HTTP API for batched LLM inference (text -> semantic speech tokens).
Uses TensorRT-LLM's ModelRunnerCpp for inflight fused batching.

Requires a pre-built TRT engine from serve/build_engine.py, which produces
a merged HuggingFace model with speech tokens added to the vocabulary.

Usage:
    python serve/setup_server.py \
        --port 50000
"""

import json
import os
import time
from typing import List, Optional

import torch

from serve.paths import CHECKPOINTS_DIR


class CosyVoiceLLMServer:
    """TensorRT-LLM backed LLM server with inflight fused batching.

    Loads the pre-built TRT engine and the merged HF tokenizer.
    Speech token IDs in the merged vocab are offset by speech_token_offset
    (stored in cosyvoice3_metadata.json).
    """

    def __init__(self, engine_dir: str, tokenizer_dir: str,
                 max_batch_size: int = 16,
                 max_output_len: int = 2048,
                 max_input_len: int = 512,
                 kv_cache_free_gpu_mem_fraction: float = 0.5,
                 temperature: float = 0.8,
                 top_k: int = 50,
                 top_p: float = 0.95,
                 repetition_penalty: float = 1.1):
        import tensorrt_llm
        from tensorrt_llm.runtime import ModelRunnerCpp
        from transformers import AutoTokenizer

        self.device = torch.device('cuda')
        self.max_batch_size = max_batch_size
        self.max_output_len = max_output_len
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

        # Load tokenizer from merged HF model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

        # Load metadata for speech token offset
        metadata_path = os.path.join(tokenizer_dir, 'cosyvoice3_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.speech_token_offset = metadata['speech_token_offset']
            self.base_speech_token_size = metadata['base_speech_token_size']
        else:
            # Fallback: compute from tokenizer vocab
            self.speech_token_offset = self._infer_speech_offset()
            self.base_speech_token_size = 6561

        # EOS: speech eos = offset + base_speech_token_size + 1
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids('<|eos1|>')
        if self.eos_token_id is None:
            self.eos_token_id = self.speech_token_offset + self.base_speech_token_size + 1

        print(f'speech_token_offset={self.speech_token_offset}, '
              f'eos_token_id={self.eos_token_id}, '
              f'tokenizer vocab size={len(self.tokenizer)}')

        # Load TRT engine
        runtime_rank = tensorrt_llm.mpi_rank()
        self.runner = ModelRunnerCpp.from_dir(
            engine_dir=engine_dir,
            rank=runtime_rank,
            max_output_len=max_output_len,
            enable_context_fmha_fp32_acc=False,
            max_batch_size=max_batch_size,
            max_input_len=max_input_len,
            kv_cache_free_gpu_memory_fraction=kv_cache_free_gpu_mem_fraction,
            cuda_graph_mode=False,
            gather_generation_logits=False,
        )
        print(f'TRT-LLM engine loaded from {engine_dir}, max_batch_size={max_batch_size}')

    def _infer_speech_offset(self) -> int:
        """Fallback: find <|s_0|> in tokenizer to determine offset."""
        tid = self.tokenizer.convert_tokens_to_ids('<|s_0|>')
        if tid is not None:
            return tid
        raise ValueError('Cannot determine speech_token_offset from tokenizer. '
                         'Ensure cosyvoice3_metadata.json exists in tokenizer_dir.')

    def prepare_input(self, text: str, prompt_speech_tokens: Optional[List[int]] = None) -> torch.Tensor:
        """Prepare input_ids from text and optional prompt speech tokens.

        For zero-shot/few-shot voice cloning:
          - text should be: "You are a helpful assistant.<|endofprompt|>{transcription}{tts_text}"
          - prompt_speech_tokens should be the full speech tokens from the reference audio

        For instruct mode (no voice cloning):
          - text should be: "You are a helpful assistant.<|endofprompt|>{tts_text}"
          - prompt_speech_tokens should be None

        Speech tokens are raw IDs (0-6560) that get offset to merged vocab IDs.
        """
        # Format: <|sos|>{text}<|task_id|>{prompt_speech_tokens}
        # Use chat template for consistent tokenization
        chat = [{"role": "user", "content": text}]
        if prompt_speech_tokens:
            # Convert raw speech IDs to merged vocab IDs
            prompt_str = ''.join(f'<|s_{t}|>' for t in prompt_speech_tokens)
            chat.append({"role": "assistant", "content": prompt_str})
            input_ids = self.tokenizer.apply_chat_template(
                chat, tokenize=True, return_tensors='pt',
                continue_final_message=True)
        else:
            input_ids = self.tokenizer.apply_chat_template(
                chat, tokenize=True, return_tensors='pt')

        return input_ids.squeeze(0).to(torch.int32)

    def generate(self, input_ids_list: List[torch.Tensor]) -> List[List[int]]:
        """Run batched LLM generation.

        Returns raw speech token IDs (0-6560), with EOS/special tokens removed.
        """
        input_lengths = [x.size(0) for x in input_ids_list]

        outputs = self.runner.generate(
            batch_input_ids=input_ids_list,
            max_new_tokens=self.max_output_len,
            end_id=self.eos_token_id,
            pad_id=self.eos_token_id,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            num_return_sequences=1,
            streaming=False,
            output_sequence_lengths=True,
            output_generation_logits=False,
            return_dict=True,
        )
        torch.cuda.synchronize()

        output_ids = outputs['output_ids']
        sequence_lengths = outputs['sequence_lengths']

        results = []
        for i in range(len(input_ids_list)):
            output_begin = input_lengths[i]
            output_end = sequence_lengths[i][0]
            generated = output_ids[i][0][output_begin:output_end].tolist()

            # Convert merged vocab IDs back to raw speech token IDs
            speech_ids = []
            for tid in generated:
                if tid >= self.speech_token_offset:
                    raw_id = tid - self.speech_token_offset
                    if raw_id < self.base_speech_token_size:
                        speech_ids.append(raw_id)
                    # else: special token (sos/eos/task_id/fill), skip
            results.append(speech_ids)

        return results


def create_app(server: CosyVoiceLLMServer):
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title='CosyVoice LLM Server')
    app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

    @app.post('/v1/generate')
    async def generate(request: dict):
        text = request['text']
        prompt_speech_tokens = request.get('prompt_speech_tokens')

        input_ids = server.prepare_input(text, prompt_speech_tokens)

        start = time.time()
        results = server.generate([input_ids])
        latency = (time.time() - start) * 1000

        speech_tokens = results[0]
        return {
            'speech_tokens': speech_tokens,
            'latency_ms': latency,
        }

    @app.post('/v1/generate_batch')
    async def generate_batch(request: dict):
        items = request['items']

        input_ids_list = []
        for item in items:
            ids = server.prepare_input(item['text'], item.get('prompt_speech_tokens'))
            input_ids_list.append(ids)

        start = time.time()
        results = server.generate(input_ids_list)
        total_latency = (time.time() - start) * 1000

        response_results = [{'speech_tokens': tokens} for tokens in results]

        return {
            'results': response_results,
            'total_latency_ms': total_latency,
            'batch_size': len(items),
        }

    @app.get('/health')
    async def health():
        return {
            'status': 'ok',
            'max_batch_size': server.max_batch_size,
            'speech_token_offset': server.speech_token_offset,
        }

    return app
