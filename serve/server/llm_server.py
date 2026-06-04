"""
TensorRT-LLM based LLM Server for CosyVoice.

Provides an OpenAI-compatible API for batched LLM inference (text tokens -> semantic speech tokens).
Uses TensorRT-LLM's ModelRunnerCpp for inflight fused batching.

Usage:
    # Step 1: Build TRT engine (one-time, see serve/build_engine.py)
    python serve/build_engine.py \
        --model-dir ./pretrained_models/CosyVoice2-0.5B \
        --engine-dir ./trt_engines \
        --dtype bfloat16

    # Step 2: Launch server
    python -m serve.server.llm_server \
        --engine-dir ./trt_engines \
        --tokenizer-dir ./pretrained_models/cosyvoice2_llm \
        --max-batch-size 16 \
        --port 50000
"""

import time
from typing import List, Optional

import torch


class CosyVoiceLLMServer:
    """TensorRT-LLM backed LLM server with inflight fused batching.

    Loads the pre-built TRT engine and exposes an inference API.
    Multiple concurrent requests are automatically batched by TRT-LLM's
    inflight batching scheduler.
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

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.prompt_template = '<|sos|>{input_text}<|task_id|>'

        # Determine EOS token id
        eos_candidates = ['<|eos1|>', '<|eos|>']
        self.eos_token_id = None
        for tok_name in eos_candidates:
            if tok_name in self.tokenizer.get_vocab():
                self.eos_token_id = self.tokenizer.convert_tokens_to_ids(tok_name)
                break
        if self.eos_token_id is None:
            self.eos_token_id = self.tokenizer.eos_token_id
        print(f'EOS token id: {self.eos_token_id}')

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

    def generate(self, input_ids_list: List[torch.Tensor]) -> List[List[int]]:
        """Run batched LLM generation.

        Args:
            input_ids_list: List of 1-D int32 tensors, each is a token sequence.

        Returns:
            List of generated semantic token id lists (input tokens excluded).
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
        sequence_lengths = outputs['sequence_length']

        results = []
        for i in range(len(input_ids_list)):
            output_begin = input_lengths[i]
            output_end = sequence_lengths[i][0]
            generated = output_ids[i][0][output_begin:output_end].tolist()
            results.append(generated)

        return results

    def prepare_input(self, text: str, prompt_speech_tokens: Optional[List[int]] = None) -> torch.Tensor:
        """Prepare input_ids from text and optional prompt speech tokens.

        The format follows CosyVoice2's convention:
            <|sos|>{prompt_text}{text}<|task_id|>{speech_token_ids}
        """
        prompt = self.prompt_template.format(input_text=text)
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        if prompt_speech_tokens is not None:
            input_ids = input_ids + list(prompt_speech_tokens)

        return torch.tensor(input_ids, dtype=torch.int32)


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
        if server.eos_token_id in speech_tokens:
            speech_tokens = speech_tokens[:speech_tokens.index(server.eos_token_id)]

        return {
            'speech_tokens': speech_tokens,
            'latency_ms': latency,
        }

    @app.post('/v1/generate_batch')
    async def generate_batch(request: dict):
        items = request['items']
        temperature = request.get('temperature', server.temperature)
        top_k = request.get('top_k', server.top_k)
        top_p = request.get('top_p', server.top_p)

        orig_temp, orig_topk, orig_topp = server.temperature, server.top_k, server.top_p
        server.temperature = temperature
        server.top_k = top_k
        server.top_p = top_p

        input_ids_list = []
        for item in items:
            ids = server.prepare_input(item['text'], item.get('prompt_speech_tokens'))
            input_ids_list.append(ids)

        start = time.time()
        results = server.generate(input_ids_list)
        total_latency = (time.time() - start) * 1000

        server.temperature, server.top_k, server.top_p = orig_temp, orig_topk, orig_topp

        response_results = []
        for tokens in results:
            if server.eos_token_id in tokens:
                tokens = tokens[:tokens.index(server.eos_token_id)]
            response_results.append({'speech_tokens': tokens})

        return {
            'results': response_results,
            'total_latency_ms': total_latency,
            'batch_size': len(items),
        }

    @app.get('/health')
    async def health():
        return {'status': 'ok', 'max_batch_size': server.max_batch_size}

    return app


