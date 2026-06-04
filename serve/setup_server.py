"""
Launch the CosyVoice LLM Server.

Thin wrapper that imports the LLM server components and starts uvicorn.
All LLM logic (model loading, API routes) lives in serve.server.llm_server.

Usage:
    python serve/setup_server.py \
        --engine-dir ./trt_engines \
        --tokenizer-dir ./pretrained_models/cosyvoice2_llm \
        --max-batch-size 16 \
        --port 50000
"""

import argparse

from serve.server.llm_server import CosyVoiceLLMServer, create_app


def main():
    parser = argparse.ArgumentParser(description='CosyVoice LLM Server (TensorRT-LLM)')
    parser.add_argument('--engine-dir', type=str, required=True,
                        help='Path to TRT engine directory')
    parser.add_argument('--tokenizer-dir', type=str, required=True,
                        help='Path to HuggingFace tokenizer')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=50000)
    parser.add_argument('--max-batch-size', type=int, default=16)
    parser.add_argument('--max-output-len', type=int, default=2048)
    parser.add_argument('--max-input-len', type=int, default=512)
    parser.add_argument('--kv-cache-free-gpu-mem-fraction', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--repetition-penalty', type=float, default=1.1)

    args = parser.parse_args()

    import uvicorn
    server = CosyVoiceLLMServer(
        engine_dir=args.engine_dir,
        tokenizer_dir=args.tokenizer_dir,
        max_batch_size=args.max_batch_size,
        max_output_len=args.max_output_len,
        max_input_len=args.max_input_len,
        kv_cache_free_gpu_mem_fraction=args.kv_cache_free_gpu_mem_fraction,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    app = create_app(server)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
