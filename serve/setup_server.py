"""
CosyVoice Service Launcher.

Starts one or all of the three inference services:
  - llm:   TensorRT-LLM based LLM server (text -> speech tokens)
  - fm:    Flow Matching server (speech tokens -> mel spectrogram)
  - vocoder: Vocoder server (mel spectrogram -> waveform)

Usage:
    # Start all services:
    python serve/setup_server.py

    # Start a specific service:
    python serve/setup_server.py --service llm
    python serve/setup_server.py --service fm
    python serve/setup_server.py --service vocoder
"""

import argparse
import os
import sys

from serve.paths import CHECKPOINTS_DIR

# Ensure CosyVoice + Matcha-TTS are importable for hyperpyyaml
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COSYVOICE_DIR = os.path.join(_REPO_ROOT, 'CosyVoice')
_MATCHA_DIR = os.path.join(_COSYVOICE_DIR, 'third_party', 'Matcha-TTS')

for p in [_REPO_ROOT, _COSYVOICE_DIR, _MATCHA_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)


def start_llm(args):
    from serve.server.llm_server import CosyVoiceLLMServer, create_app

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

    import uvicorn
    app = create_app(server)
    uvicorn.run(app, host=args.host, port=args.llm_port)


def start_fm(args):
    from serve.server.fm_server import CosyVoiceFMServer, create_app

    server = CosyVoiceFMServer(
        model_dir=args.model_dir,
        fp16=args.fp16,
        n_timesteps=args.n_timesteps,
        inference_cfg_rate=args.inference_cfg_rate,
    )

    import uvicorn
    app = create_app(server)
    uvicorn.run(app, host=args.host, port=args.fm_port)


def start_vocoder(args):
    from serve.server.vocoder_server import CosyVoiceVocoderServer, create_app

    server = CosyVoiceVocoderServer(
        model_dir=args.model_dir,
        fp16=args.fp16,
    )

    import uvicorn
    app = create_app(server)
    uvicorn.run(app, host=args.host, port=args.vocoder_port)


def start_all(args):
    """Start all three services as subprocesses."""
    import subprocess

    base_cmd = [sys.executable, '-u', os.path.abspath(__file__)]
    common = ['--host', args.host, '--model-dir', args.model_dir,
              '--engine-dir', args.engine_dir, '--tokenizer-dir', args.tokenizer_dir]

    procs = []

    def launch(service, port_flag, port):
        cmd = base_cmd + ['--service', service] + common + [port_flag, str(port)]
        print(f'Starting {service} on port {port} ...')
        p = subprocess.Popen(cmd, env=os.environ.copy())
        procs.append((service, p))

    launch('llm', '--llm-port', args.llm_port)
    launch('fm', '--fm-port', args.fm_port)
    launch('vocoder', '--vocoder-port', args.vocoder_port)

    print(f'\nAll services started:')
    print(f'  LLM:     http://{args.host}:{args.llm_port}')
    print(f'  FM:      http://{args.host}:{args.fm_port}')
    print(f'  Vocoder: http://{args.host}:{args.vocoder_port}')
    print('\nPress Ctrl+C to stop all services.')

    try:
        for _, p in procs:
            p.wait()
    except KeyboardInterrupt:
        print('\nShutting down all services ...')
        for name, p in procs:
            p.terminate()
        for name, p in procs:
            p.wait()
        print('All services stopped.')


def main():
    parser = argparse.ArgumentParser(description='CosyVoice Service Launcher')
    parser.add_argument('--service', type=str, choices=['llm', 'fm', 'vocoder', 'all'],
                        default='all', help='Which service to start (default: all)')
    parser.add_argument('--host', type=str, default='0.0.0.0')

    # Model paths
    parser.add_argument('--model-dir', type=str, default=CHECKPOINTS_DIR,
                        help='Path to model checkpoints directory')
    parser.add_argument('--engine-dir', type=str,
                        default=os.path.join(CHECKPOINTS_DIR, 'trt_engines', 'trt_engines_bfloat16'),
                        help='Path to TRT engine directory')
    parser.add_argument('--tokenizer-dir', type=str,
                        default=os.path.join(CHECKPOINTS_DIR, 'trt_engines', 'hf_merged_bfloat16'),
                        help='Path to merged HuggingFace tokenizer')

    # Port configuration
    parser.add_argument('--llm-port', type=int, default=50000)
    parser.add_argument('--fm-port', type=int, default=50001)
    parser.add_argument('--vocoder-port', type=int, default=50002)

    # LLM server params
    parser.add_argument('--max-batch-size', type=int, default=16)
    parser.add_argument('--max-output-len', type=int, default=2048)
    parser.add_argument('--max-input-len', type=int, default=512)
    parser.add_argument('--kv-cache-free-gpu-mem-fraction', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--repetition-penalty', type=float, default=1.1)

    # FM server params
    parser.add_argument('--fp16', action='store_true', help='Use FP16 for FM/Vocoder inference')
    parser.add_argument('--n-timesteps', type=int, default=10, help='Flow matching ODE steps')
    parser.add_argument('--inference-cfg-rate', type=float, default=None,
                        help='Override CFG rate for flow matching')

    args = parser.parse_args()

    if args.service == 'all':
        start_all(args)
    elif args.service == 'llm':
        start_llm(args)
    elif args.service == 'fm':
        start_fm(args)
    elif args.service == 'vocoder':
        start_vocoder(args)


if __name__ == '__main__':
    main()
