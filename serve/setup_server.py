"""
CosyVoice Service Launcher.

Starts inference services and their dynamic batchers:
  - llm:          TensorRT-LLM based LLM server (text -> speech tokens)
  - fm:           Flow Matching server (speech tokens -> mel spectrogram)
  - vocoder:      Vocoder server (mel spectrogram -> waveform)
  - llm-batcher:  Dynamic batcher in front of LLM
  - fm-batcher:   Dynamic batcher in front of FM
  - vocoder-batcher: Dynamic batcher in front of Vocoder

Usage:
    # Start all services (backends + batchers):
    python serve/setup_server.py

    # Start a specific service:
    python serve/setup_server.py --service llm
    python serve/setup_server.py --service fm
    python serve/setup_server.py --service vocoder
"""

import argparse
import os
import sys
import time

from serve.paths import CHECKPOINTS_DIR

# Ensure CosyVoice + Matcha-TTS are importable for hyperpyyaml
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COSYVOICE_DIR = os.path.join(_REPO_ROOT, 'CosyVoice')
_MATCHA_DIR = os.path.join(_COSYVOICE_DIR, 'third_party', 'Matcha-TTS')

for p in [_REPO_ROOT, _COSYVOICE_DIR, _MATCHA_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Default ports
LLM_BACKEND_PORT = 50000
FM_BACKEND_PORT = 50001
VOCODER_BACKEND_PORT = 50002
LLM_BATCHER_PORT = 50100
FM_BATCHER_PORT = 50101
VOCODER_BATCHER_PORT = 50102


def start_llm(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.llm_gpu)

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
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.fm_vocoder_gpu)

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
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.vocoder_gpu)

    from serve.server.vocoder_server import CosyVoiceVocoderServer, create_app

    server = CosyVoiceVocoderServer(
        model_dir=args.model_dir,
        fp16=args.fp16,
    )

    import uvicorn
    app = create_app(server)
    uvicorn.run(app, host=args.host, port=args.vocoder_port)


def start_batcher(args):
    """Start a single dynamic batcher as a standalone service."""
    from serve.tool_func.dynamic_batch import DynamicBatcher, create_app as create_batcher_app
    import uvicorn

    shared_keys = [k.strip() for k in args.shared_keys.split(',') if k.strip()] if args.shared_keys else []

    batcher = DynamicBatcher(
        backend_url=args.batcher_backend_url,
        max_batch_size=args.max_batch_size,
        scan_interval=args.scan_interval,
        max_wait_time=args.max_wait_time,
        request_timeout=args.request_timeout,
        shared_keys=shared_keys,
        service_name=args.batcher_service,
    )
    app = create_batcher_app(batcher)
    uvicorn.run(app, host=args.host, port=args.batcher_port)


def _wait_for_port(host: str, port: int, timeout: float = 30.0):
    """Block until a TCP port is accepting connections."""
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.3)
    return False


def start_all(args):
    """Start all backend services + dynamic batchers as subprocesses."""
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

    # --- Phase 1: Launch backend services ---
    launch('llm', '--llm-port', args.llm_port)
    launch('fm', '--fm-port', args.fm_port)
    launch('vocoder', '--vocoder-port', args.vocoder_port)

    print('Waiting for backend services to be ready ...')
    for name, port in [('LLM', args.llm_port), ('FM', args.fm_port), ('Vocoder', args.vocoder_port)]:
        ok = _wait_for_port(args.host, port, timeout=60.0)
        status = 'READY' if ok else 'TIMEOUT'
        print(f'  {name} (port {port}): {status}')

    # --- Phase 2: Launch dynamic batchers ---
    batcher_script = os.path.join(_REPO_ROOT, 'serve', 'tool_func', 'dynamic_batch.py')

    batcher_configs = [
        ('llm-batcher', args.llm_batcher_port, args.llm_port, '', 'llm'),
        ('fm-batcher', args.fm_batcher_port, args.fm_port, 'streaming,finalize', 'fm'),
        ('vocoder-batcher', args.vocoder_batcher_port, args.vocoder_port, 'finalize', 'vocoder'),
    ]

    for name, batcher_port, backend_port, shared_keys, service in batcher_configs:
        cmd = [
            sys.executable, '-u', batcher_script,
            '--backend-url', f'http://{args.host}:{backend_port}',
            '--port', str(batcher_port),
            '--host', args.host,
            '--max-batch-size', str(args.max_batch_size),
            '--service', service,
        ]
        if shared_keys:
            cmd += ['--shared-keys', shared_keys]
        print(f'Starting {name} on port {batcher_port} -> backend :{backend_port} ...')
        p = subprocess.Popen(cmd, env=os.environ.copy())
        procs.append((name, p))

    print(f'\nAll services started:')
    print(f'  LLM backend:     http://{args.host}:{args.llm_port}')
    print(f'  FM backend:      http://{args.host}:{args.fm_port}')
    print(f'  Vocoder backend: http://{args.host}:{args.vocoder_port}')
    print(f'  LLM batcher:     http://{args.host}:{args.llm_batcher_port}  (shared_keys: none)')
    print(f'  FM batcher:      http://{args.host}:{args.fm_batcher_port}  (shared_keys: streaming,finalize)')
    print(f'  Vocoder batcher: http://{args.host}:{args.vocoder_batcher_port}  (shared_keys: finalize)')
    print('\n  ↳ Clients should send requests to the BATCHER ports for dynamic batching.')
    print('Press Ctrl+C to stop all services.')

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
    parser.add_argument('--service', type=str,
                        choices=['llm', 'fm', 'vocoder', 'batcher', 'all'],
                        default='all', help='Which service to start (default: all)')
    parser.add_argument('--host', type=str, default='0.0.0.0')

    # GPU assignment
    parser.add_argument('--llm-gpu', type=int, default=0,
                        help='GPU device ID for LLM service')
    parser.add_argument('--fm-vocoder-gpu', type=int, default=1,
                        help='GPU device ID for FM service')
    parser.add_argument('--vocoder-gpu', type=int, default=2,
                        help='GPU device ID for Vocoder service')

    # Model paths
    parser.add_argument('--model-dir', type=str, default=CHECKPOINTS_DIR,
                        help='Path to model checkpoints directory')
    parser.add_argument('--engine-dir', type=str,
                        default=os.path.join(CHECKPOINTS_DIR, 'trt_engines', 'trt_engines_bfloat16'),
                        help='Path to TRT engine directory')
    parser.add_argument('--tokenizer-dir', type=str,
                        default=os.path.join(CHECKPOINTS_DIR, 'trt_engines', 'hf_merged_bfloat16'),
                        help='Path to merged HuggingFace tokenizer')

    # Backend port configuration
    parser.add_argument('--llm-port', type=int, default=LLM_BACKEND_PORT)
    parser.add_argument('--fm-port', type=int, default=FM_BACKEND_PORT)
    parser.add_argument('--vocoder-port', type=int, default=VOCODER_BACKEND_PORT)

    # Batcher port configuration
    parser.add_argument('--llm-batcher-port', type=int, default=LLM_BATCHER_PORT)
    parser.add_argument('--fm-batcher-port', type=int, default=FM_BATCHER_PORT)
    parser.add_argument('--vocoder-batcher-port', type=int, default=VOCODER_BATCHER_PORT)

    # Shared inference params
    parser.add_argument('--max-batch-size', type=int, default=16)

    # LLM server params
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

    # Batcher params (used with --service batcher)
    parser.add_argument('--batcher-backend-url', type=str, default='http://localhost:50000',
                        help='Backend URL for standalone batcher')
    parser.add_argument('--batcher-port', type=int, default=50100,
                        help='Listen port for standalone batcher')
    parser.add_argument('--batcher-service', type=str, default='unknown',
                        help='Service name label for standalone batcher')
    parser.add_argument('--shared-keys', type=str, default='',
                        help='Comma-separated shared keys for standalone batcher')
    parser.add_argument('--scan-interval', type=float, default=0.2,
                        help='Queue scan interval in seconds')
    parser.add_argument('--max-wait-time', type=float, default=0.6,
                        help='Max time a request waits before partial batch dispatch')
    parser.add_argument('--request-timeout', type=float, default=60.0,
                        help='Timeout for individual request (seconds)')

    args = parser.parse_args()

    if args.service == 'all':
        start_all(args)
    elif args.service == 'llm':
        start_llm(args)
    elif args.service == 'fm':
        start_fm(args)
    elif args.service == 'vocoder':
        start_vocoder(args)
    elif args.service == 'batcher':
        start_batcher(args)


if __name__ == '__main__':
    main()
