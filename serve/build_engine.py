"""
Build TensorRT-LLM engine from HuggingFace checkpoint.

One-time offline step that:
  1. Converts HF weights -> TRT-LLM checkpoint
  2. Builds the TRT engine (via trtllm-build)

Requires the `tensorrt_llm` package.

Usage:
    python serve/build_engine.py \\
        --model-dir ./pretrained_models/CosyVoice2-0.5B \\
        --engine-dir ./trt_engines \\
        --dtype bfloat16 \\
        --max-batch-size 16 \\
        --max-num-tokens 32768
"""

import argparse
import os
import subprocess


def build_engine(model_dir: str, engine_dir: str, dtype: str = 'bfloat16',
                 max_batch_size: int = 16, max_num_tokens: int = 32768):
    """Convert HuggingFace checkpoint to TensorRT-LLM engine.

    Args:
        model_dir: Path to HuggingFace LLM model directory.
        engine_dir: Output directory for TRT weights and engine.
        dtype: Target precision (float16, bfloat16, float32).
        max_batch_size: Maximum batch size for the engine.
        max_num_tokens: Maximum number of tokens for the engine.
    """
    from tensorrt_llm.logger import logger

    trt_weights_dir = os.path.join(engine_dir, f'trt_weights_{dtype}')
    trt_engines_dir = os.path.join(engine_dir, f'trt_engines_{dtype}')

    # Step 1: convert checkpoint
    if not os.path.exists(trt_weights_dir) or not os.listdir(trt_weights_dir):
        logger.info('Converting HuggingFace checkpoint to TensorRT-LLM format...')
        from tensorrt_llm.mapping import Mapping
        from tensorrt_llm.models import QWenForCausalLM

        mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
        qwen = QWenForCausalLM.from_hugging_face(model_dir, dtype, mapping=mapping)
        os.makedirs(trt_weights_dir, exist_ok=True)
        qwen.save_checkpoint(trt_weights_dir, save_config=True)
        del qwen
        logger.info(f'Checkpoint saved to {trt_weights_dir}')
    else:
        logger.info(f'Checkpoint already exists at {trt_weights_dir}, skipping conversion.')

    # Step 2: build engine
    if not os.path.exists(trt_engines_dir) or not os.listdir(trt_engines_dir):
        logger.info('Building TensorRT engine...')
        os.makedirs(trt_engines_dir, exist_ok=True)

        cmd = (
            f'trtllm-build '
            f'--checkpoint_dir {trt_weights_dir} '
            f'--output_dir {trt_engines_dir} '
            f'--max_batch_size {max_batch_size} '
            f'--max_num_tokens {max_num_tokens} '
            f'--gemm_plugin {dtype}'
        )
        logger.info(f'Running: {cmd}')
        ret = subprocess.call(cmd, shell=True)
        if ret != 0:
            raise RuntimeError(f'trtllm-build failed with return code {ret}')
        logger.info(f'Engine saved to {trt_engines_dir}')
    else:
        logger.info(f'Engine already exists at {trt_engines_dir}, skipping build.')

    logger.info('Engine build complete.')


def main():
    parser = argparse.ArgumentParser(description='Build TensorRT-LLM engine from HuggingFace checkpoint')
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Path to HuggingFace LLM model dir (e.g. cosyvoice2_llm)')
    parser.add_argument('--engine-dir', type=str, required=True,
                        help='Output directory for TRT engine')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--max-batch-size', type=int, default=16)
    parser.add_argument('--max-num-tokens', type=int, default=32768)

    args = parser.parse_args()
    build_engine(args.model_dir, args.engine_dir, args.dtype,
                 args.max_batch_size, args.max_num_tokens)


if __name__ == '__main__':
    main()
