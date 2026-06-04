"""
Build TensorRT-LLM engine for CosyVoice3 LLM.

Three-step pipeline:
  1. Convert CosyVoice3 LLM to merged HuggingFace format
     (speech_embedding -> embed_tokens, llm_decoder -> lm_head, extended vocab)
  2. Convert HuggingFace checkpoint to TensorRT-LLM weights
  3. Build the TRT engine via trtllm-build

Requires: tensorrt_llm, transformers

Usage:
    # Full pipeline (all steps):
    python serve/build_engine.py

    # Skip step 1 if HF model already exists:
    python serve/build_engine.py --skip-hf-convert

    # Only run specific step:
    python serve/build_engine.py --step 2
    python serve/build_engine.py --step 3
"""

import argparse
import json
import os
import subprocess
import sys

from serve.paths import CHECKPOINTS_DIR


def step1_convert_to_hf(model_dir: str, output_dir: str, dtype: str = 'bfloat16'):
    """Step 1: Convert CosyVoice3 LLM to HuggingFace format with merged embeddings.

    Loads the Qwen2 model from CosyVoice-BlankEN, loads speech_embedding and
    llm_decoder weights directly from llm.pt (bypassing hyperpyyaml), merges
    them into the Qwen2 model, extends the tokenizer with speech tokens, and
    saves a standard Qwen2ForCausalLM that TensorRT-LLM can consume.
    """
    import torch
    from transformers import AutoTokenizer, Qwen2ForCausalLM

    qwen_path = os.path.join(model_dir, 'CosyVoice-BlankEN')
    llm_pt_path = os.path.join(model_dir, 'llm.pt')

    for p, name in [(qwen_path, 'CosyVoice-BlankEN/'), (llm_pt_path, 'llm.pt')]:
        if not os.path.exists(p):
            raise FileNotFoundError(f'{name} not found in {model_dir}')

    # Load llm.pt state dict and extract speech components directly
    print(f'[Step 1] Loading llm.pt from {llm_pt_path} ...')
    llm_state = torch.load(llm_pt_path, map_location='cpu', weights_only=True)

    speech_emb_weight = llm_state['speech_embedding.weight']  # (6761, 896)
    llm_dec_weight = llm_state['llm_decoder.weight']          # (6761, 896)
    speech_token_vocab_size = speech_emb_weight.shape[0]      # 6561 + 200 = 6761

    print(f'  speech_embedding: {speech_emb_weight.shape}')
    print(f'  llm_decoder: {llm_dec_weight.shape}')
    print(f'  speech_token_vocab_size: {speech_token_vocab_size}')

    # Load Qwen2 base model
    print(f'[Step 1] Loading Qwen2 base model from {qwen_path} ...')
    qwen_model = Qwen2ForCausalLM.from_pretrained(qwen_path)

    # Load fine-tuned Qwen2 weights from llm.pt
    qwen_state = {k: v for k, v in llm_state.items()
                  if k.startswith('llm.model.')}
    qwen_state_clean = {k[len('llm.model.'):]: v for k, v in qwen_state.items()}
    qwen_model.load_state_dict(qwen_state_clean, strict=True)
    print(f'  Loaded {len(qwen_state_clean)} Qwen2 parameters from llm.pt')

    del llm_state, qwen_state, qwen_state_clean

    # Extend tokenizer
    tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
    base_vocab_size = len(tokenizer)

    # CosyVoice3 text special tokens
    special_tokens = {
        'eos_token': '',
        'pad_token': '',
        'additional_special_tokens': [
            '<|im_start|>', '<|im_end|>', '<|endofprompt|>',
            '[breath]', '<strong>', '</strong>', '[noise]',
            '[laughter]', '[cough]', '[clucking]', '[accent]',
            '[quick_breath]',
            '<laughter>', '</laughter>',
            '[hissing]', '[sigh]', '[vocalized-noise]',
            '[lipsmack]', '[mn]', '<|endofsystem|>',
            # Phoneme tokens (matching CosyVoice3Tokenizer)
            '[AA]', '[AA0]', '[AA1]', '[AA2]', '[AE]', '[AE0]', '[AE1]', '[AE2]',
            '[AH]', '[AH0]', '[AH1]', '[AH2]', '[AO]', '[AO0]', '[AO1]', '[AO2]',
            '[AW]', '[AW0]', '[AW1]', '[AW2]', '[AY]', '[AY0]', '[AY1]', '[AY2]',
            '[B]', '[CH]', '[D]', '[DH]', '[EH]', '[EH0]', '[EH1]', '[EH2]',
            '[ER]', '[ER0]', '[ER1]', '[ER2]', '[EY]', '[EY0]', '[EY1]', '[EY2]',
            '[F]', '[G]', '[HH]', '[IH]', '[IH0]', '[IH1]', '[IH2]',
            '[IY]', '[IY0]', '[IY1]', '[IY2]', '[JH]', '[K]', '[L]', '[M]',
            '[N]', '[NG]', '[OW]', '[OW0]', '[OW1]', '[OW2]',
            '[OY]', '[OY0]', '[OY1]', '[OY2]', '[P]', '[R]', '[S]', '[SH]',
            '[T]', '[TH]', '[UH]', '[UH0]', '[UH1]', '[UH2]',
            '[UW]', '[UW0]', '[UW1]', '[UW2]', '[V]', '[W]', '[Y]', '[Z]', '[ZH]',
            '[a]', '[ai]', '[an]', '[ang]', '[ao]', '[b]', '[c]', '[ch]', '[d]',
            '[e]', '[ei]', '[en]', '[eng]', '[f]', '[g]', '[h]', '[i]', '[ian]',
            '[in]', '[ing]', '[iu]', '[j]', '[k]', '[l]', '[m]', '[n]', '[o]',
            '[ong]', '[ou]', '[p]', '[q]', '[r]', '[s]', '[sh]', '[t]', '[u]',
            '[uang]', '[ue]', '[un]', '[uo]', '[v]', '[w]', '[x]', '[y]', '[z]', '[zh]',
        ],
    }
    tokenizer.add_special_tokens(special_tokens)
    text_vocab_size = len(tokenizer)

    # Add speech tokens
    speech_tokens = [f'<|s_{i}|>' for i in range(speech_token_vocab_size)]
    speech_tokens[6561] = '<|sos|>'
    speech_tokens[6562] = '<|eos1|>'
    speech_tokens[6563] = '<|task_id|>'
    speech_tokens[6564] = '<|fill|>'
    tokenizer.add_tokens(speech_tokens)

    new_vocab_size = len(tokenizer)
    speech_token_offset = text_vocab_size

    # Pad to 128 for TRT efficiency
    padded_vocab_size = ((new_vocab_size + 127) // 128) * 128
    qwen_model.resize_token_embeddings(padded_vocab_size)

    # Merge speech_embedding into embed_tokens
    input_embeddings = qwen_model.get_input_embeddings()
    with torch.no_grad():
        src_size = min(speech_emb_weight.shape[0], speech_token_vocab_size)
        input_embeddings.weight[speech_token_offset:speech_token_offset + src_size] = \
            speech_emb_weight[:src_size].to(input_embeddings.weight.dtype)

    # Merge llm_decoder into lm_head
    new_lm_head = torch.nn.Linear(
        in_features=input_embeddings.weight.shape[1],
        out_features=padded_vocab_size,
        bias=False,
    )
    with torch.no_grad():
        new_lm_head.weight.data.zero_()

        # Copy original lm_head for text tokens
        orig_lm_head = qwen_model.lm_head
        if orig_lm_head is not None and orig_lm_head.weight.shape[0] >= text_vocab_size:
            new_lm_head.weight[:text_vocab_size] = orig_lm_head.weight[:text_vocab_size]

        # Copy llm_decoder for speech tokens
        decoder_size = min(llm_dec_weight.shape[0], speech_token_vocab_size)
        new_lm_head.weight[speech_token_offset:speech_token_offset + decoder_size] = \
            llm_dec_weight[:decoder_size].to(new_lm_head.weight.dtype)

    qwen_model.lm_head = new_lm_head

    # Update config
    qwen_model.config.vocab_size = padded_vocab_size
    qwen_model.config.tie_word_embeddings = False
    base_speech_token_size = 6561  # CosyVoice3LM.speech_token_size
    eos_id = speech_token_offset + base_speech_token_size + 1
    qwen_model.config.eos_token_id = eos_id
    qwen_model.generation_config.eos_token_id = eos_id
    qwen_model.generation_config.pad_token_id = eos_id
    qwen_model.generation_config.max_new_tokens = 2048

    # Save
    dtype_map = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}
    qwen_model.to(dtype_map[dtype])

    os.makedirs(output_dir, exist_ok=True)
    qwen_model.save_pretrained(output_dir)

    TEMPLATE = (
        "{%- for message in messages %}"
        "{%- if message['role'] == 'user' %}{{- '<|sos|>' + message['content'] + '<|task_id|>' }}"
        "{%- elif message['role'] == 'assistant' %}{{- message['content']}}"
        "{%- endif %}{%- endfor %}"
    )
    tokenizer.chat_template = TEMPLATE
    tokenizer.save_pretrained(output_dir)

    metadata = {
        'original_vocab_size': base_vocab_size,
        'text_vocab_size': text_vocab_size,
        'base_speech_token_size': base_speech_token_size,
        'embedding_size': speech_token_vocab_size,
        'padded_vocab_size': padded_vocab_size,
        'eos_token_id': eos_id,
        'speech_token_offset': speech_token_offset,
        'dtype': dtype,
    }
    with open(os.path.join(output_dir, 'cosyvoice3_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f'[Step 1] HF model saved to {output_dir}')
    print(f'  vocab: {base_vocab_size} -> {padded_vocab_size} (padded)')
    print(f'  speech_token_offset: {speech_token_offset}')
    print(f'  eos_token_id: {eos_id}')

    del qwen_model
    return metadata


def step2_convert_trt_weights(hf_model_dir: str, output_dir: str, dtype: str = 'bfloat16'):
    """Step 2: Convert HuggingFace checkpoint to TensorRT-LLM weights."""
    from tensorrt_llm.logger import logger

    if os.path.exists(output_dir) and os.listdir(output_dir):
        logger.info(f'[Step 2] TRT weights already exist at {output_dir}, skipping.')
        return

    logger.info(f'[Step 2] Converting HF checkpoint to TRT-LLM weights ...')
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models import QWenForCausalLM

    mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
    qwen = QWenForCausalLM.from_hugging_face(hf_model_dir, dtype, mapping=mapping)
    os.makedirs(output_dir, exist_ok=True)
    qwen.save_checkpoint(output_dir, save_config=True)
    del qwen
    logger.info(f'[Step 2] TRT weights saved to {output_dir}')


def step3_build_engine(trt_weights_dir: str, output_dir: str, dtype: str = 'bfloat16',
                       max_batch_size: int = 16, max_num_tokens: int = 32768):
    """Step 3: Build TensorRT engine from TRT weights."""
    from tensorrt_llm.logger import logger

    if os.path.exists(output_dir) and os.listdir(output_dir):
        logger.info(f'[Step 3] TRT engine already exists at {output_dir}, skipping.')
        return

    logger.info(f'[Step 3] Building TensorRT engine ...')
    os.makedirs(output_dir, exist_ok=True)

    cmd = (
        f'trtllm-build '
        f'--checkpoint_dir {trt_weights_dir} '
        f'--output_dir {output_dir} '
        f'--max_batch_size {max_batch_size} '
        f'--max_num_tokens {max_num_tokens} '
        f'--gemm_plugin {dtype}'
    )
    logger.info(f'Running: {cmd}')
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        raise RuntimeError(f'trtllm-build failed with return code {ret}')
    logger.info(f'[Step 3] Engine saved to {output_dir}')


def main():
    parser = argparse.ArgumentParser(description='Build TensorRT-LLM engine for CosyVoice3')
    parser.add_argument('--model-dir', type=str, default=CHECKPOINTS_DIR,
                        help='Path to CosyVoice3 model directory (CosyVoice-BlankEN/ + llm.pt)')
    parser.add_argument('--engine-dir', type=str, default=os.path.join(CHECKPOINTS_DIR, 'trt_engines'),
                        help='Output directory for TRT engine')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--max-batch-size', type=int, default=16)
    parser.add_argument('--max-num-tokens', type=int, default=32768)
    parser.add_argument('--skip-hf-convert', action='store_true',
                        help='Skip step 1 if HF model already exists')
    parser.add_argument('--step', type=int, choices=[1, 2, 3],
                        help='Run only a specific step (1=HF convert, 2=TRT weights, 3=engine build)')

    args = parser.parse_args()

    hf_model_dir = os.path.join(args.engine_dir, f'hf_merged_{args.dtype}')
    trt_weights_dir = os.path.join(args.engine_dir, f'trt_weights_{args.dtype}')
    trt_engines_dir = os.path.join(args.engine_dir, f'trt_engines_{args.dtype}')

    if args.step is None or args.step == 1:
        if args.skip_hf_convert and os.path.exists(hf_model_dir) and os.listdir(hf_model_dir):
            print(f'[Step 1] Skipping (HF model exists at {hf_model_dir})')
        else:
            step1_convert_to_hf(args.model_dir, hf_model_dir, args.dtype)

    if args.step is None or args.step == 2:
        step2_convert_trt_weights(hf_model_dir, trt_weights_dir, args.dtype)

    if args.step is None or args.step == 3:
        step3_build_engine(trt_weights_dir, trt_engines_dir, args.dtype,
                           args.max_batch_size, args.max_num_tokens)

    print('\nBuild complete.')
    if args.step is None or args.step == 3:
        print(f'  TRT engine: {trt_engines_dir}')
        print(f'  HF model:   {hf_model_dir} (for tokenizer)')


if __name__ == '__main__':
    main()
