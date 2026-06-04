# CosyVoice-Engine

基于 TensorRT-LLM 的 CosyVoice3 推理服务。

## 1. 下载权重

```bash
hf download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --local-dir ./checkpoints
```

下载完成后 `checkpoints/` 目录应包含：

```
checkpoints/
├── cosyvoice3.yaml
├── llm.pt
├── flow.pt
├── hift.pt
├── CosyVoice-BlankEN/   # Qwen2 基座 tokenizer + 权重
├── speech_tokenizer_v3.onnx
├── speech_tokenizer_v3.batch.onnx
└── campplus.onnx
```

## 2. 构建 TRT Engine

**土豆注**：因为TRT是严格要求 GPU 型号，TRT版本，torch版本，CUDA版本全都匹配的，所以每换一种新卡

三步 pipeline，将 CosyVoice3 LLM 转换为 TensorRT engine：

| 步骤 | 输入 | 输出 | 说明 |
|------|------|------|------|
| Step 1 | `llm.pt` + `CosyVoice-BlankEN/` | `trt_engines/hf_merged_bfloat16/` | 将 speech_embedding/llm_decoder 合并到 Qwen2，扩展 tokenizer，保存标准 HF 格式 |
| Step 2 | HF merged model | `trt_engines/trt_weights_bfloat16/` | 转换为 TensorRT-LLM 权重 |
| Step 3 | TRT weights | `trt_engines/trt_engines_bfloat16/` | 编译 TRT engine |

### 环境准备

```bash
conda create -n cosyvoice python=3.10 -y
conda activate cosyvoice

# 核心依赖
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install transformers hyperpyyaml

# TensorRT-LLM
pip install tensorrt-llm
conda install -y -c conda-forge openmpi mpi4py
```

### 执行构建

```bash
# 完整三步（推荐）：
PYTHONPATH=. python serve/build_engine.py

# 或逐步执行：
PYTHONPATH=. python serve/build_engine.py --step 1   # HF 转换
PYTHONPATH=. python serve/build_engine.py --step 2   # TRT 权重
PYTHONPATH=. python serve/build_engine.py --step 3   # Engine 编译

# 若 HF merged model 已存在，跳过 Step 1：
PYTHONPATH=. python serve/build_engine.py --skip-hf-convert
```

构建完成后产物位于 `checkpoints/trt_engines/`：

```
checkpoints/trt_engines/
├── hf_merged_bfloat16/          # Step 1: 合并后的 HF 模型（含 tokenizer）
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── chat_template.jinja
│   └── cosyvoice3_metadata.json  # speech_token_offset 等元信息
├── trt_weights_bfloat16/        # Step 2: TRT-LLM 权重
│   ├── rank0.safetensors
│   └── config.json
└── trt_engines_bfloat16/        # Step 3: 可部署的 TRT engine
    ├── rank0.engine              # ~1.3 GB
    └── config.json
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dtype` | `bfloat16` | 精度（float16/bfloat16/float32） |
| `--max-batch-size` | 16 | 最大批大小 |
| `--max-num-tokens` | 32768 | 最大 token 数（影响显存占用） |
| `--model-dir` | `./checkpoints` | 模型目录 |
| `--engine-dir` | `./checkpoints/trt_engines` | 输出目录 |

## 3. 启动服务

```bash
# LLM Server（TRT-LLM）
PYTHONPATH=. python serve/setup_server.py --port 50000

# Flow Matching Server
PYTHONPATH=. python -m serve.server.fm_server --port 50001

# Vocoder Server
PYTHONPATH=. python -m serve.server.vocoder_server --port 50002
```

所有服务默认从 `./checkpoints/` 加载权重，无需手动指定路径。可通过 `COSYVOICE_CHECKPOINTS_DIR` 环境变量覆盖。
