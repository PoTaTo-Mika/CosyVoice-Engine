"""Centralized path configuration for CosyVoice-Engine.

All model checkpoint paths are resolved relative to the repository root.
Override via environment variables if needed.
"""

import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CHECKPOINTS = os.path.join(_REPO_ROOT, "checkpoints")

CHECKPOINTS_DIR = os.environ.get("COSYVOICE_CHECKPOINTS_DIR", _DEFAULT_CHECKPOINTS)

# Sub-paths within checkpoints directory
COSYVOICE3_YAML = os.path.join(CHECKPOINTS_DIR, "cosyvoice3.yaml")
FLOW_PT = os.path.join(CHECKPOINTS_DIR, "flow.pt")
HIFT_PT = os.path.join(CHECKPOINTS_DIR, "hift.pt")
LLM_PT = os.path.join(CHECKPOINTS_DIR, "llm.pt")
QWEN_DIR = os.path.join(CHECKPOINTS_DIR, "CosyVoice-BlankEN")
SPEECH_TOKENIZER_ONNX = os.path.join(CHECKPOINTS_DIR, "speech_tokenizer_v3.onnx")
SPEECH_TOKENIZER_BATCH_ONNX = os.path.join(CHECKPOINTS_DIR, "speech_tokenizer_v3.batch.onnx")
CAMPPLUS_ONNX = os.path.join(CHECKPOINTS_DIR, "campplus.onnx")
