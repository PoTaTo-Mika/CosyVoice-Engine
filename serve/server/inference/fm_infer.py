"""
Flow Matching inference for CosyVoice3.

Converts semantic speech tokens + speaker embedding -> mel spectrogram
via CausalConditionalCFM (Euler ODE + CFG) with DiT estimator.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple

import torch

from .flow import CausalMaskedDiffWithDiT

logger = logging.getLogger(__name__)


class FlowMatchingEngine:
    """CosyVoice3 flow matching inference engine.

    Loads CausalMaskedDiffWithDiT and runs speech-token -> mel inference
    using Euler ODE solver with Classifier-Free Guidance.
    """

    def __init__(
        self,
        model_dir: str,
        fp16: bool = False,
        device: Optional[str] = None,
        n_timesteps: int = 10,
        inference_cfg_rate: Optional[float] = None,
    ):
        """
        Args:
            model_dir: Path to pretrained model directory containing cosyvoice3.yaml and flow.pt.
            fp16: Whether to use fp16 inference via torch.cuda.amp.
            device: Device string (e.g. 'cuda', 'cpu'). Auto-detected if None.
            n_timesteps: Number of Euler ODE steps during inference (default 10).
            inference_cfg_rate: Override CFG strength. If None, uses model default (typically 0.7).
        """
        self.model_dir = model_dir
        self.fp16 = fp16
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.n_timesteps = n_timesteps

        self._load_model(inference_cfg_rate)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, inference_cfg_rate: Optional[float]):
        """Load CausalMaskedDiffWithDiT from model_dir using hyperpyyaml config."""
        from hyperpyyaml import load_hyperpyyaml

        yaml_path = os.path.join(self.model_dir, 'cosyvoice3.yaml')
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f'cosyvoice3.yaml not found in {self.model_dir}')

        logger.info(f'Loading flow model config from {yaml_path}')

        qwen_path = os.path.join(self.model_dir, 'CosyVoice-BlankEN')
        overrides = {'qwen_pretrain_path': qwen_path}

        with open(yaml_path, 'r') as f:
            configs = load_hyperpyyaml(f, overrides=overrides)

        # Extract flow model
        self.model = configs['flow']
        if not isinstance(self.model, CausalMaskedDiffWithDiT):
            raise TypeError(
                f'Expected CausalMaskedDiffWithDiT, got {type(self.model).__name__}. '
                f'This engine only supports CosyVoice3.'
            )

        self.sample_rate = configs.get('sample_rate', 24000)
        self.token_mel_ratio = self.model.token_mel_ratio
        self.pre_lookahead_len = self.model.pre_lookahead_len

        # Override CFG rate if requested
        if inference_cfg_rate is not None:
            self.model.decoder.inference_cfg_rate = inference_cfg_rate

        # Load flow weights
        flow_pt_path = os.path.join(self.model_dir, 'flow.pt')
        if not os.path.exists(flow_pt_path):
            raise FileNotFoundError(f'flow.pt not found in {self.model_dir}')
        state_dict = torch.load(flow_pt_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device).eval()

        del configs
        logger.info(
            f'Flow model loaded: CausalMaskedDiffWithDiT, '
            f'device={self.device}, fp16={self.fp16}, n_timesteps={self.n_timesteps}'
        )

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def inference(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        token_len: Optional[torch.Tensor] = None,
        prompt_token_len: Optional[torch.Tensor] = None,
        prompt_feat_len: Optional[torch.Tensor] = None,
        streaming: bool = False,
        finalize: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run flow matching inference: speech tokens -> mel spectrogram.

        Args:
            token: Speech token ids. Shape (1, T), int32/int64.
            prompt_token: Prompt speech token ids. Shape (1, Tp), int32/int64.
            prompt_feat: Prompt mel features. Shape (1, Tp_mel, 80), float.
            embedding: Speaker embedding. Shape (1, 192), float.
            token_len: Token lengths. Auto-computed if None.
            prompt_token_len: Prompt token lengths. Auto-computed if None.
            prompt_feat_len: Prompt feat lengths. Auto-computed if None.
            streaming: Enable streaming mode.
            finalize: Whether this is the final chunk.

        Returns:
            (mel, cache) tuple.
            mel: Generated mel spectrogram. Shape (1, 80, T_mel), float32.
            cache: Updated flow cache for next streaming call, or None.
        """
        # Move inputs to device
        token = token.to(self.device, dtype=torch.int32)
        prompt_token = prompt_token.to(self.device)
        prompt_feat = prompt_feat.to(self.device)
        embedding = embedding.to(self.device)

        # Auto-compute lengths
        if token_len is None:
            token_len = torch.tensor([token.shape[1]], dtype=torch.int32, device=self.device)
        else:
            token_len = token_len.to(self.device, dtype=torch.int32)
        if prompt_token_len is None:
            prompt_token_len = torch.tensor([prompt_token.shape[1]], dtype=torch.int32, device=self.device)
        else:
            prompt_token_len = prompt_token_len.to(self.device, dtype=torch.int32)
        if prompt_feat_len is None:
            prompt_feat_len = torch.tensor([prompt_feat.shape[1]], dtype=torch.int32, device=self.device)
        else:
            prompt_feat_len = prompt_feat_len.to(self.device, dtype=torch.int32)

        with torch.cuda.amp.autocast(self.fp16):
            mel, cache = self.model.inference(
                token=token,
                token_len=token_len,
                prompt_token=prompt_token,
                prompt_token_len=prompt_token_len,
                prompt_feat=prompt_feat,
                prompt_feat_len=prompt_feat_len,
                embedding=embedding,
                streaming=streaming,
                finalize=finalize,
            )

        return mel.float(), cache

    @torch.inference_mode()
    def inference_batch(
        self,
        batch: List[Dict[str, torch.Tensor]],
        streaming: bool = False,
        finalize: bool = True,
    ) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Run batched flow matching inference with padding.

        Items with varying token/prompt lengths are padded to the same size
        within the batch, then run in a single forward pass.

        Args:
            batch: List of dicts, each with keys: token, prompt_token, prompt_feat, embedding.
            streaming: Enable streaming mode.
            finalize: Whether this is the final chunk.

        Returns:
            List of (mel, cache) tuples, one per item.
        """
        if len(batch) == 0:
            return []

        # Move all tensors to device first
        items = []
        for item in batch:
            items.append({
                'token': item['token'].to(self.device, dtype=torch.int32),
                'prompt_token': item['prompt_token'].to(self.device),
                'prompt_feat': item['prompt_feat'].to(self.device),
                'embedding': item['embedding'].to(self.device),
            })

        # Pad to same lengths within the batch
        max_token_len = max(it['token'].shape[1] for it in items)
        max_prompt_token_len = max(it['prompt_token'].shape[1] for it in items)
        max_prompt_feat_len = max(it['prompt_feat'].shape[1] for it in items)

        token_list, prompt_token_list, prompt_feat_list, embedding_list = [], [], [], []
        token_len_list, prompt_token_len_list, prompt_feat_len_list = [], [], []

        for it in items:
            T = it['token'].shape[1]
            PT = it['prompt_token'].shape[1]
            PF = it['prompt_feat'].shape[1]

            # Pad token: (1, T) -> (1, max_token_len)
            if T < max_token_len:
                pad = torch.zeros(1, max_token_len - T, dtype=it['token'].dtype, device=self.device)
                token_list.append(torch.cat([it['token'], pad], dim=1))
            else:
                token_list.append(it['token'])
            token_len_list.append(T)

            # Pad prompt_token: (1, PT) -> (1, max_prompt_token_len)
            if PT < max_prompt_token_len:
                pad = torch.zeros(1, max_prompt_token_len - PT, dtype=it['prompt_token'].dtype, device=self.device)
                prompt_token_list.append(torch.cat([it['prompt_token'], pad], dim=1))
            else:
                prompt_token_list.append(it['prompt_token'])
            prompt_token_len_list.append(PT)

            # Pad prompt_feat: (1, PF, 80) -> (1, max_prompt_feat_len, 80)
            if PF < max_prompt_feat_len:
                pad = torch.zeros(1, max_prompt_feat_len - PF, it['prompt_feat'].shape[2], dtype=it['prompt_feat'].dtype, device=self.device)
                prompt_feat_list.append(torch.cat([it['prompt_feat'], pad], dim=1))
            else:
                prompt_feat_list.append(it['prompt_feat'])
            prompt_feat_len_list.append(PF)

            embedding_list.append(it['embedding'])

        # Stack into batched tensors: (B, ...)
        token = torch.cat(token_list, dim=0)
        prompt_token = torch.cat(prompt_token_list, dim=0)
        prompt_feat = torch.cat(prompt_feat_list, dim=0)
        embedding = torch.cat(embedding_list, dim=0)
        token_len = torch.tensor(token_len_list, dtype=torch.int32, device=self.device)
        prompt_token_len = torch.tensor(prompt_token_len_list, dtype=torch.int32, device=self.device)
        prompt_feat_len = torch.tensor(prompt_feat_len_list, dtype=torch.int32, device=self.device)

        with torch.cuda.amp.autocast(self.fp16):
            mel, cache = self.model.inference(
                token=token,
                token_len=token_len,
                prompt_token=prompt_token,
                prompt_token_len=prompt_token_len,
                prompt_feat=prompt_feat,
                prompt_feat_len=prompt_feat_len,
                embedding=embedding,
                streaming=streaming,
                finalize=finalize,
            )

        # Un-batch: trim each mel to its actual length
        mel = mel.float()
        results = []
        for i in range(len(items)):
            # mel_len2 for each item = total_mel_len - prompt_feat_len
            pf_len = prompt_feat_len_list[i]
            # The model output mel shape is (B, 80, T_mel), where T_mel = mel_len1 + mel_len2
            # Each item's actual mel_len2 depends on its own token_len
            item_mel = mel[i:i+1, :, pf_len:]
            results.append((item_mel, None))

        return results

    # ------------------------------------------------------------------
    # Convenience: full token2mel pipeline (matching model.py token2wav flow part)
    # ------------------------------------------------------------------

    def token2mel(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        streaming: bool = False,
        finalize: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Alias for inference() with a shorter name."""
        return self.inference(
            token=token,
            prompt_token=prompt_token,
            prompt_feat=prompt_feat,
            embedding=embedding,
            streaming=streaming,
            finalize=finalize,
        )

