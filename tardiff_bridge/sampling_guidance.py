from __future__ import annotations

from typing import Dict, List

import torch

from tardiff_bridge.grad_scorer import slice_batch


class TarDiffSamplingGuidance:
    """Sampling-time TarDiff guidance on latent x0 predictions.

    This keeps TarDiff's core structure:
    1. cache train-set downstream gradients
    2. compute candidate-sample downstream gradients
    3. use gradient-dot as the influence signal
    4. backprop the influence signal to the current latent sample
    """

    def __init__(
        self,
        guide_model: torch.nn.Module,
        train_loader,
        device: str | torch.device | None = None,
        normalize: str = "l2",
        positive_guidance: bool = True,
        negative_guidance: bool = True,
        guidance_scale: float = 0.1,
        detach_x0: bool = True,
    ):
        self.model = guide_model
        self.train_loader = train_loader
        self.normalize = normalize
        self.positive_guidance = positive_guidance
        self.negative_guidance = negative_guidance
        self.guidance_scale = guidance_scale
        self.detach_x0 = detach_x0
        self.device = device or next(guide_model.parameters()).device
        self.model.to(self.device)
        self.model.eval()
        self.grad_params = [p for p in self.model.parameters() if p.requires_grad]
        self.cached_train_grads = self._compute_train_grad_sum()

    def _normalize_gradients(self, grads: List[torch.Tensor]) -> List[torch.Tensor]:
        if not grads:
            return grads
        if self.normalize == "l2":
            total_norm = torch.sqrt(sum((g ** 2).sum() for g in grads)).clamp_min(1e-12)
            return [g / total_norm for g in grads]
        if self.normalize == "l1":
            total_norm = sum(g.abs().sum() for g in grads).clamp_min(1e-12)
            return [g / total_norm for g in grads]
        return grads

    def _label_gate(self, output: Dict[str, torch.Tensor]) -> bool:
        y_true = output.get("y_true")
        if y_true is None or not torch.is_tensor(y_true) or y_true.numel() == 0:
            return True
        is_positive = bool((y_true.detach() > 0).any().item())
        if is_positive and not self.positive_guidance:
            return False
        if (not is_positive) and not self.negative_guidance:
            return False
        return True

    def _build_guide_batch(self, raw_batch: Dict, latent: torch.Tensor) -> Dict:
        batch = dict(raw_batch)
        batch["new_conditions"] = latent[:, 0, :]
        batch["new_procedures"] = latent[:, 1, :]
        batch["new_drugs"] = latent[:, 2, :]
        batch.setdefault("miss_feature", 0)
        return batch

    def _loss_and_grads(self, batch: Dict, create_graph: bool = False) -> List[torch.Tensor]:
        self.model.zero_grad(set_to_none=True)

        # ===================== 最终修复 =====================
        # 生成阶段的 guide_batch 没有标签，直接返回空梯度
        label_keys = ['labels', 'label', 'Label', 'Labels', 'LABEL', 'LABELS']
        if not any(k in batch for k in label_keys):
            return []

        # 有标签时才正常运行
        output = self.model(**batch)

        if not self._label_gate(output):
            return []

        loss = output["loss"]
        grads = torch.autograd.grad(
            loss,
            self.grad_params,
            create_graph=create_graph,
            allow_unused=True,
        )
        grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, self.grad_params)]
        return self._normalize_gradients(grads)

    def _compute_train_grad_sum(self) -> List[torch.Tensor]:
        grad_acc = None
        for batch in self.train_loader:
            batch_size = len(next(iter(batch.values())))
            for i in range(batch_size):
                sample_batch = slice_batch(batch, i)
                grads = self._loss_and_grads(sample_batch, create_graph=False)
                if not grads:
                    continue
                if grad_acc is None:
                    grad_acc = [g.detach().clone() for g in grads]
                else:
                    grad_acc = [acc + g.detach() for acc, g in zip(grad_acc, grads)]
        if grad_acc is None:
            raise ValueError("No training gradients were collected for TarDiff sampling guidance.")
        return self._normalize_gradients(grad_acc)

    def latent_guidance(self, raw_batch: Dict, x0_pred: torch.Tensor) -> torch.Tensor:
        x0_var = x0_pred.detach().requires_grad_(True) if self.detach_x0 else x0_pred.requires_grad_(True)
        per_sample_grads = []
        batch_size = x0_var.size(0)

        for i in range(batch_size):
            raw_sample = slice_batch(raw_batch, i)
            latent_sample = x0_var[i:i + 1]
            guide_batch = self._build_guide_batch(raw_sample, latent_sample)
            grads = self._loss_and_grads(guide_batch, create_graph=True)
            if not grads:
                per_sample_grads.append(torch.zeros_like(latent_sample))
                continue
            influence = sum((g * cache).sum() for g, cache in zip(grads, self.cached_train_grads))
            latent_grad = torch.autograd.grad(influence, latent_sample, retain_graph=True, allow_unused=True)[0]
            if latent_grad is None:
                latent_grad = torch.zeros_like(latent_sample)
            per_sample_grads.append(latent_grad)

        latent_grad = torch.cat(per_sample_grads, dim=0)
        if self.normalize == "l2":
            grad_norm = torch.sqrt((latent_grad ** 2).sum(dim=(1, 2), keepdim=True)).clamp_min(1e-12)
            latent_grad = latent_grad / grad_norm
        elif self.normalize == "l1":
            grad_norm = latent_grad.abs().sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)
            latent_grad = latent_grad / grad_norm
        return latent_grad