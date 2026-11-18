from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch


def _slice_value(value, index: int):
    if torch.is_tensor(value):
        return value[index:index + 1]
    if isinstance(value, list):
        return [value[index]]
    return value


def slice_batch(batch: Dict, index: int) -> Dict:
    return {key: _slice_value(value, index) for key, value in batch.items()}


class GradDotSampleScorer:
    """Adapt TarDiff's gradient-dot scoring to the current downstream model API.

    The original TarDiff implementation caches the sum of training gradients from
    a downstream classifier, then scores candidate synthetic samples using the
    dot product between sample gradients and cached train gradients. Here we keep
    the same idea, but adapt it to a dict-based EHR model interface.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,
        device: str | torch.device | None = None,
        normalize: str = "l2",
        positive_guidance: bool = True,
        negative_guidance: bool = True,
    ):
        self.model = model
        self.train_loader = train_loader
        self.normalize = normalize
        self.positive_guidance = positive_guidance
        self.negative_guidance = negative_guidance
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)
        self.model.eval()
        self.grad_params = [p for p in self.model.parameters() if p.requires_grad]
        self.cached_train_grads = self._compute_train_grad_sum()

    def _normalize_gradients(self, grads: List[torch.Tensor]) -> List[torch.Tensor]:
        if not grads:
            return grads
        if self.normalize == "l2":
            total_norm = torch.sqrt(sum((g**2).sum() for g in grads)).clamp_min(1e-12)
            return [g / total_norm for g in grads]
        if self.normalize == "l1":
            total_norm = sum(g.abs().sum() for g in grads).clamp_min(1e-12)
            return [g / total_norm for g in grads]
        return grads

    def _label_gate(self, output: Dict[str, torch.Tensor]) -> bool:
        y_true = output.get("y_true")
        if y_true is None or not torch.is_tensor(y_true):
            return True
        target = y_true.detach()
        if target.numel() == 0:
            return True
        is_positive = bool((target > 0).any().item())
        if is_positive and not self.positive_guidance:
            return False
        if (not is_positive) and not self.negative_guidance:
            return False
        return True

    def _loss_and_grads(self, batch: Dict, create_graph: bool = False) -> List[torch.Tensor]:
        self.model.zero_grad(set_to_none=True)
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
        grads = [
            g if g is not None else torch.zeros_like(p)
            for g, p in zip(grads, self.grad_params)
        ]
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
            raise ValueError("No training gradients were collected for TarDiff bridge scorer.")
        return self._normalize_gradients(grad_acc)

    def score_batch(self, batch: Dict) -> torch.Tensor:
        scores = []
        batch_size = len(next(iter(batch.values())))
        for i in range(batch_size):
            sample_batch = slice_batch(batch, i)
            grads = self._loss_and_grads(sample_batch, create_graph=False)
            if not grads:
                scores.append(torch.tensor(0.0, device=self.device))
                continue
            total_dot = sum((g * cache).sum() for g, cache in zip(grads, self.cached_train_grads))
            scores.append(total_dot.detach())
        return torch.stack(scores)
