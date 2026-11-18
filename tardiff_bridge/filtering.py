from __future__ import annotations

import copy
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from data import convert_dataset
from loader import get_dataloader, get_special_input

from .grad_scorer import GradDotSampleScorer


def _attach_scores(samples: Sequence[Dict], scores: Sequence[float], field_name: str) -> List[Dict]:
    out = copy.deepcopy(list(samples))
    for sample, score in zip(out, scores):
        sample[field_name] = float(score)
    return out


def _filter_topk(samples: Sequence[Dict], scores: np.ndarray, keep_ratio: float, min_keep: int, field_name: str):
    if len(samples) == 0:
        return [], np.empty((0,), dtype=np.int64)
    keep_n = min(len(samples), max(int(round(len(samples) * keep_ratio)), min_keep))
    keep_n = max(1, keep_n)
    order = np.argsort(-scores)
    kept_idx = order[:keep_n]
    kept = _attach_scores([samples[i] for i in kept_idx], scores[kept_idx], field_name)
    return kept, kept_idx


def _score_samples(
    scorer: GradDotSampleScorer,
    samples: Sequence[Dict],
    task_config: Dict,
    batch_size: int,
) -> np.ndarray:
    dataset = convert_dataset(samples, dataset_name=task_config["DATASET"], task_name=task_config["TASK"], valid=False)
    collate_fn = get_special_input(task_config)
    loader = get_dataloader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)

    all_scores = []
    seen = 0
    progress = tqdm(loader, desc="Scoring synthetic samples")
    for batch in progress:
        batch_scores = scorer.score_batch(batch)
        all_scores.append(batch_scores.detach().cpu())
        seen += int(batch_scores.shape[0])
        progress.set_postfix(scored=seen, total=len(samples))
    if not all_scores:
        return np.empty((0,), dtype=np.float32)
    return torch.cat(all_scores, dim=0).numpy()


def filter_augmented_samples_by_graddot(
    guide_model: torch.nn.Module,
    real_train_loader,
    augmented_samples: Sequence[Dict],
    task_config: Dict,
    keep_ratio: float = 0.5,
    min_keep: int = 128,
    batch_size: int = 16,
    normalize: str = "l2",
    positive_guidance: bool = True,
    negative_guidance: bool = True,
    field_name: str = "tardiff_graddot",
) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    scorer = GradDotSampleScorer(
        model=guide_model,
        train_loader=real_train_loader,
        normalize=normalize,
        positive_guidance=positive_guidance,
        negative_guidance=negative_guidance,
    )
    scores = _score_samples(
        scorer=scorer,
        samples=augmented_samples,
        task_config=task_config,
        batch_size=batch_size,
    )
    kept, kept_idx = _filter_topk(
        samples=augmented_samples,
        scores=scores,
        keep_ratio=keep_ratio,
        min_keep=min_keep,
        field_name=field_name,
    )
    return kept, scores, kept_idx
