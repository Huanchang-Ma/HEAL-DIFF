from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional


@dataclass
class TarDiffBridgeConfig:
    enabled: bool = True
    score_mode: str = "graddot"
    keep_ratio: float = 0.5
    min_keep: int = 128
    batch_size: int = 16
    use_new_merge: bool = True
    include_val: bool = False
    random_seed: int = 528
    save_suffix: str = "tardiff"
    positive_guidance: bool = True
    negative_guidance: bool = True
    normalize: str = "l2"
    score_field: str = "tardiff_graddot"
    reuse_existing_guide: bool = True
    use_sampling_guidance: bool = True
    guidance_scale: float = 0.1
    guidance_score_mode: str = "graddot"
    guidance_detach_x0: bool = True
    guided_save_suffix: str = "_tardiffguided"
    reuse_existing_diffusion: bool = True
    reuse_existing_guided_samples: bool = True
    run_original_mix: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_bridge_config(overrides: Optional[Dict[str, Any]] = None) -> TarDiffBridgeConfig:
    cfg = TarDiffBridgeConfig()
    if not overrides:
        return cfg

    valid = {f.name for f in fields(TarDiffBridgeConfig)}
    for key, value in overrides.items():
        if key in valid:
            setattr(cfg, key, value)
    return cfg
