# HEAL_DIFF

This directory contains the DP-SGD ablation workflow for multi-view electronic health record (EHR) imputation and downstream prediction. The three views are diagnoses (`conditions`), procedures (`procedures`), and medications (`drugs`).

The entry point is [`main_unify_DiffmvDP0420_Ablation.py`](main_unify_DiffmvDP0420_Ablation.py). It prepares patient-level train, validation, and test splits, calls [`train_dffs_DiffmvDP0420_Ablation2.py`](train_dffs_DiffmvDP0420_Ablation2.py), generates augmented samples, and runs downstream prediction.

## Method

| Module | Implementation |
| --- | --- |
| **Adaptive Diffusion Module** | `Schedule` and `LSimple()` apply view masks to the diffusion process and training objective. The network conditions denoising on visit context. The current ablation fixes the masked view to `conditions`. [Diffusion utilities](diffuser/diff_utils.py), [denoising network](diffuser/unitrans.py), [encoders](diffuser/auto.py), [DPM-Solver](diffuser/dpm.py). |
| **Weak-view Supervision Augmentation Module** | `DPFullDiffmvWrapper.forward()` combines per-view binary cross-entropy and Soft Jaccard losses. The diagnostic view receives additional ranking supervision and a scheduled increase in supervision weight. [Ablation trainer](train_dffs_DiffmvDP0420_Ablation2.py). |
| **View-aware Prior Module** | `build_medical_prior_from_loader()` estimates positive pointwise mutual information (PPMI) across views from the training split. With the default settings, the prior adjusts medication logits during training. The trainer's `decode_with_stats()` adjusts diagnostic and medication logits during sampling. [Ablation trainer](train_dffs_DiffmvDP0420_Ablation2.py). |

## Source Files

| Area | Files |
| --- | --- |
| Entry point and diffusion training | [`main_unify_DiffmvDP0420_Ablation.py`](main_unify_DiffmvDP0420_Ablation.py), [`train_dffs_DiffmvDP0420_Ablation2.py`](train_dffs_DiffmvDP0420_Ablation2.py) |
| Configuration | [`config.py`](config.py) for data and downstream training. [`diffuser/diff_config.py`](diffuser/diff_config.py) for the denoiser, DP-SGD, sampling, and output paths. |
| EHR data and batching | [`data.py`](data.py), [`utils.py`](utils.py), [`baseline_utils.py`](baseline_utils.py), [`loader.py`](loader.py), [`diffuser/diff_dataset.py`](diffuser/diff_dataset.py) |
| Diffusion package | [`diffuser/__init__.py`](diffuser/__init__.py), `diff_utils.py`, `unitrans.py`, `auto.py`, and `dpm.py` in [`diffuser/`](diffuser/) |
| Checkpoint loading and imputation | [`eval_dffs.py`](eval_dffs.py) loads DP weights and augments the train, validation, and test splits. It also contains latent MAE/RMSE routines. |
| Downstream prediction | [`train_hp.py`](train_hp.py), [`trainer.py`](trainer.py), [`models.py`](models.py), [`models_soft.py`](models_soft.py), [`metrics.py`](metrics.py), and [`baselines/`](baselines/). The current `run_pcf()` also imports and runs [`downstream_MIA.py`](downstream_MIA.py). |

## Setup and Run

Use the Python/CUDA environment associated with the experiment or checkpoint. The workflow uses PyTorch, PyHealth, Opacus, DGL, Faiss, NumPy, pandas, scikit-learn, PyYAML, Matplotlib, einops, tqdm, absl-py, and W&B. Raw `MIV-Note` preprocessing also imports `pandarallel`.

1. Set `DATASET`, `TASK`, `MODEL`, `GPU`, `SEED`, and `LOGDIR` in `config.py`. This snapshot defaults to `eICU`, `LOS`, and `ours`. The configuration also includes `MIII`, `MIV-Note`, and the `PHE` prediction task.
2. Update the raw data paths, `root_to`, and `root_to2` in the entry point. Set the data directory in `data.py` so it can load `mimic-hcup_ccs_2015_definitions.yaml` at import time.
3. Set `ckpt_root`, `workdir`, `sample_dir`, and `sample['path']` in `diffuser/diff_config.py`. Keep the trailing `/` on directory settings used in string concatenation.
4. From this directory, run the entry point.

```bash
python main_unify_DiffmvDP0420_Ablation.py
```

The entry point defaults to `augment=True` and `retrain=True` for `ours` and `MedDiff`. If `datasets_pre_stand.pkl` is missing, the first invocation preprocesses the raw data and exits. Run it again to create or reuse the patient-level split files, train diffusion, generate augmented copies, and run downstream prediction.

For checkpoint reuse, set `retrain=False` and verify that `best_model_path.txt`, model configuration, tokenizer vocabulary order, and data split belong to the same experiment. If no checkpoint directory exists for the dataset, the entry point starts training.

## Checkpoints and Evaluation

During diffusion training, the entry point passes `[train_dataset, val_dataset]`. The diffusion dataset wrapper names its second split `test` internally. In this call, that split contains validation records.

The trainer selects the checkpoint by the mean of the three validation Jaccard scores. Scores within `1e-8` are tied, with lower MSE breaking the tie. Saving a new best checkpoint removes the other `.ckpt` directories under the same dataset, task, and epsilon tag. Use a separate `ckpt_root` to keep checkpoints from independent runs. If a DP epoch has no training updates yet, the code can submit the placeholder `(1.0, 1.0, 1.0)` to checkpoint selection. Check `eval.log` for a corresponding sampling evaluation.

| Output | Location |
| --- | --- |
| Diffusion weights | `<ckpt_root>/<DATASET>/<TASK>/<eps_tag>/<epoch>.ckpt/nnet.pth`, with tags such as `eps0p5`. Keep the full checkpoint directory to retain EMA weights and training state. |
| Checkpoint pointer | `<ckpt_root>/<DATASET>/<TASK>/best_model_path.txt`, containing the weight path and training visit count. |
| Best validation metrics | `<workdir>/<DATASET>/<TASK>/generation_dp_summary.csv`, with `best_jaccard_*`, `best_epoch`, `nnet_path`, `actual_epsilon`, and `synth_dataset_path`. Use `exp_num` and `target_epsilon` to identify the run. |
| Evaluation history | `eval.log` and `<DATASET>-<TASK>-<exp_num>-epoch_history.csv` under `<workdir>/<DATASET>/<TASK>/`. |
| Augmented test metrics | `<root_to2>/test_jaccard_<DATASET>_<TASK>.csv`, with per-view scores and `test_jaccard_mean`. |

When `dp_noise_multiplier` is set, DP-SGD uses that value. The directory tag uses the configured `dp_target_epsilon`. The measured epsilon is reported in `actual_epsilon` when privacy accounting succeeds.

### Test-Set Jaccard

For `MODEL='ours'`, generate the train, validation, and test augmentation caches with the desired checkpoint. Then run the entry point with `augment=True` and `retrain=False` to read the three cached files and write the test CSV. The default cache directory is `data/<TASK>/<DATASET>/processed0602/`. The cache filenames are shared within that directory, so verify the checkpoint and split used to generate them.

The test metric averages sample-level code-set Jaccard within each view and uses `nanmean` across the three view scores. It scores the samples generated by `augment_dataset1()`, which uses DPM-Solver and threshold decoding. The trainer's validation evaluation uses EMA weights in DP mode and uses its own best-of-k sampling and prior-aware decoder. Keep the evaluation path with any reported score.

## Existing Files and Experiment History

You can keep the existing files in this workspace. The entry point loads the modules listed above, and the other scripts remain available for reproducing earlier experiments. For a source-only copy of the current method, the following groups are optional.

| Files or directories | Why keep them in the workspace |
| --- | --- |
| [`main_unify.py`](main_unify.py), [`train_dffs.py`](train_dffs.py), [`main_unify_DiffmvDP0420.py`](main_unify_DiffmvDP0420.py), [`train_dffs_DiffmvDP0420_Ablation.py`](train_dffs_DiffmvDP0420_Ablation.py), and other historical entry points and trainers | Reproduce original Diffmv, older DP settings, and earlier ablations. |
| [`main_unify_pmi_3split.py`](main_unify_pmi_3split.py), [`train_dffs_pmi_3split.py`](train_dffs_pmi_3split.py), and [`tardiff_bridge/`](tardiff_bridge/) | Reproduce PMI and TarDiff experiments. |
| [`config_DP.py`](config_DP.py), [`train_hp_DP.py`](train_hp_DP.py), [`trainer_DP.py`](trainer_DP.py), [`models_DP.py`](models_DP.py), [`eval_diff.py`](eval_diff.py), and [`eval_dffs1222.py`](eval_dffs1222.py) | Retain the alternate downstream DP and evaluation workflows. |
| [`audit/`](audit/), [`scripts/`](scripts/), [`run_all_eps_downstreamMIA.py`](run_all_eps_downstreamMIA.py), and [`compare_dp_nondp_outputs.py`](compare_dp_nondp_outputs.py) | Reproduce privacy audits, multi-epsilon sweeps, comparisons, and cluster jobs. |
| [`outputs/`](outputs/), [`MAE/`](MAE/), and [`eps_plots/`](eps_plots/) | Retain prior results, figures, and their supporting analysis. |

Keep the full checkpoint, experiment configuration, patient split, and tokenizer vocabulary order together when preserving a specific run. The trainer can rebuild `medical_prior.pt` from the training split. Keep the original prior cache when exact reuse matters.
