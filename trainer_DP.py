# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : trainer.py
# Time       ：4/11/2024 3:25 pm
# Author     ：Chuang Zhao
# version    ：python 
# Description：
"""

import logging
import os

from datetime import datetime
from typing import Callable, Dict, List, Optional, Type

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.autonotebook import trange

from metrics import multilabel_metrics_fn, regression_metrics_fn, multiclass_metrics_fn, binary_metrics_fn

from pyhealth.utils import create_directory
from config import config
from utils import adjust_learning_rate

import faiss
import torch.nn.functional as F
from utils import get_contribution
from opacus import PrivacyEngine
import torch.nn as nn
from opacus.layers import DPMultiheadAttention
from opacus.validators import ModuleValidator
from opacus import PrivacyEngine
logger = logging.getLogger(__name__)

import copy
def sanitize_module_for_dp(module: nn.Module, verbose=True):
    for mod_name, mod in module.named_modules():
        to_delete = []
        for k, v in list(vars(mod).items()):
            if k in {"_parameters", "_buffers", "_modules"}:
                continue
            try:
                copy.deepcopy(v)
            except Exception:
                to_delete.append((k, type(v).__name__))
        for k, typ in to_delete:
            if verbose:
                print(f"[sanitize] remove {mod_name}.{k} ({typ})")
            try:
                delattr(mod, k)
            except Exception:
                setattr(mod, k, None)
    return module

def replace_multihead_attention_with_dp(module: nn.Module):
    for name, child in module.named_children():
        if isinstance(child, nn.MultiheadAttention):
            dp_mha = DPMultiheadAttention(
                embed_dim=child.embed_dim,
                num_heads=child.num_heads,
                dropout=child.dropout,
                bias=(child.in_proj_bias is not None),
                add_bias_kv=(child.bias_k is not None and child.bias_v is not None),
                add_zero_attn=child.add_zero_attn,
                kdim=child.kdim,
                vdim=child.vdim,
                batch_first=getattr(child, "batch_first", False),
            )

            # 直接加载原始 MultiheadAttention 的权重
            dp_mha.load_state_dict(child.state_dict())
            setattr(module, name, dp_mha)
        else:
            replace_multihead_attention_with_dp(child)

    return module

def is_best(best_score: float, score: float, monitor_criterion: str) -> bool:
    if monitor_criterion == "max":
        return score > best_score
    elif monitor_criterion == "min":
        return score < best_score
    else:
        raise ValueError(f"Monitor criterion {monitor_criterion} is not supported")


def set_logger(log_path: str) -> None:
    create_directory(log_path)
    log_filename = os.path.join(log_path, "log.txt")
    logger.setLevel(logging.DEBUG)
    # 创建前先删除，避免重复打印
    logger.handlers.clear()
    # 文件日志
    handler = logging.FileHandler(log_filename)
    formatter = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # 控制台日志
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return

def get_metrics_fn(mode: str) -> Callable:
    if mode == "binary":
        return binary_metrics_fn
    elif mode == "multiclass":
        return multiclass_metrics_fn
    elif mode == "multilabel":
        return multilabel_metrics_fn
    elif mode == "regression":
        return regression_metrics_fn
    else:
        raise ValueError(f"Mode {mode} is not supported")


def _filter_empty_param_groups(param_groups):
    return [g for g in param_groups if len(g["params"]) > 0]
def _find_missing_grad_sample(model):
    bad = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if not hasattr(p, "grad_sample") or p.grad_sample is None:
            bad.append(name)
    return bad
def _unwrap_model(model):
    # Opacus make_private 后，self.model 可能变成 GradSampleModule
    return model._module if hasattr(model, "_module") else model
def _freeze_for_dp_head_only(model, logger=None):
    """
    只保留最终预测头 fina_proj 可训练；
    其它参数全部冻结。
    这能最大概率避免 Opacus 因为自定义 Transformer/辅助参数缺少 grad_sample 而报错。
    """
    # 先全冻结
    for _, p in model.named_parameters():
        p.requires_grad = False

    # 只打开最终分类头
    # 模型里主要是:
    # rec_layer.pred_layer_subs.{conditions,procedures,drugs}.fina_proj.*
    for name, p in model.named_parameters():
        if ".fina_proj." in name:
            p.requires_grad = True

    # 这些即使名字里不小心匹配到，也强制冻结
    hard_freeze_keywords = [
        "_dummy_param",
        ".clz_token",
        "beta_logits",
        ".gru.",
        ".rnns.",
        ".self_attn.",
        ".linear1.",
        ".linear2.",
        ".norm1.",
        ".norm2.",
        "id_proj",
        "diff_proj",
        "note_linear",
        "incomplete_emb",
        "positional_enc",
        "embeddings.",
        "pred_joint",   # 当前loss_joint 没进最终返回 loss，先整支冻结
    ]
    for name, p in model.named_parameters():
        if any(k in name for k in hard_freeze_keywords):
            p.requires_grad = False

    if logger is not None:
        logger.info("[DP] trainable params after head-only freeze:")
        for name, p in model.named_parameters():
            if p.requires_grad:
                logger.info(f"[DP] trainable: {name}")

class Trainer:
    """Trainer for PyTorch models.

    Args:
        model: PyTorch model.
        checkpoint_path: Path to the checkpoint. Default is None, which means
            the model will be randomly initialized.
        metrics: List of metric names to be calculated. Default is None, which
            means the default metrics in each metrics_fn will be used.
        device: Device to be used for training. Default is None, which means
            the device will be GPU if available, otherwise CPU.
        enable_logging: Whether to enable logging. Default is True.
        output_path: Path to save the output. Default is "./output".
        exp_name: Name of the experiment. Default is current datetime.
    """

    def __init__(
        self,
        model: nn.Module,
        checkpoint_path: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        device: Optional[str] = None,
        enable_logging: bool = True,
        output_path: Optional[str] = None,
        exp_name: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model
        self.metrics = metrics
        self.device = device

        # set logger
        if enable_logging:
            if output_path is None:
                output_path = os.path.join(os.getcwd(), "output")
            if exp_name is None:
                exp_name = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.exp_path = os.path.join(output_path, exp_name)
            set_logger(self.exp_path)
        else:
            self.exp_path = None
        logger.info(f"Experiment path: {self.exp_path}")
        logger.info(config)
        logger.info("==============================")

        # set device
        self.model.to(self.device)

        # logging
        # logger.info(self.model)
        logger.info(f"Metrics: {self.metrics}")
        logger.info(f"Device: {self.device}")

        # load checkpoint
        if checkpoint_path is not None:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            self.load_ckpt(checkpoint_path)

        logger.info("")
        return

    def train_prefix(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        epochs: int = 5,
        optimizer_class: Type[Optimizer] = torch.optim.AdamW, # tuning的时候看看要不要换成Adam MIMIC-III Drug
        optimizer_params: Optional[Dict[str, object]] = None,
        steps_per_epoch: int = None,
        evaluation_steps: int = 1,
        weight_decay: float = 0.0,
        max_grad_norm: float = None,
        monitor: Optional[str] = None,
        monitor_criterion: str = "max",
        load_best_model_at_last: bool = True,
        aux_data: Optional[Dict[str, object]] = None,
    ):
        """Trains the model. for multi-stage model; 仅需要改collate_fn的EHR model (such as RAREMed)

        Args:
            train_dataloader: Dataloader for training.
            val_dataloader: Dataloader for validation. Default is None.
            test_dataloader: Dataloader for testing. Default is None.
            epochs: Number of epochs. Default is 5.
            optimizer_class: Optimizer class. Default is torch.optim.Adam.
            optimizer_params: Parameters for the optimizer. Default is {"lr": 1e-3}.
            steps_per_epoch: Number of steps per epoch. Default is None.
            weight_decay: Weight decay. Default is 0.0.
            max_grad_norm: Maximum gradient norm. Default is None.
            monitor: Metric name to monitor. Default is None.
            monitor_criterion: Criterion to monitor. Default is "max".
            load_best_model_at_last: Whether to load the best model at the last.
                Default is True.
        """
        if optimizer_params is None:
            optimizer_params = {"lr": 1e-3}

        # logging
        logger.info("Training:")
        logger.info(f"Batch size: {train_dataloader.batch_size}")
        logger.info(f"Optimizer: {optimizer_class}")
        logger.info(f"Optimizer params: {optimizer_params}")
        logger.info(f"Weight decay: {weight_decay}")
        logger.info(f"Max grad norm: {max_grad_norm}")
        logger.info(f"Val dataloader: {val_dataloader}")
        logger.info(f"Monitor: {monitor}")
        logger.info(f"Monitor criterion: {monitor_criterion}")
        logger.info(f"Epochs: {epochs}")

        # set optimizer
        param = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in param if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
        #                                                        T_max=epochs//10)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epochs + 1))

        # initialize
        data_iterator = iter(train_dataloader) # 生成迭代器，减少内存消耗
        best_score = -1 * float("inf") if monitor_criterion == "max" else float("inf")
        if steps_per_epoch == None:
            steps_per_epoch = len(train_dataloader)
        global_step = 0
        # epoch training loop
        for epoch in range(epochs):
            max_step, cur_step = len(train_dataloader), 0

            training_loss = []
            self.model.zero_grad()
            self.model.train()
            # batch training loop
            logger.info("")
            for _ in trange(
                steps_per_epoch,
                desc=f"Epoch {epoch} / {epochs}",
                smoothing=0.05,
            ):
                try:
                    data = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(train_dataloader)
                    data = next(data_iterator)

                # forward
                cur_step +=1
                # adjust_learning_rate(optimizer, cur_step, config, max_step) # 真的离谱
                if epoch<0:
                    output = self.model(train_mode='pretrain_mask', **data) # 传入一个字典对象，下面根据其名称对应的取值 train_mode='fine_tune', 可以先放前面
                    loss = output["loss"]
                elif epoch <0:
                    output = self.model(train_mode='pretrain_nsp', **data) # 前面的预训练真的很扯淡
                    loss = output["loss"]
                else: # 20个epoch
                    output = self.model(train_mode='finetune', **data)
                    loss = output["loss"]

                # backward
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm
                    )
                # update
                optimizer.step()
                optimizer.zero_grad()
                training_loss.append(loss.item())
                global_step += 1
            # scheduler.step()

            # log and save
            logger.info(f"--- Train epoch-{epoch}, step-{global_step} ---")
            logger.info(f"loss: {sum(training_loss) / len(training_loss):.4f}")
            if self.exp_path is not None:
                pass
                # self.save_ckpt(os.path.join(self.exp_path, "last.ckpt"))

            # validation
            if val_dataloader is not None:
                scores = self.evaluate(val_dataloader, aux_data)
                logger.info(f"--- Eval epoch-{epoch}, step-{global_step} ---")
                for key in scores.keys():
                    logger.info("{}: {}".format(key, scores[key]))
                    # logger.info("{}: {:.4f}".format(key, scores[key]))

                # save best model
                if monitor is not None:
                    score = scores[monitor]
                    if is_best(best_score, score, monitor_criterion):
                        logger.info(
                            f"New best {monitor} score ({score:.4f}) "
                            f"at epoch-{epoch}, step-{global_step}"
                        )
                        best_score = score

                        if self.exp_path is not None:
                            self.save_ckpt(os.path.join(self.exp_path, "best.ckpt"))

        # load best model
        if load_best_model_at_last and self.exp_path is not None and os.path.isfile(
            os.path.join(self.exp_path, "best.ckpt")):
            logger.info("Loaded best model")
            self.load_ckpt(os.path.join(self.exp_path, "best.ckpt"))

        # test
        if test_dataloader is not None:
            scores = self.evaluate(test_dataloader, aux_data)
            logger.info(f"--- Test ---")
            for key in scores.keys():
                # logger.info("{}: {:.4f}".format(key, scores[key]))
                if key.endswith('grouped'):
                    logger.info("{}: {}".format(key, scores[key])) # 列表
                else:
                    logger.info("{}: {:4f}".format(key, scores[key])) # 浮点数
                # logger.info("{}: {}".format(key, scores[key])) # 这里会不会有问题【因为有topK】

        return

    def set_params(self, flag):
        if config['MODEL']=='SMART':
            if flag:
                for param in self.model.parameters():
                    param.requires_grad = False
                for param in self.model.rec_layer.fina_proj.parameters():
                    param.requires_grad = True
        else:
            print("Please set warm epoch for model!")

    # def kmeans(self, embeddings): # 其实要传入patient_emb，这样得提前存储
    #     if config['MODEL'] == 'PRISM':
    #         condition_emb = embeddings['conditions'].detach().cpu().numpy()
    #         # procedure_emb = embeddings['procedures'].detach().cpu().numpy()
    #         # drugs_emb = embeddings['drugs'].detach().cpu().numpy()
    #         kmeans = faiss.Kmeans(d=128, k=100, gpu=False)  # GPU可能调不动，OOM
    #         kmeans.train(condition_emb)
    #         cluster_cents = kmeans.centroids
    #         centroids = torch.Tensor(cluster_cents).to(self.device)
    #         centroids = F.normalize(centroids, p=2, dim=1) # 100, 128
    #         self.model.update(centroids)
    #     else:
    #         pass

    # def train(
    #     self,
    #     train_dataloader: DataLoader,
    #     val_dataloader: Optional[DataLoader] = None,
    #     test_dataloader: Optional[DataLoader] = None,
    #     epochs: int = 5,
    #     warm_epochs: Optional[int] =None,
    #     optimizer_class: Type[Optimizer] = torch.optim.AdamW,
    #     optimizer_params: Optional[Dict[str, object]] = None,
    #     steps_per_epoch: int = None,
    #     evaluation_steps: int = 1,
    #     weight_decay: float = 0.0,
    #     max_grad_norm: float = None,
    #     monitor: Optional[str] = None,
    #     monitor_criterion: str = "max",
    #     load_best_model_at_last: bool = True,
    #     aux_data: Optional[Dict[str, object]] = None,
    #     ema_used: bool = False
    # ):
    #     """Trains the model.
    #
    #     Args:
    #         train_dataloader: Dataloader for training.
    #         val_dataloader: Dataloader for validation. Default is None.
    #         test_dataloader: Dataloader for testing. Default is None.
    #         epochs: Number of epochs. Default is 5.
    #         optimizer_class: Optimizer class. Default is torch.optim.Adam.
    #         optimizer_params: Parameters for the optimizer. Default is {"lr": 1e-3}.
    #         steps_per_epoch: Number of steps per epoch. Default is None.
    #         weight_decay: Weight decay. Default is 0.0.
    #         max_grad_norm: Maximum gradient norm. Default is None.
    #         monitor: Metric name to monitor. Default is None.
    #         monitor_criterion: Criterion to monitor. Default is "max".
    #         load_best_model_at_last: Whether to load the best model at the last.
    #             Default is True.
    #     """
    #     if optimizer_params is None:
    #         optimizer_params = {"lr": 1e-3}
    #
    #     # logging
    #     logger.info("Training:")
    #     logger.info(f"Batch size: {train_dataloader.batch_size}")
    #     logger.info(f"Optimizer: {optimizer_class}")
    #     logger.info(f"Optimizer params: {optimizer_params}")
    #     logger.info(f"Weight decay: {weight_decay}")
    #     logger.info(f"Max grad norm: {max_grad_norm}")
    #     logger.info(f"Val dataloader: {val_dataloader}")
    #     logger.info(f"Monitor: {monitor}")
    #     logger.info(f"Monitor criterion: {monitor_criterion}")
    #     logger.info(f"Epochs: {epochs}")
    #
    #     # set optimizer
    #     param = list(self.model.named_parameters())
    #     no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    #     optimizer_grouped_parameters = [
    #         {
    #             "params": [p for n, p in param if not any(nd in n for nd in no_decay)],
    #             "weight_decay": weight_decay,
    #         },
    #         {
    #             "params": [p for n, p in param if any(nd in n for nd in no_decay)],
    #             "weight_decay": 0.0,
    #         },
    #     ]
    #
    #     optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
    #
    #     # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
    #     #                                                        T_max=epochs//10)
    #     # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epochs + 1))
    #
    #     #ema
    #     if ema_used:
    #         ema = EMA(self.model, 0.999)
    #         ema.register()
    #
    #     # initialize
    #     data_iterator = iter(train_dataloader) # 生成迭代器，减少内存消耗
    #     best_score = -1 * float("inf") if monitor_criterion == "max" else float("inf")
    #     if steps_per_epoch == None:
    #         steps_per_epoch = len(train_dataloader)
    #     global_step = 0
    #     # epoch training loop
    #     for epoch in range(epochs):
    #         # if epoch > 0:
    #         #     self.kmeans(self.model.embeddings)
    #
    #         if warm_epochs is not None:
    #             if epoch > warm_epochs: # 冻结部分参数
    #                 self.set_params(True)
    #
    #         max_step, cur_step = len(train_dataloader), 0
    #
    #         training_loss = []
    #         self.model.zero_grad()
    #         self.model.train()
    #         # batch training loop
    #         logger.info("")
    #         for _ in trange(
    #             steps_per_epoch,
    #             desc=f"Epoch {epoch} / {epochs}",
    #             smoothing=0.05,
    #         ):
    #             try:
    #                 data = next(data_iterator)
    #             except StopIteration:
    #                 data_iterator = iter(train_dataloader)
    #                 data = next(data_iterator)
    #
    #             # forward
    #             cur_step +=1
    #             adjust_learning_rate(optimizer, cur_step, config, max_step) # 真的离谱，这个第二阶段感觉不能用，会造成模型过拟合。
    #
    #             output = self.model(**data) # 传入一个字典对象，下面根据其名称对应的取值 train_mode='fine_tune', 可以先放前面
    #             loss = output["loss"]
    #             # backward
    #             loss.backward()
    #             if max_grad_norm is not None:
    #                 torch.nn.utils.clip_grad_norm_(
    #                     self.model.parameters(), max_grad_norm
    #                 )
    #
    #             # 计算梯度
    #             # get_contribution(self.model, config)
    #
    #
    #
    #             # update
    #             optimizer.step()
    #             if ema_used:
    #                 ema.update()
    #
    #             optimizer.zero_grad()
    #             training_loss.append(loss.item())
    #             global_step += 1
    #         # scheduler.step()
    #
    #         # log and save
    #         logger.info(f"--- Train epoch-{epoch}, step-{global_step} ---")
    #         logger.info(f"loss: {sum(training_loss) / len(training_loss):.4f}")
    #         if self.exp_path is not None:
    #             pass
    #             # self.save_ckpt(os.path.join(self.exp_path, "last.ckpt"))
    #
    #         # validation
    #         if val_dataloader is not None:
    #             if ema_used:
    #                 ema.apply_shadow()
    #
    #             scores = self.evaluate(val_dataloader, aux_data)
    #             logger.info(f"--- Eval epoch-{epoch}, step-{global_step} ---")
    #             for key in scores.keys():
    #                 logger.info("{}: {}".format(key, scores[key]))
    #                 # logger.info("{}: {:.4f}".format(key, scores[key]))
    #
    #             # save best model
    #             if monitor is not None:
    #                 score = scores[monitor]
    #                 if is_best(best_score, score, monitor_criterion):
    #                     logger.info(
    #                         f"New best {monitor} score ({score:.4f}) "
    #                         f"at epoch-{epoch}, step-{global_step}"
    #                     )
    #                     best_score = score
    #
    #                     if self.exp_path is not None:
    #                         self.save_ckpt(os.path.join(self.exp_path, "best.ckpt"))
    #             if ema_used:
    #                 ema.restore()
    #
    #     # load best model
    #     if load_best_model_at_last and self.exp_path is not None and os.path.isfile(
    #         os.path.join(self.exp_path, "best.ckpt")):
    #         logger.info("Loaded best model")
    #         self.load_ckpt(os.path.join(self.exp_path, "best.ckpt"))
    #
    #     # test
    #     if test_dataloader is not None:
    #         if ema_used:
    #             ema.apply_shadow()
    #         scores = self.evaluate(test_dataloader, aux_data)
    #         logger.info(f"--- Test ---")
    #         for key in scores.keys():
    #             # logger.info("{}: {:.4f}".format(key, scores[key]))
    #             if key.endswith('grouped'):
    #                 logger.info("{}: {}".format(key, scores[key])) # 列表
    #             else:
    #                 logger.info("{}: {:4f}".format(key, scores[key])) # 浮点数
    #             # logger.info("{}: {}".format(key, scores[key])) # 这里会不会有问题【因为有topK】
    #         if ema_used:
    #             ema.restore()
    #     return
    def train(
            self,
            train_dataloader: DataLoader,
            val_dataloader: Optional[DataLoader] = None,
            test_dataloader: Optional[DataLoader] = None,
            epochs: int = 5,
            warm_epochs: Optional[int] = None,
            optimizer_class: Type[Optimizer] = torch.optim.AdamW,
            optimizer_params: Optional[Dict[str, object]] = None,
            steps_per_epoch: int = None,
            evaluation_steps: int = 1,
            weight_decay: float = 0.0,
            max_grad_norm: float = None,
            monitor: Optional[str] = None,
            monitor_criterion: str = "max",
            load_best_model_at_last: bool = True,
            aux_data: Optional[Dict[str, object]] = None,
            ema_used: bool = False,
            use_dp: bool = False,
            dp_config: Optional[Dict[str, object]] = None,
    ):
        if optimizer_params is None:
            optimizer_params = {"lr": 1e-3}

        if dp_config is None:
            dp_config = {}

        logger.info("Training:")
        logger.info(f"Batch size: {train_dataloader.batch_size}")
        logger.info(f"Optimizer: {optimizer_class}")
        logger.info(f"Optimizer params: {optimizer_params}")
        logger.info(f"Weight decay: {weight_decay}")
        logger.info(f"Max grad norm: {max_grad_norm}")
        logger.info(f"Val dataloader: {val_dataloader}")
        logger.info(f"Monitor: {monitor}")
        logger.info(f"Monitor criterion: {monitor_criterion}")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Use DP: {use_dp}")
        if use_dp:
            logger.info(f"DP config: {dp_config}")

        # ========= DP 模式：只训练 head =========
        if use_dp:
            _freeze_for_dp_head_only(self.model, logger=logger)

        # ========== DP 兼容性检查 ==========
        if use_dp:
            try:
                errors = ModuleValidator.validate(self.model, strict=False)
                if len(errors) > 0:
                    logger.info("[DP] incompatible modules found:")
                    for e in errors:
                        logger.info(f"[DP] {e}")
                else:
                    logger.info("[DP] model passed ModuleValidator.validate()")
            except Exception as e:
                logger.info(f"[DP] ModuleValidator failed: {e}")
                raise

        # ========== optimizer ==========
        param = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param
                    if p.requires_grad and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in param
                    if p.requires_grad and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer_grouped_parameters = _filter_empty_param_groups(optimizer_grouped_parameters)

        # 防御：如果一个可训练参数都没有，直接报错
        n_trainable = sum(p.numel() for _, p in self.model.named_parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {n_trainable}")
        if n_trainable == 0:
            raise RuntimeError("No trainable parameters left after DP freezing.")

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        # ========== EMA ==========
        # DP 首轮建议关闭 EMA，避免先引入额外复杂性
        if ema_used and use_dp:
            logger.info("[DP] EMA disabled for DP training.")
            ema_used = False

        if ema_used:
            ema = EMA(self.model, 0.999)
            ema.register()

        # ========== DP 包装 ==========
        privacy_engine = None
        if use_dp:
            privacy_engine = PrivacyEngine(accountant=dp_config.get("accountant", "rdp"))

            self.model, optimizer, train_dataloader = privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=optimizer,
                data_loader=train_dataloader,
                target_epsilon=dp_config["target_epsilon"],
                target_delta=dp_config["target_delta"],
                epochs=epochs,
                max_grad_norm=dp_config["max_grad_norm"],
                poisson_sampling=dp_config.get("poisson_sampling", True),
                # head-only 微调时 hooks 通常足够；若还出问题可改成 "functorch"
                grad_sample_mode=dp_config.get("grad_sample_mode", "functorch"),
                clipping=dp_config.get("clipping", "flat"),
            )
            logger.info("[DP] PrivacyEngine attached.")

        # make_private 后再取长度
        if steps_per_epoch is None:
            steps_per_epoch = len(train_dataloader)

        best_score = -1 * float("inf") if monitor_criterion == "max" else float("inf")
        global_step = 0

        # ========== epoch loop ==========
        for epoch in range(epochs):
            # DP 模式下不要在 make_private 后动态改 requires_grad
            if warm_epochs is not None:
                if (not use_dp) and epoch > warm_epochs:
                    self.set_params(True)
                elif use_dp and epoch > warm_epochs:
                    logger.info("[DP] skip dynamic set_params(True) after make_private")

            max_step, cur_step = len(train_dataloader), 0
            training_loss = []

            self.model.zero_grad()
            self.model.train()

            logger.info("")
            data_iterator = iter(train_dataloader)

            for _ in trange(
                    steps_per_epoch,
                    desc=f"Epoch {epoch} / {epochs}",
                    smoothing=0.05,
            ):
                try:
                    data = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(train_dataloader)
                    data = next(data_iterator)

                # Poisson sampling 下可能出现空 batch
                if data is None:
                    continue

                if isinstance(data, dict):
                    first_tensor = None
                    for v in data.values():
                        if torch.is_tensor(v):
                            first_tensor = v
                            break
                    if first_tensor is not None and first_tensor.shape[0] == 0:
                        continue

                cur_step += 1

                # 保留原来的 LR 调度
                adjust_learning_rate(optimizer, cur_step, config, max_step)

                optimizer.zero_grad()

                output = self.model(**data)
                loss = output["loss"]

                loss.backward()

                # 非 DP 时才做普通 global grad clipping
                if (not use_dp) and (max_grad_norm is not None):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                # DP 调试：检查缺失 grad_sample
                if use_dp:
                    bad = _find_missing_grad_sample(self.model)
                    if len(bad) > 0:
                        logger.info("[DP] missing grad_sample parameters:")
                        for n in bad:
                            logger.info(f"[DP] missing grad_sample: {n}")
                        raise RuntimeError(
                            "Some trainable parameters have no grad_sample. "
                            "Freeze them or make sure they participate in the returned loss."
                        )

                optimizer.step()

                if ema_used:
                    ema.update()

                training_loss.append(loss.item())
                global_step += 1

            logger.info(f"--- Train epoch-{epoch}, step-{global_step} ---")
            logger.info(f"loss: {sum(training_loss) / len(training_loss):.4f}")

            if use_dp and privacy_engine is not None:
                eps = privacy_engine.get_epsilon(delta=dp_config.get("target_delta", 1e-5))
                logger.info(
                    f"[DP] epoch-{epoch}, epsilon={eps:.6f}, "
                    f"delta={dp_config.get('target_delta', 1e-5)}"
                )

            # validation
            if val_dataloader is not None:
                if ema_used:
                    ema.apply_shadow()

                scores = self.evaluate(val_dataloader, aux_data)
                logger.info(f"--- Eval epoch-{epoch}, step-{global_step} ---")
                for key in scores.keys():
                    logger.info("{}: {}".format(key, scores[key]))

                if monitor is not None:
                    score = scores[monitor]
                    if is_best(best_score, score, monitor_criterion):
                        logger.info(
                            f"New best {monitor} score ({score:.4f}) "
                            f"at epoch-{epoch}, step-{global_step}"
                        )
                        best_score = score

                        if self.exp_path is not None:
                            self.save_ckpt(os.path.join(self.exp_path, "best.ckpt"))

                if ema_used:
                    ema.restore()

        if load_best_model_at_last and self.exp_path is not None and os.path.isfile(
                os.path.join(self.exp_path, "best.ckpt")
        ):
            logger.info("Loaded best model")
            self.load_ckpt(os.path.join(self.exp_path, "best.ckpt"))

        if test_dataloader is not None:
            if ema_used:
                ema.apply_shadow()

            scores = self.evaluate(test_dataloader, aux_data)
            logger.info(f"--- Test ---")
            for key in scores.keys():
                if key.endswith('grouped'):
                    logger.info("{}: {}".format(key, scores[key]))
                else:
                    logger.info("{}: {:4f}".format(key, scores[key]))

            if ema_used:
                ema.restore()

        return



    def train_gan(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        epochs: int = 5,
        warm_epochs: Optional[int] =None,
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        optimizer_params: Optional[Dict[str, object]] = None,
        steps_per_epoch: int = None,
        evaluation_steps: int = 1,
        weight_decay: float = 0.0,
        max_grad_norm: float = None,
        monitor: Optional[str] = None,
        monitor_criterion: str = "max",
        load_best_model_at_last: bool = True,
        aux_data: Optional[Dict[str, object]] = None,
        ema_used: bool = False
    ):
        """Trains the model.

        Args:
            train_dataloader: Dataloader for training.
            val_dataloader: Dataloader for validation. Default is None.
            test_dataloader: Dataloader for testing. Default is None.
            epochs: Number of epochs. Default is 5.
            optimizer_class: Optimizer class. Default is torch.optim.Adam.
            optimizer_params: Parameters for the optimizer. Default is {"lr": 1e-3}.
            steps_per_epoch: Number of steps per epoch. Default is None.
            weight_decay: Weight decay. Default is 0.0.
            max_grad_norm: Maximum gradient norm. Default is None.
            monitor: Metric name to monitor. Default is None.
            monitor_criterion: Criterion to monitor. Default is "max".
            load_best_model_at_last: Whether to load the best model at the last.
                Default is True.
        """
        if optimizer_params is None:
            optimizer_params = {"lr": 1e-3}

        # logging
        logger.info("Training:")
        logger.info(f"Batch size: {train_dataloader.batch_size}")
        logger.info(f"Optimizer: {optimizer_class}")
        logger.info(f"Optimizer params: {optimizer_params}")
        logger.info(f"Weight decay: {weight_decay}")
        logger.info(f"Max grad norm: {max_grad_norm}")
        logger.info(f"Val dataloader: {val_dataloader}")
        logger.info(f"Monitor: {monitor}")
        logger.info(f"Monitor criterion: {monitor_criterion}")
        logger.info(f"Epochs: {epochs}")

        # set optimizer
        param = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in param if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        # optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        # Optimizer GAN
        g_params = [{'params': self.model.generator.parameters()},
                    {'params': self.model.autoencoderDecoder.parameters(), 'lr': optimizer_params['lr']}]
        # g_params = list(model.generator.parameters()) + list(autoencoderModel.decoder.parameters())
        optimizer_G = torch.optim.Adam(g_params, lr=optimizer_params['lr'], betas=(optimizer_params['b1'], optimizer_params['b2']), weight_decay=weight_decay)
        optimizer_D = torch.optim.Adam(self.model.discriminator.parameters(), lr=optimizer_params['lr'], betas=(optimizer_params['b1'], optimizer_params['b2']),
                                       weight_decay=weight_decay)
        dis_step = optimizer_params['n_iter']

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
        #                                                        T_max=epochs//10)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epochs + 1))


        # initialize
        data_iterator = iter(train_dataloader) # 生成迭代器，减少内存消耗
        best_score = -1 * float("inf") if monitor_criterion == "max" else float("inf")
        if steps_per_epoch == None:
            steps_per_epoch = len(train_dataloader)

        global_step = 0
        # epoch training loop
        for epoch in range(epochs):
            if warm_epochs is not None:
                if epoch > warm_epochs: # 冻结部分参数
                    self.set_params(True)

            max_step, cur_step = len(train_dataloader), 0

            training_loss = []
            self.model.zero_grad()
            self.model.train()
            # batch training loop
            logger.info("")
            for _ in trange(
                steps_per_epoch,
                desc=f"Epoch {epoch} / {epochs}",
                smoothing=0.05,
            ):
                try:
                    data = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(train_dataloader)
                    data = next(data_iterator)

                # forward
                cur_step +=1
                # adjust_learning_rate(optimizer, cur_step, config, max_step) # 真的离谱，这个第二阶段感觉不能用，会造成模型过拟合。

                if cur_step % dis_step !=0 or cur_step==0:
                    for p in self.model.discriminator.parameters():
                        p.requires_grad = True
                    for p in self.model.generator.parameters():  # reset requires_grad
                        p.requires_grad = False
                    for p in self.model.autoencoder.parameters():
                        p.requires_grad = False
                    output = self.model.forward_dis(**data) # 传入一个字典对象，下面根据其名称对应的取值 train_mode='fine_tune', 可以先放前面

                else:
                    for p in self.model.generator.parameters():  # reset requires_grad
                        p.requires_grad = True
                    for p in self.model.autoencoder.parameters():
                        p.requires_grad = True
                    for p in self.model.discriminator.parameters():
                        p.requires_grad = False
                    output = self.model.forward_gen(**data)
                loss = output["loss"]
                # backward
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm
                    )
                # update
                if cur_step % dis_step !=0 or cur_step==0:
                    optimizer_D.step()
                    optimizer_D.zero_grad()
                else:
                    optimizer_G.step()
                    optimizer_G.zero_grad()

                training_loss.append(loss.item())
                global_step += 1
            # scheduler.step()

            # log and save
            logger.info(f"--- Train epoch-{epoch}, step-{global_step} ---")
            logger.info(f"loss: {sum(training_loss) / len(training_loss):.4f}")
            if self.exp_path is not None:
                pass
                # self.save_ckpt(os.path.join(self.exp_path, "last.ckpt"))

            # validation
            if val_dataloader is not None:
                scores = self.evaluate_gan(val_dataloader, aux_data)
                logger.info(f"--- Eval epoch-{epoch}, step-{global_step} ---")
                for key in scores.keys():
                    logger.info("{}: {}".format(key, scores[key]))
                    # logger.info("{}: {:.4f}".format(key, scores[key]))

                # save best model
                if monitor is not None:
                    score = scores[monitor]
                    if is_best(best_score, score, monitor_criterion):
                        logger.info(
                            f"New best {monitor} score ({score:.4f}) "
                            f"at epoch-{epoch}, step-{global_step}"
                        )
                        best_score = score

                        if self.exp_path is not None:
                            self.save_ckpt(os.path.join(self.exp_path, "best.ckpt"))


        # load best model
        if load_best_model_at_last and self.exp_path is not None and os.path.isfile(
            os.path.join(self.exp_path, "best.ckpt")):
            logger.info("Loaded best model")
            self.load_ckpt(os.path.join(self.exp_path, "best.ckpt"))

        # test
        if test_dataloader is not None:
            scores = self.evaluate(test_dataloader, aux_data)
            logger.info(f"--- Test ---")
            for key in scores.keys():
                # logger.info("{}: {:.4f}".format(key, scores[key]))
                if key.endswith('grouped'):
                    logger.info("{}: {}".format(key, scores[key])) # 列表
                else:
                    logger.info("{}: {:4f}".format(key, scores[key])) # 浮点数
        return


    def train_auto(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        epochs: int = 5,
        warm_epochs: Optional[int] =None,
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        optimizer_params: Optional[Dict[str, object]] = None,
        steps_per_epoch: int = None,
        evaluation_steps: int = 1,
        weight_decay: float = 0.0,
        max_grad_norm: float = None,
        monitor: Optional[str] = None,
        monitor_criterion: str = "max",
        load_best_model_at_last: bool = True,
        aux_data: Optional[Dict[str, object]] = None,
        ema_used: bool = False
    ):
        """Trains the model.

        Args:
            train_dataloader: Dataloader for training.
            val_dataloader: Dataloader for validation. Default is None.
            test_dataloader: Dataloader for testing. Default is None.
            epochs: Number of epochs. Default is 5.
            optimizer_class: Optimizer class. Default is torch.optim.Adam.
            optimizer_params: Parameters for the optimizer. Default is {"lr": 1e-3}.
            steps_per_epoch: Number of steps per epoch. Default is None.
            weight_decay: Weight decay. Default is 0.0.
            max_grad_norm: Maximum gradient norm. Default is None.
            monitor: Metric name to monitor. Default is None.
            monitor_criterion: Criterion to monitor. Default is "max".
            load_best_model_at_last: Whether to load the best model at the last.
                Default is True.
        """
        if optimizer_params is None:
            optimizer_params = {"lr": 1e-3}

        # logging
        logger.info("Training:")
        logger.info(f"Batch size: {train_dataloader.batch_size}")
        logger.info(f"Optimizer: {optimizer_class}")
        logger.info(f"Optimizer params: {optimizer_params}")
        logger.info(f"Weight decay: {weight_decay}")
        logger.info(f"Max grad norm: {max_grad_norm}")
        logger.info(f"Val dataloader: {val_dataloader}")
        logger.info(f"Monitor: {monitor}")
        logger.info(f"Monitor criterion: {monitor_criterion}")
        logger.info(f"Epochs: {epochs}")

        # set optimizer
        param = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in param if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
        #                                                        T_max=epochs//10)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epochs + 1))

        #ema
        if ema_used:
            ema = EMA(self.model, 0.999)
            ema.register()

        # initialize
        data_iterator = iter(train_dataloader) # 生成迭代器，减少内存消耗
        best_score = -1 * float("inf") if monitor_criterion == "max" else float("inf")
        if steps_per_epoch == None:
            steps_per_epoch = len(train_dataloader)
        global_step = 0
        # epoch training loop
        for epoch in range(epochs):
            # if epoch > 0:
            #     self.kmeans(self.model.embeddings)

            if warm_epochs is not None:
                if epoch > warm_epochs: # 冻结部分参数
                    self.set_params(True)

            max_step, cur_step = len(train_dataloader), 0

            training_loss = []
            self.model.zero_grad()
            self.model.train()
            # batch training loop
            logger.info("")
            for _ in trange(
                steps_per_epoch,
                desc=f"Epoch {epoch} / {epochs}",
                smoothing=0.05,
            ):
                try:
                    data = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(train_dataloader)
                    data = next(data_iterator)

                # forward
                cur_step +=1
                # adjust_learning_rate(optimizer, cur_step, config, max_step) # 真的离谱，这个第二阶段感觉不能用，会造成模型过拟合。

                output = self.model.forward_auto(**data) # 传入一个字典对象，下面根据其名称对应的取值 train_mode='fine_tune', 可以先放前面
                loss = output["loss"]
                # backward
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm
                    )
                # update
                optimizer.step()
                if ema_used:
                    ema.update()

                optimizer.zero_grad()
                training_loss.append(loss.item())
                global_step += 1
            # scheduler.step()

            # log and save
            logger.info(f"--- Train epoch-{epoch}, step-{global_step} ---")
            logger.info(f"loss: {sum(training_loss) / len(training_loss):.4f}")
            if self.exp_path is not None:
                pass
                # self.save_ckpt(os.path.join(self.exp_path, "last.ckpt"))

            # validation
            if val_dataloader is not None:
                if ema_used:
                    ema.apply_shadow()

                scores = self.evaluate(val_dataloader, aux_data)
                logger.info(f"--- Eval epoch-{epoch}, step-{global_step} ---")
                for key in scores.keys():
                    logger.info("{}: {}".format(key, scores[key]))
                    # logger.info("{}: {:.4f}".format(key, scores[key]))

                # save best model
                if monitor is not None:
                    score = scores[monitor]
                    if is_best(best_score, score, monitor_criterion):
                        logger.info(
                            f"New best {monitor} score ({score:.4f}) "
                            f"at epoch-{epoch}, step-{global_step}"
                        )
                        best_score = score

                        if self.exp_path is not None:
                            self.save_ckpt(os.path.join(self.exp_path, "best.ckpt"))
                if ema_used:
                    ema.restore()

        # load best model
        if load_best_model_at_last and self.exp_path is not None and os.path.isfile(
            os.path.join(self.exp_path, "best.ckpt")):
            logger.info("Loaded best model")
            self.load_ckpt(os.path.join(self.exp_path, "best.ckpt"))

        # test
        if test_dataloader is not None:
            if ema_used:
                ema.apply_shadow()
            scores = self.evaluate(test_dataloader, aux_data)
            logger.info(f"--- Test ---")
            for key in scores.keys():
                # logger.info("{}: {:.4f}".format(key, scores[key]))
                if key.endswith('grouped'):
                    logger.info("{}: {}".format(key, scores[key])) # 列表
                else:
                    logger.info("{}: {:4f}".format(key, scores[key])) # 浮点数
                # logger.info("{}: {}".format(key, scores[key])) # 这里会不会有问题【因为有topK】
            if ema_used:
                ema.restore()
        return


    def inference_gan(self, dataloader, additional_outputs=None,
                  return_patient_ids=False) -> Dict[str, float]:
        """Model inference.

        Args:
            dataloader: Dataloader for evaluation.
            additional_outputs: List of additional output to collect.
                Defaults to None ([]).

        Returns:
            y_true_all: List of true labels.
            y_prob_all: List of predicted probabilities.
            loss_mean: Mean loss over batches.
            additional_outputs (only if requested): Dict of additional results.
            patient_ids (only if requested): List of patient ids in the same order as y_true_all/y_prob_all.
        """
        loss_all = []
        y_true_all = []
        y_prob_all = []
        patient_ids = []

        if additional_outputs is not None:
            additional_outputs = {k: [] for k in additional_outputs}
        for data in tqdm(dataloader, desc="Evaluation"):
            self.model.eval()
            with torch.no_grad():
                output = self.model.forward_dis(**data) # 这里是看discriminator的输出
                loss = output["loss"]
                y_true = output["y_true"].cpu().numpy()
                y_prob = output["y_prob"].cpu().numpy()
                loss_all.append(loss.item())
                y_true_all.append(y_true)
                y_prob_all.append(y_prob)

                if additional_outputs is not None:
                    for key in additional_outputs.keys():
                        additional_outputs[key].append(output[key].cpu().numpy())
            if return_patient_ids:
                patient_ids.extend(data["patient_id"])
        loss_mean = sum(loss_all) / len(loss_all)

        # torch.save({'data': datas, 'pred': preds}, '/home/czhaobo/UDCHealth/src/data_pred.pt')

        y_true_all = np.concatenate(y_true_all, axis=0)
        y_prob_all = np.concatenate(y_prob_all, axis=0)
        outputs = [y_true_all, y_prob_all, loss_mean]
        if additional_outputs is not None:
            additional_outputs = {key: np.concatenate(val)
                                  for key, val in additional_outputs.items()}
            outputs.append(additional_outputs)
        if return_patient_ids:
            outputs.append(patient_ids)
        return outputs # y_true_all, y_prob_all, loss_mean, additional_outputs

    def evaluate_gan(self, dataloader, aux_data) -> Dict[str, float]:
        """Evaluates the model.

        Args:
            dataloader: Dataloader for evaluation.

        Returns:
            scores: a dictionary of scores.
        """
        if self.model.mode is not None: # mode指定了metric
            y_true_all, y_prob_all, loss_mean = self.inference_gan(dataloader, additional_outputs=None, return_patient_ids=False)
            mode = self.model.mode
            metrics_fn = get_metrics_fn(mode)
            scores = metrics_fn(y_true_all, y_prob_all, metrics=self.metrics)
            scores["loss"] = loss_mean
        else:
            loss_all = []
            for data in tqdm(dataloader, desc="Evaluation"):
                self.model.eval()
                with torch.no_grad():
                    output = self.model(**data)
                    loss = output["loss"]
                    loss_all.append(loss.item())
            loss_mean = sum(loss_all) / len(loss_all)
            scores = {"loss": loss_mean}
        return scores


    def inference(self, dataloader, additional_outputs=None,
                  return_patient_ids=False) -> Dict[str, float]:
        """Model inference.

        Args:
            dataloader: Dataloader for evaluation.
            additional_outputs: List of additional output to collect.
                Defaults to None ([]).

        Returns:
            y_true_all: List of true labels.
            y_prob_all: List of predicted probabilities.
            loss_mean: Mean loss over batches.
            additional_outputs (only if requested): Dict of additional results.
            patient_ids (only if requested): List of patient ids in the same order as y_true_all/y_prob_all.
        """
        loss_all = []
        y_true_all = []
        y_prob_all = []
        patient_ids = []

        datas, preds = [], []  # add
        base_model = _unwrap_model(self.model)

        if additional_outputs is not None:
            additional_outputs = {k: [] for k in additional_outputs}
        for data in tqdm(dataloader, desc="Evaluation"):
            base_model.eval()
            with torch.no_grad():
                output = base_model(train_mode='finetune', **data) # 永远只看fine_tune(便于smart)，不然最好改成不同的trainer，这样可以指定多个metric
                loss = output["loss"]
                y_true = output["y_true"].cpu().numpy()
                y_prob = output["y_prob"].cpu().numpy()
                loss_all.append(loss.item())
                y_true_all.append(y_true)
                y_prob_all.append(y_prob)

                # f1_score, pred = cal_data(self.model, output['y_prob'], output['labels_copy']) # med set
                # print(f1_score, len(output['labels_copy'][0])) # 记得test_loader的batch设小点。
                # if 8<len(output['labels_copy'][0])<=12 and f1_score[-3]>=0.35:
                #     datas.append(data)
                #     preds.append(pred)


                if additional_outputs is not None:
                    for key in additional_outputs.keys():
                        additional_outputs[key].append(output[key].cpu().numpy())
            if return_patient_ids:
                patient_ids.extend(data["patient_id"])
        loss_mean = sum(loss_all) / len(loss_all)

        # torch.save({'data': datas, 'pred': preds}, '/home/czhaobo/MMHealth/src/data_pred.pt')

        y_true_all = np.concatenate(y_true_all, axis=0)
        y_prob_all = np.concatenate(y_prob_all, axis=0)
        # print("AAAAAAAAA", y_true_all.shape, y_prob_all.shape, np.unique(y_true_all)) # trainset要重新生成。
        outputs = [y_true_all, y_prob_all, loss_mean]
        if additional_outputs is not None:
            additional_outputs = {key: np.concatenate(val)
                                  for key, val in additional_outputs.items()}
            outputs.append(additional_outputs)
        if return_patient_ids:
            outputs.append(patient_ids)
        return outputs # y_true_all, y_prob_all, loss_mean, additional_outputs

    def evaluate(self, dataloader, aux_data) -> Dict[str, float]:
        """Evaluates the model.

        Args:
            dataloader: Dataloader for evaluation.

        Returns:
            scores: a dictionary of scores.
        """
        base_model = _unwrap_model(self.model)
        if base_model.mode is not None: # mode指定了metric
            y_true_all, y_prob_all, loss_mean, patient_ids = self.inference(dataloader, additional_outputs=None, return_patient_ids=True)
            mode = base_model.mode
            metrics_fn = get_metrics_fn(mode)
            scores = metrics_fn(y_true_all, y_prob_all, metrics=self.metrics, aux_data=aux_data, patient_ids=patient_ids)
            scores["loss"] = loss_mean
        else:
            loss_all = []
            for data in tqdm(dataloader, desc="Evaluation"):
                base_model.eval()
                with torch.no_grad():
                    output = base_model(**data)
                    loss = output["loss"]
                    loss_all.append(loss.item())
            loss_mean = sum(loss_all) / len(loss_all)
            scores = {"loss": loss_mean}
        return scores

    # def save_ckpt(self, ckpt_path: str) -> None:
    #     """Saves the model checkpoint."""
    #     state_dict = self.model.state_dict()
    #     torch.save(state_dict, ckpt_path)
    #     return
    #
    # def load_ckpt(self, ckpt_path: str) -> None:
    #     """Saves the model checkpoint."""
    #     state_dict = torch.load(ckpt_path, map_location=self.device)
    #     self.model.load_state_dict(state_dict, strict=False)
    #     return

    def save_ckpt(self, path):
        model_to_save = self.model._module if hasattr(self.model, "_module") else self.model
        torch.save(model_to_save.state_dict(), path)

    def load_ckpt(self, path):
        state_dict = torch.load(path, map_location=self.device)
        model_to_load = self.model._module if hasattr(self.model, "_module") else self.model
        model_to_load.load_state_dict(state_dict, strict=False)





class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


from utils import f1_cal
def cal_data(model, y_prob, label):
    pred = model.decode_label(y_prob.cpu().detach().numpy(), model.label_tokenizer)
    # print(pred[0], label[0])
    f1_score = f1_cal(pred[0], label[0])
    return f1_score, pred[0]
# 初始化
# ema = EMA(model, 0.999)
# ema.register()

# # 训练过程中，更新完参数后，同步update shadow weights
# def train():
#     optimizer.step()
#     ema.update()
#
# # eval前，apply shadow weights；eval之后，恢复原来模型的参数
# def evaluate():
#     ema.apply_shadow()
#     # evaluate
#     ema.restore()

if __name__ == "__main__":
    pass
