# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : loader.py
# Time       ：4/11/2024 3:25 pm
# Author     ：Chuang Zhao
# version    ：python 
# Description：several dataloader, 关于数据的放在这里，不同的方法的可以在baseline_utils修改
"""
import torch
from torch.utils.data import DataLoader, Dataset
from baseline_utils import collate_fn_corgan, collate_fn_smart, collate_fn_prism



def collate_fn_dict(batch):
    return {key: [d[key] for d in batch] for key in batch[0]} # conditions: [B,V,M]

def get_dataloader(dataset, batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn_dict):
    """for first, third stage"""
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=drop_last
    )

    return dataloader


def _to01(x):
    try:
        return 0 if int(x) == 0 else 1
    except Exception:
        return 1

def _get_inc_matrix(d):
    """
    返回 inc 矩阵：list[list] 形状约为 (V,3) 或 (1,3)
    兼容 incomplete_raw: [3] / incomplete: [[3]] / incomplete: [V,3]
    """
    if 'incomplete_raw' in d:
        inc = d['incomplete_raw']
        if isinstance(inc, list) and len(inc) == 3 and not isinstance(inc[0], list):
            return [inc]

    if 'incomplete' in d:
        inc = d['incomplete']
        # [[a,b,c]] 或 [[...],[...],...]
        if isinstance(inc, list) and len(inc) > 0 and isinstance(inc[0], list):
            return inc
        # [a,b,c]
        if isinstance(inc, list) and len(inc) == 3 and not isinstance(inc[0], list):
            return [inc]

    return [[0, 0, 0]]

def _is_list_of_list(x):
    return isinstance(x, list) and len(x) > 0 and isinstance(x[0], list)

def _match_structure(new_val, orig_val):
    # 让 new 尽量匹配 orig 的嵌套结构，避免 tokenizer/validate 出错
    if _is_list_of_list(orig_val):
        if _is_list_of_list(new_val):
            return new_val
        return [new_val]
    else:
        if _is_list_of_list(new_val):
            if len(new_val) == 1:
                return new_val[0]
            flat = []
            for vv in new_val:
                flat.extend(vv)
            return flat
        return new_val

def _merge_view(orig_val, new_val, inc_col_01):
    """
    按 visit 维度融合（若 inc 是 Vx3 且 orig/new 都是 list-of-list）
    否则按样本级融合（inc_col_01 只有一个值）
    """
    # visit-wise
    if len(inc_col_01) > 1 and _is_list_of_list(orig_val) and _is_list_of_list(new_val):
        V = min(len(orig_val), len(new_val), len(inc_col_01))
        merged = []
        for v in range(V):
            merged.append(new_val[v] if inc_col_01[v] == 1 else orig_val[v])
        if len(orig_val) > V:
            merged.extend(orig_val[V:])
        return merged

    # sample-wise
    miss = inc_col_01[0] if len(inc_col_01) > 0 else 0
    if miss == 1 and new_val is not None:
        return _match_structure(new_val, orig_val)
    return orig_val

def collate_fn_dict_merge_new(batch):
    """
    不修改样本本身：batch 输出里把 conditions/procedures/drugs 替换成融合后的版本：
    - 缺失视图 -> new_*
    - 不缺失 -> 原始
    """
    out = {key: [d.get(key, None) for d in batch] for key in batch[0]}

    feats = ['conditions', 'procedures', 'drugs']
    for j, feat in enumerate(feats):
        merged_list = []
        new_key = f'new_{feat}'
        for d in batch:
            inc = _get_inc_matrix(d)
            inc_col = [_to01(row[j]) for row in inc]  # per-visit or single
            orig_val = d.get(feat, None)
            new_val = d.get(new_key, None)

            # 如果没有 orig_val，就原样
            if orig_val is None:
                merged_list.append(orig_val)
                continue

            merged_list.append(_merge_view(orig_val, new_val, inc_col))
        out[feat] = merged_list

    return out


# def get_note():
#     pass
#
def collate_fn_note(batch):
    data = {key: [d[key] for d in batch] for key in batch[0]} # 干脆直接在论文里处理。就是model forward多一个部分
    # data['note'] = [[d['note']] * len(d['conditions']) for d in batch] # 把这个note复制到每个condition上，不然存储的太大了。
    return data # conditions: [B,V,M]


# def get_special_input(config):
#     if config['MODEL']== 'PRISM':
#         return collate_fn_prism
#     elif config['DATASET'] == 'MIV-Note':
#         return collate_fn_note
#     else:
#         return collate_fn_dict
def get_special_input(config):
    if config['MODEL'] == 'PRISM':
        return collate_fn_prism
    elif config['DATASET'] == 'MIV-Note':
        return collate_fn_note
    else:
        # 开启“缺失用new_融合”，但不改 pkl
        if config.get('USE_NEW_MERGE', False):
            return collate_fn_dict_merge_new
        return collate_fn_dict
