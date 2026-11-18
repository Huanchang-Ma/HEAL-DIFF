import numpy as np
import torch
import torch.nn.functional as F

@torch.no_grad()
def collect_downstream_per_sample_loss(
    model,
    loader,
    device,
    mode="multilabel",
    max_batches=None,
    return_ids=False,
    auto_logits=True,   # 自动判断 y_prob 是 prob 还是 logits
):
    model.eval()
    model.to(device)

    all_losses = []
    all_ids = []

    for bi, item in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break

        # 兼容：loader 可能返回 (indices, batch_dict) 或 batch_dict
        if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], dict):
            batch = item[1]
        elif isinstance(item, dict):
            batch = item
        else:
            raise TypeError(f"Unexpected batch type: {type(item)}")

        #  forward（模型内部会做 tokenizer，并把 labels to(device)）
        out = model(**batch)

        y_prob = out["y_prob"]   # [B,C]（multilabel）或 [B,C]（multiclass概率/或logits）
        y_true = out["y_true"]   # multilabel: [B,C]  其他: [B] 或 [B,C]

        if mode == "multilabel":
            # ---- 自动判别 prob vs logits ----
            if auto_logits:
                is_prob = (float(y_prob.min()) >= 0.0) and (float(y_prob.max()) <= 1.0)
            else:
                is_prob = True

            if is_prob:
                # 防止 log(0) 的数值问题
                y_prob2 = y_prob.clamp(1e-6, 1 - 1e-6)
                per = F.binary_cross_entropy(y_prob2, y_true, reduction="none").sum(dim=1)  # [B]
            else:
                per = F.binary_cross_entropy_with_logits(y_prob, y_true, reduction="none").sum(dim=1)

        elif mode in ["multiclass", "binary"]:
            # y_prob 当成概率分布
            eps = 1e-12
            if y_true.ndim > 1:
                y_true_idx = y_true.argmax(dim=1)
            else:
                y_true_idx = y_true
            per = -torch.log(y_prob[torch.arange(y_prob.size(0), device=y_prob.device), y_true_idx] + eps)

        else:
            raise ValueError(f"Unknown mode={mode}")

        all_losses.append(per.detach().cpu())

        if return_ids:
            pid = batch.get("patient_id", None)
            vid = batch.get("visit_id", None)
            vih = batch.get("visit_id_hist", None)
            B = per.size(0)
            for i in range(B):
                all_ids.append({
                    "patient_id": pid[i] if isinstance(pid, list) else None,
                    "visit_id": vid[i] if isinstance(vid, list) else None,
                    "visit_id_hist": vih[i] if isinstance(vih, list) else None,
                })

    losses = torch.cat(all_losses, dim=0).numpy() if len(all_losses) else np.array([])
    return (losses, all_ids) if return_ids else losses


# def mia_attackP_from_losses(train_losses, test_losses, alphas=(0.05, 0.01, 0.001)):
#     res = {}
#     if len(test_losses) == 0 or len(train_losses) == 0:
#         return res
#
#     for a in alphas:
#         thr = float(np.quantile(test_losses, a))
#         tpr = float((train_losses <= thr).mean())
#         fpr = float((test_losses  <= thr).mean())
#         res[a] = {
#             "threshold": thr,
#             "TPR": tpr,
#             "FPR": fpr,
#             "ADV": float(tpr - fpr),  # 越大越“可区分” => 越可能泄露
#             "N_train": int(len(train_losses)),
#             "N_test": int(len(test_losses)),
#         }
#     return res
def mia_attackP_from_losses(train_losses, test_losses, alphas=(0.7,0.6,0.5,0.4,0.3,0.2,0.1)):
    # train_losses/test_losses: list[float]
    res = {}
    n_tr = len(train_losses)
    n_te = len(test_losses)

    # 排序一次，加速分位数阈值
    te_sorted = sorted(test_losses)

    def quantile_sorted(arr_sorted, q):
        # q in (0,1); 取最简单的 nearest-rank
        idx = int(round(q * (len(arr_sorted)-1)))
        idx = max(0, min(len(arr_sorted)-1, idx))
        return arr_sorted[idx]

    for a in alphas:
        thr = quantile_sorted(te_sorted, a)  # 用 holdout(test) 的分位数做阈值
        tpr = sum(1 for x in train_losses if x <= thr) / max(1, n_tr)
        fpr = sum(1 for x in test_losses  if x <= thr) / max(1, n_te)
        res[a] = {"thr": float(thr), "tpr": float(tpr), "fpr": float(fpr)}
    return res


def downstream_mia_attackP(
    model, train_loader, test_loader, device, mode="multilabel",
    alphas=(0.05, 0.01, 0.001),
    max_batches_train=None, max_batches_test=None,
    print_topk=20,  #找“最像 member”的样本
):
    (train_losses, train_ids) = collect_downstream_per_sample_loss(
        model, train_loader, device, mode=mode, max_batches=max_batches_train, return_ids=True
    )
    (test_losses, test_ids) = collect_downstream_per_sample_loss(
        model, test_loader, device, mode=mode, max_batches=max_batches_test, return_ids=True
    )

    mia_res = mia_attackP_from_losses(train_losses, test_losses, alphas=alphas)

    # ---- 额外：打印最可疑的 train 样本（loss 最小）----
    if print_topk and len(train_losses) > 0:
        order = np.argsort(train_losses)[:print_topk]
        print("\n==== [Downstream MIA] Most-suspected MEMBERS (lowest train loss) ====")
        for r, idx in enumerate(order, 1):
            print(f"[{r}] loss={train_losses[idx]:.6f} id={train_ids[idx]}")

    return mia_res, train_losses, test_losses
