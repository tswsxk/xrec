# coding: utf-8
# create by tongshiwei on 2019-9-1
__all__ = ["fit_f", "eval_f"]

import numpy as np
from itertools import chain
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import autograd
from tqdm import tqdm

from longling import as_list
from longling.ML.MxnetHelper.toolkit.ctx import split_and_load
from sklearn.metrics import mean_squared_error, median_absolute_error, ndcg_score
from longling.ML.metrics import classification_report


# Pairwise
def _fit_f(_net, _data, bp_loss_f, loss_function, loss_monitor):
    a, pos, neg = _data
    # todo modify the input to net
    pos_output = _net(a, pos)
    neg_output = _net(a, neg)
    bp_loss = None
    for name, func in loss_function.items():
        # todo modify the input to func
        loss = func(pos_output, neg_output)
        if name in bp_loss_f:
            bp_loss = loss
        loss_value = nd.mean(loss).asscalar()
        if loss_monitor:
            loss_monitor.update(name, loss_value)
    return bp_loss


# Pointwise
# def _fit_f(_net, _data, bp_loss_f, loss_function, loss_monitor):
#     user, item, label = _data
#     # todo modify the input to net
#     output = _net(user, item)
#     bp_loss = None
#     for name, func in loss_function.items():
#         # todo modify the input to func
#         loss = func(output, label.astype("float32"))
#         if name in bp_loss_f:
#             bp_loss = loss
#         loss_value = nd.mean(loss).asscalar()
#         if loss_monitor:
#             loss_monitor.update(name, loss_value)
#     return bp_loss


def _hit_rate(y_true, k):
    hit_count = 0
    for _y_true in y_true:
        if sum(_y_true[:k]) >= 1:
            hit_count += 1
    return hit_count / len(y_true)


def eval_f(_net, test_data, ctx=mx.cpu()):
    k = test_data[1]["k"]
    k = as_list(k) if k is not None else []
    max_k = max(k) if k else None
    top_k_ground_truth = []
    top_k_prediction = []
    ground_truth = []
    prediction = []

    for batch_data in tqdm(test_data[0], "evaluating"):
        ctx_data = split_and_load(
            ctx, *batch_data,
            even_split=False
        )
        for (user, item, label) in ctx_data:
            output = _net(user, item)
            pred = output
            label = label.asnumpy().astype("int")
            pred = pred.asnumpy()
            ground_truth.append(label.tolist())
            prediction.append(pred.tolist())
            if max_k:
                top_k_indices = np.argsort(pred)[::-1]
                _top_k_indices = top_k_indices[:max_k]
                padding = [0] * (max_k - len(_top_k_indices)) if len(_top_k_indices) < max_k else []
                top_k_prediction.append(pred[_top_k_indices].tolist() + padding)
                top_k_ground_truth.append(label[_top_k_indices].tolist() + padding)

    chained_ground_truth = list(chain(*ground_truth))
    chained_prediction = list(chain(*prediction))
    metrics = {
        "rmse": mean_squared_error(chained_ground_truth, chained_prediction),
        "mae": median_absolute_error(chained_ground_truth, chained_prediction),
    }

    metrics.update(classification_report(
        chained_ground_truth,
        [0 if v < 0.5 else 1 for v in chained_prediction],
        chained_prediction
    ))

    if k:
        metrics_k = {
            "ndcg": {},
            "HR": {}
        }
        for _k in k:
            metrics_k["ndcg"][_k] = ndcg_score(top_k_ground_truth, top_k_prediction, k=_k)
            metrics_k["HR"][_k] = _hit_rate(top_k_ground_truth, k=_k)
        metrics.update(metrics_k)
    return metrics


def fit_f(net, batch_size, batch_data,
          trainer, bp_loss_f, loss_function, loss_monitor=None,
          ctx=mx.cpu()):
    """
    Defined how each step of batch train goes

    Parameters
    ----------
    net: HybridBlock
        The network which has been initialized
        or loaded from the existed model
    batch_size: int
            The size of each batch
    batch_data: Iterable
        The batch data for train
    trainer:
        The trainer used to update the parameters of the net
    bp_loss_f: dict with only one value and one key
        The function to compute the loss for the procession
        of back propagation
    loss_function: dict of function
        Some other measurement in addition to bp_loss_f
    loss_monitor: LossMonitor
        Default to ``None``
    ctx: Context or list of Context
        Defaults to ``mx.cpu()``.

    Returns
    -------

    """
    ctx_data = split_and_load(
        ctx, *batch_data,
        even_split=False
    )

    with autograd.record():
        for _data in ctx_data:
            bp_loss = _fit_f(
                net, _data, bp_loss_f, loss_function, loss_monitor
            )
            assert bp_loss is not None
            bp_loss.backward()
    # todo: confirm whether the train step is equal to batch_size
    trainer.step(batch_size)
