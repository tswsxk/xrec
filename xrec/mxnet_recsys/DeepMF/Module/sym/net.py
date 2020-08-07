# coding: utf-8
# create by tongshiwei on 2019-9-1

__all__ = ["get_net", "get_bp_loss"]

# todo: define your network symbol and back propagation loss function

import mxnet as mx
from mxnet import gluon


def get_net(**kwargs):
    return NeuralMatrixFactorization(**kwargs)


def get_bp_loss(**kwargs):
    # return {"LogisticLoss": gluon.loss.LogisticLoss(label_format="binary")}
    return {"PairwiseLoss": PairwiseLoss(margin=0.5)}


class PairwiseLoss(gluon.loss.Loss):
    def __init__(self, weight=None, batch_axis=-1, margin=0.0, **kwargs):
        super(PairwiseLoss, self).__init__(weight, batch_axis, **kwargs)
        self.margin = margin

    def hybrid_forward(self, F, pos, neg, *args, **kwargs):
        loss = F.add(neg - pos, self.margin)
        loss = F.relu(loss)
        return loss


class NeuralMatrixFactorization(gluon.HybridBlock):
    def __init__(self, num_a, num_b, vec_dim, op="linear", prefix=None, params=None):
        super(NeuralMatrixFactorization, self).__init__(prefix, params)
        self.op = op
        self._op = None
        self.num_a = num_a
        self.num_b = num_b
        self.vec_dim = vec_dim
        with self.name_scope():
            self.embeddings_a = gluon.nn.HybridSequential()
            self.embeddings_a.add(
                gluon.nn.Embedding(self.num_a, self.vec_dim),
            )
            self.embeddings_b = gluon.nn.HybridSequential()
            self.embeddings_b.add(
                gluon.nn.Embedding(self.num_b, self.vec_dim),
            )
            self._op = None
            if self.op in {"linear", "mlp"}:
                self.embeddings_a.add(
                    gluon.nn.Activation("relu")
                )
                self.embeddings_b.add(
                    gluon.nn.Activation("relu")
                )
                if self.op == "mlp":
                    self.embeddings_a.add(gluon.nn.Dense(self.vec_dim, flatten=False))
                    self.embeddings_b.add(gluon.nn.Dense(self.vec_dim, flatten=False))
            elif self.op == "nn":
                self._op = gluon.nn.HybridSequential()
                self._op.add(
                    gluon.nn.Dense(self.vec_dim),
                )
            elif self.op == "cls":
                self._op = gluon.nn.HybridSequential()
                self._op.add(
                    gluon.nn.Activation("sigmoid")
                )

    def hybrid_forward(self, F, user, item, *args, **kwargs):
        embedding_a = self.embeddings_a(user)
        embedding_b = self.embeddings_b(item)

        if self.op in {"linear", "mlp"}:
            return F.sum(embedding_a * embedding_b, axis=-1)
        elif self.op == "nn":
            return self._op(F.concat(embedding_a, embedding_b)).flatten()
        elif self.op == "cls":
            return self._op(F.sum(embedding_a * embedding_b, axis=-1))
