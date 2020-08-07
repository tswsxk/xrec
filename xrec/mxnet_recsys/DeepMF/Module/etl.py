# coding: utf-8
# create by tongshiwei on 2019/4/12


from longling import LoopIter, AsyncLoopIter, iterwrap
import mxnet as mx
import itertools
import json
from collections import defaultdict
from mxnet import gluon
from tqdm import tqdm
from longling import as_io

__all__ = ["extract", "transform", "etl", "pseudo_data_iter", "etl_eval"]

sampler = None


# todo: define extract-transform-load process and implement the pesudo data iterator for testing

def pseudo_data_iter(_cfg):
    def pseudo_data_generation():
        # 在这里定义测试用伪数据流
        import random
        random.seed(10)

        raw_data = [
            [
                random.randint(0, 999),
                random.randint(0, 99),
                random.randint(0, 1)
            ]
            for _ in range(100)
        ]
        return raw_data

    def pseudo_eval_data_generation():
        # 在这里定义测试用伪数据流
        import random
        random.seed(10)

        raw_data = [
            [
                random.randint(0, 999),
                [random.randint(0, 99) for _ in range(3)],
                [random.randint(0, 99) for _ in range(3)],
                [random.randint(0, 99) for _ in range(3)],
            ]
            for _ in range(10)
        ]
        return raw_data

    return load(transform(pseudo_data_generation(), _cfg), _cfg), [load(
        transform_eval(pseudo_eval_data_generation(), _cfg), _cfg), _cfg.eval_params]


def extract(data_src):
    user_item_rating = []
    with as_io(data_src) as f:
        for line in tqdm(f, "extracting file"):
            _user_item_rating = []
            user_id, item_id, rating = json.loads(line)
            _user_item_rating.append(int(user_id))
            _user_item_rating.append(int(item_id))
            if int(rating) <= 3:
                _user_item_rating.append(0)
            else:
                _user_item_rating.append(1)
            user_item_rating.append(_user_item_rating)
    return user_item_rating


def transform(raw_data, params):
    # 定义数据转换接口
    # raw_data --> batch_data

    batch_size = params.batch_size

    return gluon.data.DataLoader(
        gluon.data.ArrayDataset(*list(zip(*raw_data))),
        batch_size
    )


@iterwrap(tank_size=16)
def sampling_transform(raw_data, params):
    batch_size = params.batch_size
    neg_num = params.data_params["neg_num"]

    global sampler

    from xrec.utils.datasets.sampler import PairwiseRandomSampler

    if sampler is None:
        if params.data_params.get("bank"):
            sampler = PairwiseRandomSampler(candidates_file=params.data_params["bank"])
        else:
            sampler = PairwiseRandomSampler(params.data_params["b_num"])

    batch_data = []
    for _ in range(neg_num):
        for _data in sampler(*raw_data):
            if len(batch_data) == batch_size:
                yield [
                    mx.nd.array(_batch_data, dtype="int") for _batch_data in zip(*batch_data)
                ]
                batch_data = []
            batch_data.append(_data)
    if batch_data:
        yield [
            mx.nd.array(_batch_data, dtype="int") for _batch_data in zip(*batch_data)
        ]


def load(transformed_data, params):
    return transformed_data


def etl(*args, params):
    raw_data = extract(*args)
    # transformed_data = transform(raw_data, params)
    transformed_data = sampling_transform(raw_data, params)
    return load(transformed_data, params)


def extract_eval(src):
    src_data = []
    with as_io(src) as f:
        for line in f:
            user, like, unlabeled, dislike = json.loads(line)
            src_data.append([user, like, unlabeled, dislike])
    return src_data


@LoopIter.wrap
def transform_eval(raw_data, params):
    # 定义数据转换接口
    # raw_data --> batch_data

    unlabeled_value = params.eval_params["unlabeled_value"]
    pointwise = params.eval_params["pointwise"]
    raw_data = raw_data

    if not pointwise:
        for user, like, unlabeled, dislike in raw_data:
            user_like = itertools.product([user], like, [1])
            user_unlabeled = itertools.product([user], unlabeled, [unlabeled_value])
            user_dislike = itertools.product([user], dislike, [0])
            user, item, rating = list(zip(*user_like, *user_unlabeled, *user_dislike))
            _data = [
                mx.nd.array(user),
                mx.nd.array(item),
                mx.nd.array(rating),
            ]
            yield _data
    else:
        for user, like, unlabeled, dislike in raw_data:
            user_dislike = list(itertools.product([user], dislike, [0]))
            for _like in like:
                user_like = itertools.product([user], [_like], [1])
                _user, _item, _rating = list(zip(*user_like, *user_dislike))
                _data = [
                    mx.nd.array(_user),
                    mx.nd.array(_item),
                    mx.nd.array(_rating),
                ]
                yield _data


def etl_eval(*args, params):
    raw_data = extract_eval(*args)
    eval_data = transform_eval(raw_data, params)
    info = params.eval_params
    return eval_data, info


if __name__ == '__main__':
    from longling.lib.structure import AttrDict
    import os

    filename = "../../../../data/ml-100k/u1.base"
    print(os.path.abspath(filename))

    for data in tqdm(extract(filename)):
        pass

    parameters = AttrDict({"batch_size": 128})
    for data in tqdm(etl(filename, params=parameters)):
        pass
