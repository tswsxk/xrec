# coding: utf-8
# 2020/6/14 @ tongshiwei

import json
from tqdm import tqdm
from longling import as_io, as_out_io
import random
from collections import defaultdict


def prepare_ranking_file(src, tar, item_num, threshold=None, sampling_num=None, unified_num=False, excluded_files=None):
    user_items = {}
    with as_io(src) as f:
        for line in tqdm(f, "preparing ranking file"):
            user, item, rating = json.loads(line)
            user = int(user)
            item = int(item)
            if user not in user_items:
                user_items[user] = [[], [], []]  # like, unlabeled, dislike
            rating = float(rating)
            if threshold is not None:
                rating = 0 if rating <= threshold else 1
                pos = 0 if rating == 1 else 2
            else:
                pos = 0

            user_items[user][pos].append(item)

    excluded_user_items = defaultdict(set)
    if excluded_files:
        with as_io(excluded_files) as f:
            for line in f:
                user, item, _ = json.loads(line)
                user = int(user)
                item = int(item)
                excluded_user_items[user].add(item)

    for user, items in tqdm(user_items.items(), "sampling"):
        current_items = set(items[0]) | set(items[2]) | set(items[1])
        unlabeled = set(range(item_num)) - current_items - excluded_user_items.get(user, set())
        if sampling_num:
            if unified_num:
                _sampling_num = sampling_num - len(current_items)
            else:
                _sampling_num = sampling_num
            items[1].extend(random.sample(unlabeled, _sampling_num))
        else:
            items[1].extend(list(unlabeled))

    with as_out_io(tar) as wf:
        for user, items in tqdm(user_items.items(), "write to %s" % tar):
            _data = [user] + items
            print(json.dumps(_data), file=wf)


if __name__ == '__main__':
    # prepare_ranking_file("ml1m_train.jsonl", "u1_test_full.jsonl", 1682, 3)
    # prepare_ranking_file("ml1m_train.jsonl", "u1_test_100.jsonl", 1682, 3, 100)
    # prepare_ranking_file("ml1m_train.jsonl", "u1_test_100_uni.jsonl", 1682, 3, 100, unified_num=True)

    # prepare_ranking_file("ml1m_train.jsonl", "u1_test_full.jsonl", 1682, 3)
    # prepare_ranking_file("ml1m_train.jsonl", "u1_test_100.jsonl", 1682, 3, 100)
    prepare_ranking_file(
        "ml1m.test.jsonl", "ml1m_test_100_uni.jsonl", 3900, 3, 100, excluded_files="ml1m.train.jsonl"
    )
