# coding: utf-8
# 2020/6/14 @ tongshiwei

import json
from tqdm import tqdm
from longling import as_io, as_out_io
from longling.ML.toolkit.dataset import train_test


def movielens(src, tar, separator):
    with as_io(src) as f, as_out_io(tar) as wf:
        for line in tqdm(f, "reformatting from %s to %s" % (src, tar)):
            user, item, rating, _ = line.strip().split(separator)
            print(json.dumps([int(user), int(item), int(rating)]), file=wf)


if __name__ == '__main__':
    movielens("ratings.dat", "ml1m.jsonl", "::")
    train_test("ml1m.jsonl")
