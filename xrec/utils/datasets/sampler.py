# coding: utf-8
# 2020/6/16 @ tongshiwei

import random
import json
from longling import as_io


class PairwiseRandomSampler(object):
    def __init__(self, objs_num=None, candidates=None, candidates_file=None):
        self.objs_num = objs_num
        self.candidates = {}
        if candidates:
            if isinstance(candidates, dict):
                self.candidates = candidates
            elif isinstance(candidates, list):
                self.candidates_from_iter(candidates)

        elif candidates_file:
            self.candidates_from_file(candidates_file)

    def candidates_from_iter(self, obj):
        for a, pos_b, implicit_b, neg_b in obj:
            if a not in self.candidates:
                self.candidates[a] = [[], [], []]
            self.candidates[a][0].extend(pos_b)
            self.candidates[a][1].extend(implicit_b)
            self.candidates[a][2].extend(neg_b)

    def candidates_from_file(self, filename):
        def iter_from_file():
            with as_io(filename) as f:
                for line in f:
                    yield json.loads(line)

        self.candidates_from_iter(iter_from_file())

    def sample(self, a, b, pos):
        if self.candidates:
            pos = 0 if pos else 2
            c = random.choice(self.candidates[a][pos] + self.candidates[a][1])
            while c == b:
                c = random.choice(self.candidates[a][pos] + self.candidates[a][1])
        else:
            c = random.randint(0, self.objs_num - 1)
            while c == b:
                c = random.randint(0, self.objs_num - 1)
        return c

    def __call__(self, *obj_tuple, threshold=0.5):
        for _obj_tuple in obj_tuple:
            if len(_obj_tuple) == 3:
                a, b, rating = _obj_tuple
                if rating <= threshold:  # negative
                    c = self.sample(a, b, False)
                    yield a, c, b
                else:  # positive
                    c = self.sample(a, b, True)
                    yield a, b, c
            else:  # without rating, default to positive
                a, b = _obj_tuple
                c = self.sample(a, b, True)
                yield a, b, c


class Sampler(object):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class RandomSampler(Sampler):
    def __init__(self, objs_num):
        self.objs_num = objs_num

    def __call__(self, *obj_tuple):
        raise
