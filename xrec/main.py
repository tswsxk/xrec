# coding: utf-8
# 2020/6/14 @ tongshiwei


from sklearn.metrics import ndcg_score

rel = [[10, 0, 0, 5, 0, 0, 0, 0]]
score = [[1, 0, 0, 1, 0, 0, 0, 0]]

print(ndcg_score(rel, score))

rel = [[10, 0, 0, 5, 0]]
score = [[1, 0, 0, 1, 0]]

print(ndcg_score(rel, score))

rel = [[10, 5, 0, 0, 0]]
score = [[1, 1, 0, 0, 0]]

print(ndcg_score(rel, score, ignore_ties=True))

rel = [[10, 5, 0, 0, 0]]
score = [[1, 1, 0, 0, 0]]

print(ndcg_score(rel, score))

rel = [[10, 5, 0, 0, 0] + [0] * 5]
score = [[1, 1, 0, 0, 0] + [0] * 5]

print(ndcg_score(rel, score, k=5))
print(ndcg_score(rel, score, k=10))
