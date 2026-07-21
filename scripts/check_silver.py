import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from src.data_utils import Taxonomy, load_doc2labels
from src.evaluate import example_f1

d = Path("reference_data/Amazon-531")
taxo = Taxonomy.load(d/"labels.txt", d/"label_hierarchy.txt")
cores = json.loads(Path("results/core_classes_train.json").read_text())
true = load_doc2labels(d/"train"/"doc2labels.txt")

# 1) mined core의 정밀도 (hit rate 71%보다 엄밀한 지표)
tp = sum(len(set(c) & set(t)) for c, t in zip(cores, true))
tot = sum(len(c) for c in cores)
print(f"core precision: {tp/tot:.4f}  (mined core 중 정답인 비율)")
print(f"avg cores/doc : {tot/len(cores):.2f}")

# 2) 천장: C+ = cores ∪ 조상 을 그대로 예측으로 썼을 때 (Eq.7과 동일 구성)
def ancestors(c):
    out, stack = set(), list(taxo.parents.get(c, ()))
    while stack:
        p = stack.pop()
        if p not in out:
            out.add(p); stack.extend(taxo.parents.get(p, ()))
    return out

preds = []
for c in cores:
    s = set(c)
    for x in c:
        s |= ancestors(x)
    preds.append(list(s))
print(f"silver ceiling (train Example-F1): {example_f1(true, preds):.4f}")