"""Phase 5b: LLM budget (1,000 calls) for uncertain core-class decisions.

Selection: among (doc, candidate) pairs, take those whose conf(D,c) is
closest to the class median (the accept/reject boundary) - these are the
decisions where Eq. 2-3 is least reliable. One call judges one document
(all its boundary candidates at once), so 1,000 calls = 1,000 docs.

LLM verdict overrides the rule: accepted candidates are added to the doc's
core set, rejected ones removed. Output: results/core_classes_train_llm.json
"""
import argparse, json, os, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_utils import Taxonomy, load_corpus
from src.nli_topdown import SENTINEL

def rivals(taxo, c):
    ps = taxo.parents.get(c, ())
    if not ps: return taxo.roots - {c}
    out = set()
    for p in ps:
        out.add(p); out.update(taxo.children.get(p, ()))
    out.discard(c); return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--min_conf", type=float, default=0.15)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--dry_run", action="store_true", help="select docs, no API calls")
    args = ap.parse_args()

    d = Path(args.data_dir)
    taxo = Taxonomy.load(d/"labels.txt", d/"label_hierarchy.txt")
    S = np.load("cache/sim_train.npz")["scores"].astype(np.float32)
    cands = json.loads(Path("cache/cand_train.json").read_text())
    texts = load_corpus(d/"train"/"corpus.txt")
    cores = json.loads(Path("results/core_classes_train.json").read_text())
    n = len(cands)

    # recompute conf + medians (same as 03)
    conf, by_class = {}, defaultdict(list)
    for i in range(n):
        for c in cands[i]:
            riv = [S[i, r] for r in rivals(taxo, c) if S[i, r] != SENTINEL]
            v = float(S[i, c] - max(riv)) if riv else float(S[i, c])
            conf[(i, c)] = v; by_class[c].append(v)
    med = {c: float(np.median(v)) for c, v in by_class.items()}

    # uncertainty = distance to the decision boundary (max of the two gates)
    doc_unc = {}
    for i in range(n):
        ds = [abs(conf[(i, c)] - max(med[c], args.min_conf)) for c in cands[i]]
        if ds: doc_unc[i] = min(ds)
    chosen = sorted(doc_unc, key=doc_unc.get)[:args.budget]
    print(f"selected {len(chosen)} boundary docs "
          f"(uncertainty range {doc_unc[chosen[0]]:.4f}..{doc_unc[chosen[-1]]:.4f})")
    if args.dry_run: return

    from openai import OpenAI
    client = OpenAI()  # needs OPENAI_API_KEY env var
    changed, calls = 0, 0
    for i in chosen:
        cs = sorted(cands[i], key=lambda c: -conf[(i, c)])[:6]
        names = {c: taxo.names[c] for c in cs}
        prompt = (
            "You label Amazon product reviews with taxonomy classes.\n"
            f"Review: {texts[i][:1500]}\n\n"
            "Candidate classes:\n" +
            "\n".join(f"{c}: {nm}" for c, nm in names.items()) +
            "\n\nWhich candidates are CORE classes of this review (the most "
            "essential specific topics)? Reply ONLY with the numeric ids, "
            "comma-separated. Reply NONE if none apply."
        )
        try:
            r = client.chat.completions.create(
                model=args.model, max_tokens=30, temperature=0,
                messages=[{"role": "user", "content": prompt}])
            ans = r.choices[0].message.content.strip()
            calls += 1
        except Exception as e:
            print("API error, stopping:", e); break
        keep = set()
        if ans.upper() != "NONE":
            for t in ans.replace(",", " ").split():
                if t.isdigit() and int(t) in names: keep.add(int(t))
        new = sorted(keep)
        if new != sorted(cores[i]):
            cores[i] = new; changed += 1
        if calls % 100 == 0:
            print(f"  {calls}/{len(chosen)} calls, {changed} docs changed", flush=True)
            Path("results/core_classes_train_llm.json").write_text(json.dumps(cores))
    Path("results/core_classes_train_llm.json").write_text(json.dumps(cores))
    print(f"done: {calls} calls, {changed} docs modified -> results/core_classes_train_llm.json")

if __name__ == "__main__":
    main()