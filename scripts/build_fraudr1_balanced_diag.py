"""Build fraudr1_balanced_diag.jsonl (600 rows) from E1 480 + romance 120.

Composition target (guide section 4.3): 300 unsafe / 300 safe, 300 zh / 300 en,
5 categories x ~120 (Phishing, Impersonation, Fraudulent Services, Fake Job
Postings, Online Relationships).
"""
from __future__ import annotations

import collections
import json
import os

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", "exp2_prior_work_comparison", "fraudr1", "diagnostic"))
WORK = os.path.join(BASE, "_romance_work")  # romance annotation/synthesis intermediates


def load(path, base=BASE):
    rows = []
    with open(os.path.join(base, path), encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def lang_of(row):
    i = row["id"]
    return "Chinese" if ("zh_" in i or "_zh_" in i) else "English"


def main():
    e1 = load("balanced_e1_selected.jsonl")
    romance_real_ann = {r["id"]: r for r in load("romance_audit_adjudicated.jsonl", WORK)}
    romance_syn_ann = {r["id"]: r for r in load("romance_audit_adjudicated_synth.jsonl", WORK)}
    romance_syn2_ann = {r["id"]: r for r in load("romance_audit_adjudicated_2.jsonl", WORK)}
    inp1 = {r["id"]: r for r in load("romance_annotation_input.jsonl", WORK)}
    inp1.update({r["id"]: r for r in load("romance_synthetic_input.jsonl", WORK)})
    inp2 = {r["id"]: r for r in load("romance_annotation_input2.jsonl", WORK)}
    synth1 = {r["id"]: r for r in load("romance_synthetic_generated.jsonl", WORK)}
    synth2 = {r["id"]: r for r in load("romance_synthetic_generated2.jsonl", WORK)}

    out = []

    # ---- E1: 480 rows, keep as-is with normalized fields
    for r in e1:
        out.append({
            "id": r["id"],
            "group_id": r["group_id"],
            "benchmark": "fraudr1_diag",
            "category": r["category"],
            "language": r["language"],
            "query": r["query"],
            "answer": r["answer"],
            "gold_binary": int(r["gold_binary"]),
            "gold_type": "unsafe" if int(r["gold_binary"]) == 1 else "safe",
            "gold_source": "e1_v32_balanced",
            "expected_behavior": None,
            "sub_category": None,
            "target_model": None,
            "metadata": {
                "provenance": "e1_v32_panel",
                "trust_flag": bool(r.get("trust_flag")),
                "source_case": r.get("id"),
            },
        })

    # ---- Romance rows: gather annotated (query, answer, binary, four_type)
    romance_pool = []
    for ann, inp, src, prov in [
        (romance_real_ann, inp1, None, "romance_real_audit"),
        (romance_syn_ann, inp1, synth1, "romance_synth_audit"),
        (romance_syn2_ann, inp2, synth2, "romance_synth2_audit"),
    ]:
        for sid, a in ann.items():
            base_row = inp.get(sid)
            if base_row is None:
                continue
            binary = int(a.get("binary", 0))
            four_type = a.get("four_type", "safe")
            romance_pool.append({
                "id": sid,
                "group_id": base_row.get("group_id") or sid.rsplit("_", 1)[0],
                "category": "Online Relationships",
                "language": base_row.get("language") or lang_of(base_row),
                "query": base_row["query"],
                "answer": base_row["answer"],
                "gold_binary": binary,
                "four_type": four_type,
                "provenance": prov,
            })

    pos = [r for r in romance_pool if r["gold_binary"] == 1]
    neg = [r for r in romance_pool if r["gold_binary"] == 0]
    print(f"romance pool: pos={len(pos)} neg={len(neg)}")
    print("pos lang:", collections.Counter(r["language"] for r in pos))
    print("neg lang:", collections.Counter(r["language"] for r in neg))

    # select 30 zh + 30 en positives (real first), 30 zh + 30 en negatives (real first)
    def select(rows, lang, n, real_first=True):
        real = [r for r in rows if r["language"] == lang and r["provenance"] == "romance_real_audit"]
        syn = [r for r in rows if r["language"] == lang and r["provenance"] != "romance_real_audit"]
        picked = (real + syn)[:n]
        return picked

    pos_sel = select(pos, "Chinese", 30) + select(pos, "English", 30)
    neg_sel = select(neg, "Chinese", 30) + select(neg, "English", 30)
    print("pos selected:", collections.Counter(r["language"] for r in pos_sel),
          "neg selected:", collections.Counter(r["language"] for r in neg_sel))

    for r in pos_sel + neg_sel:
        out.append({
            "id": r["id"],
            "group_id": r["group_id"],
            "benchmark": "fraudr1_diag",
            "category": "Online Relationships",
            "language": r["language"],
            "query": r["query"],
            "answer": r["answer"],
            "gold_binary": r["gold_binary"],
            "gold_type": "unsafe" if r["gold_binary"] == 1 else "safe",
            "gold_source": r["provenance"],
            "expected_behavior": None,
            "sub_category": r.get("four_type"),
            "target_model": None,
            "metadata": {
                "provenance": r["provenance"],
                "trust_flag": r["gold_binary"] == 1 and r["provenance"] != "romance_real_audit",
                "source_case": r["id"],
                "four_type": r.get("four_type"),
            },
        })

    # ---- validation
    ids = [r["id"] for r in out]
    assert len(ids) == len(set(ids)), "duplicate ids"
    assert len(out) == 600, len(out)
    g = collections.Counter(r["gold_binary"] for r in out)
    la = collections.Counter(r["language"] for r in out)
    ca = collections.Counter(r["category"] for r in out)
    print("total:", len(out), "gold:", dict(g), "lang:", dict(la), "cat:", dict(ca))
    empty = [r["id"] for r in out if not r["query"] or not r["answer"]]
    assert not empty, empty[:5]

    with open(os.path.join(BASE, "fraudr1_balanced_diag.jsonl"), "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "name": "fraudr1_balanced_diag",
        "n": len(out),
        "gold_binary": dict(g),
        "language": dict(la),
        "category": dict(ca),
        "trust_facilitation_positives": sum(1 for r in out if r["gold_binary"] == 1 and r["metadata"].get("trust_flag")),
        "gold_source": dict(collections.Counter(r["gold_source"] for r in out)),
        "four_type_pos": dict(collections.Counter(r.get("sub_category") for r in out if r["gold_binary"] == 1)),
    }
    with open(os.path.join(BASE, "fraudr1_balanced_diag_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("manifest:", json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
