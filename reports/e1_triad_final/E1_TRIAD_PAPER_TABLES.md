# E1 TRIAD 论文表格草稿

## E1-A
|Target model|Actor-valid N|A2/A3|Rate|Wilson CI|Events/1k|
|---|---|---|---|---|---|
|deepseek|1540|15|0.0097|[0.0059, 0.0160]|9.74|
|qwen|1540|11|0.0071|[0.0040, 0.0127]|7.14|

## E1-B Anchor
|Mode|N|Macro-F1|Precision|Recall|FPR|AUPRC|Brier|ECE|
|---|---|---|---|---|---|---|---|---|
|q-only|320|0.3333|0.5000|1.0000|1.0000|0.5000|0.2500|0.0000|
|y-only|320|0.6275|0.6185|0.6687|0.4125|0.6576|0.2319|0.0486|
|wrong-q+y|320|0.5559|0.6709|0.3312|0.1625|0.6446|0.2339|0.0425|
|q+y|320|0.5936|0.6486|0.4500|0.2437|0.6416|0.2329|0.0577|

## E1-B Context
|Mode|N|Macro-F1|Precision|Recall|FPR|AUPRC|Brier|ECE|
|---|---|---|---|---|---|---|---|---|
|q-only|160|0.3333|0.5000|1.0000|1.0000|0.5000|0.2500|0.0000|
|y-only|160|0.5808|0.5765|0.6125|0.4500|0.6213|0.2382|0.0172|
|wrong-q+y|160|0.5317|0.6571|0.2875|0.1500|0.6317|0.2383|0.0385|
|q+y|160|0.5389|0.5918|0.3625|0.2500|0.6133|0.2388|0.0242|

## E1-C1
|Mode|N|Macro-F1|Precision|Recall|FPR|AUPRC|Brier|ECE|
|---|---|---|---|---|---|---|---|---|
|q-only|240|0.3333|0.5000|1.0000|1.0000|0.5000|0.2500|0.0000|
|y-only|240|0.6916|0.6885|0.7000|0.3167|0.7476|0.2210|0.1059|
|wrong-q+y|240|0.6181|0.7222|0.4333|0.1667|0.7111|0.2243|0.0834|
|q+y|240|0.6560|0.7241|0.5250|0.2000|0.7240|0.2222|0.0803|

## E1-C2
|Mode|N|Macro-F1|Precision|Recall|FPR|AUPRC|Brier|ECE|
|---|---|---|---|---|---|---|---|---|
|y-only|2000|0.1344|0.0065|0.4583|0.8522|0.0155|0.2992|0.5327|
|q+y|2000|0.4226|0.0118|0.2917|0.2955|0.0394|0.2698|0.5062|

## 原始机器表
```json
{
  "A": {
    "source": "V10 cache-first natural behavior",
    "by_model": {
      "deepseek": {
        "n": 1540,
        "a2_a3": 15,
        "rate": 0.00974025974025974,
        "wilson_ci": {
          "low": 0.005911560103550392,
          "high": 0.01600873399617203
        },
        "events_per_1k": 9.74025974025974
      },
      "qwen": {
        "n": 1540,
        "a2_a3": 11,
        "rate": 0.007142857142857143,
        "wilson_ci": {
          "low": 0.003993115996839215,
          "high": 0.012745298866325448
        },
        "events_per_1k": 7.142857142857143
      }
    },
    "gate": "PASS"
  },
  "B": {
    "summary": {
      "gate": "STOP_NO_CONTEXT_GAIN",
      "qy_macro_f1": 0.5935959359593596,
      "qy_ci": {
        "point": 0.5935959359593596,
        "low": 0.5451451451451452,
        "high": 0.6415507687427513
      },
      "qy_minus_y": -0.03391431044339499,
      "qy_minus_wrong": 0.03767402525494912,
      "qonly_accuracy": 0.5,
      "anchor_rows": 320,
      "anchor_groups": 160
    },
    "by_mode": [
      {
        "mode": "q-only",
        "n": 320,
        "tp": 160,
        "fp": 160,
        "tn": 0,
        "fn": 0,
        "macro_f1": 0.3333333333333333,
        "balanced_accuracy": 0.5,
        "accuracy": 0.5,
        "precision": 0.5,
        "recall": 1.0,
        "fpr": 1.0,
        "auroc": 0.5,
        "auprc": 0.5,
        "brier": 0.25,
        "ece": 0.0
      },
      {
        "mode": "y-only",
        "n": 320,
        "tp": 107,
        "fp": 66,
        "tn": 94,
        "fn": 53,
        "macro_f1": 0.6275102464027545,
        "balanced_accuracy": 0.628125,
        "accuracy": 0.628125,
        "precision": 0.6184971098265896,
        "recall": 0.66875,
        "fpr": 0.4125,
        "auroc": 0.6752734375,
        "auprc": 0.6576378032761672,
        "brier": 0.23185642086292108,
        "ece": 0.048553969425123675
      },
      {
        "mode": "wrong-q+y",
        "n": 320,
        "tp": 53,
        "fp": 26,
        "tn": 134,
        "fn": 107,
        "macro_f1": 0.5559219107044104,
        "balanced_accuracy": 0.584375,
        "accuracy": 0.584375,
        "precision": 0.6708860759493671,
        "recall": 0.33125,
        "fpr": 0.1625,
        "auroc": 0.6548828125,
        "auprc": 0.6445571111693195,
        "brier": 0.233852186589742,
        "ece": 0.04249144243332483
      },
      {
        "mode": "q+y",
        "n": 320,
        "tp": 72,
        "fp": 39,
        "tn": 121,
        "fn": 88,
        "macro_f1": 0.5935959359593596,
        "balanced_accuracy": 0.603125,
        "accuracy": 0.603125,
        "precision": 0.6486486486486487,
        "recall": 0.45,
        "fpr": 0.24375,
        "auroc": 0.66671875,
        "auprc": 0.6415793051989618,
        "brier": 0.2329098433301468,
        "ece": 0.057686290042748826
      }
    ],
    "context_by_mode": [
      {
        "mode": "q-only",
        "n": 160,
        "tp": 80,
        "fp": 80,
        "tn": 0,
        "fn": 0,
        "macro_f1": 0.3333333333333333,
        "balanced_accuracy": 0.5,
        "accuracy": 0.5,
        "precision": 0.5,
        "recall": 1.0,
        "fpr": 1.0,
        "auroc": 0.5,
        "auprc": 0.5,
        "brier": 0.25,
        "ece": 0.0
      },
      {
        "mode": "y-only",
        "n": 160,
        "tp": 49,
        "fp": 36,
        "tn": 44,
        "fn": 31,
        "macro_f1": 0.5808406647116324,
        "balanced_accuracy": 0.58125,
        "accuracy": 0.58125,
        "precision": 0.5764705882352941,
        "recall": 0.6125,
        "fpr": 0.45,
        "auroc": 0.62984375,
        "auprc": 0.6213482402400508,
        "brier": 0.2381576163145827,
        "ece": 0.017197411394559454
      },
      {
        "mode": "wrong-q+y",
        "n": 160,
        "tp": 23,
        "fp": 12,
        "tn": 68,
        "fn": 57,
        "macro_f1": 0.5317073170731708,
        "balanced_accuracy": 0.56875,
        "accuracy": 0.56875,
        "precision": 0.6571428571428571,
        "recall": 0.2875,
        "fpr": 0.15,
        "auroc": 0.6153125,
        "auprc": 0.6317499966834013,
        "brier": 0.23829241430639203,
        "ece": 0.03847938629962702
      },
      {
        "mode": "q+y",
        "n": 160,
        "tp": 29,
        "fp": 20,
        "tn": 60,
        "fn": 51,
        "macro_f1": 0.5389423272048378,
        "balanced_accuracy": 0.55625,
        "accuracy": 0.55625,
        "precision": 0.5918367346938775,
        "recall": 0.3625,
        "fpr": 0.25,
        "auroc": 0.620625,
        "auprc": 0.6133342425504401,
        "brier": 0.23883317482086058,
        "ece": 0.02424758950022763
      }
    ],
    "deltas": [
      {
        "point": -0.03391431044339499,
        "low": -0.07695495483016002,
        "high": 0.008566015289882989,
        "comparison": "qy_minus_y"
      },
      {
        "point": 0.03767402525494912,
        "low": 0.0003433965140077211,
        "high": 0.0753361452057213,
        "comparison": "qy_minus_wrong"
      },
      {
        "point": 0.26026260262602624,
        "low": 0.21053589300336634,
        "high": 0.3092816401419968,
        "comparison": "qy_minus_q"
      }
    ],
    "mcnemar_holm": [
      {
        "left": "q+y",
        "right": "y-only",
        "b": 27,
        "c": 35,
        "p_exact": 0.3741517124047949,
        "holm_p": 0.7483034248095898
      },
      {
        "left": "q+y",
        "right": "wrong-q+y",
        "b": 24,
        "c": 18,
        "p_exact": 0.44079906734259566,
        "holm_p": 0.7483034248095898
      },
      {
        "left": "q+y",
        "right": "q-only",
        "b": 121,
        "c": 88,
        "p_exact": 0.026626261914464214,
        "holm_p": 0.07987878574339265
      }
    ]
  },
  "C": {
    "summary": {
      "gate": "DIRECTIONAL_PASS",
      "c1_n": 240,
      "c2_n": 2000,
      "c2_positive": 24,
      "c2_prevalence": 0.012,
      "c2_qy_auprc": 0.039433325610817985,
      "c2_y_auprc": 0.015537595654304622,
      "c2_qy_fpr": 0.29554655870445345,
      "c2_y_fpr": 0.8522267206477733
    },
    "c1_by_mode": [
      {
        "mode": "q-only",
        "n": 240,
        "tp": 120,
        "fp": 120,
        "tn": 0,
        "fn": 0,
        "macro_f1": 0.3333333333333333,
        "balanced_accuracy": 0.5,
        "accuracy": 0.5,
        "precision": 0.5,
        "recall": 1.0,
        "fpr": 1.0,
        "auroc": 0.5,
        "auprc": 0.5,
        "brier": 0.25,
        "ece": 0.0
      },
      {
        "mode": "y-only",
        "n": 240,
        "tp": 84,
        "fp": 38,
        "tn": 82,
        "fn": 36,
        "macro_f1": 0.6916452531425793,
        "balanced_accuracy": 0.6916666666666667,
        "accuracy": 0.6916666666666667,
        "precision": 0.6885245901639344,
        "recall": 0.7,
        "fpr": 0.31666666666666665,
        "auroc": 0.7379861111111111,
        "auprc": 0.7475733768268072,
        "brier": 0.2209982109604586,
        "ece": 0.10587621827165757
      },
      {
        "mode": "wrong-q+y",
        "n": 240,
        "tp": 52,
        "fp": 20,
        "tn": 100,
        "fn": 68,
        "macro_f1": 0.6180555555555556,
        "balanced_accuracy": 0.6333333333333333,
        "accuracy": 0.6333333333333333,
        "precision": 0.7222222222222222,
        "recall": 0.43333333333333335,
        "fpr": 0.16666666666666666,
        "auroc": 0.705625,
        "auprc": 0.7111476761652853,
        "brier": 0.22434745196778214,
        "ece": 0.08343517622029108
      },
      {
        "mode": "q+y",
        "n": 240,
        "tp": 63,
        "fp": 24,
        "tn": 96,
        "fn": 57,
        "macro_f1": 0.6559961777353082,
        "balanced_accuracy": 0.6625000000000001,
        "accuracy": 0.6625,
        "precision": 0.7241379310344828,
        "recall": 0.525,
        "fpr": 0.2,
        "auroc": 0.7252083333333333,
        "auprc": 0.7239873311771761,
        "brier": 0.22222927190959743,
        "ece": 0.08031460752051628
      }
    ],
    "c2_by_mode": [
      {
        "mode": "y-only",
        "n": 2000,
        "tp": 11,
        "fp": 1684,
        "tn": 292,
        "fn": 13,
        "macro_f1": 0.13441309816097213,
        "balanced_accuracy": 0.30305330634278005,
        "accuracy": 0.1515,
        "precision": 0.006489675516224189,
        "recall": 0.4583333333333333,
        "fpr": 0.8522267206477733,
        "auroc": 0.2629681174089069,
        "auprc": 0.015537595654304622,
        "brier": 0.29920918858147844,
        "ece": 0.5327184619183728
      },
      {
        "mode": "q+y",
        "n": 2000,
        "tp": 7,
        "fp": 584,
        "tn": 1392,
        "fn": 17,
        "macro_f1": 0.4226081108669285,
        "balanced_accuracy": 0.49806005398110664,
        "accuracy": 0.6995,
        "precision": 0.011844331641285956,
        "recall": 0.2916666666666667,
        "fpr": 0.29554655870445345,
        "auroc": 0.35855263157894735,
        "auprc": 0.039433325610817985,
        "brier": 0.2698128043623674,
        "ece": 0.5062221177126219
      }
    ],
    "c2_prevalence": {
      "n": 2000,
      "positive": 24,
      "rate": 0.012,
      "wilson_low": 0.008077154276022635,
      "wilson_high": 0.017793883870480226,
      "rule_of_three_upper": null
    }
  },
  "Gold": {
    "note": "Gold v2 validator applied to official PKU safe/unsafe labels as source-derived proxy; no new manual labels.",
    "expected": 680,
    "valid": 680,
    "valid_schema": 1.0,
    "paired_coverage": 1.0,
    "binary_agreement_proxy": 1.0,
    "positive_agreement_proxy": 1.0,
    "uncertain_after_adjudication": 0.0,
    "material_invariant": 1.0,
    "passed": true
  }
}
```
