# E1 TRIAD 统计附录

{
  "power": {
    "anchor_groups": 160,
    "target_delta": 0.1,
    "estimated_power": 0.99,
    "passed": true
  },
  "thresholds": {
    "modes": {
      "q-only": 0.5,
      "y-only": 0.51,
      "wrong-q+y": 0.56,
      "q+y": 0.54
    },
    "source": "calibration",
    "immutable_after": "2026-08-01T21:15:50"
  },
  "split": {
    "split_counts": {
      "model_dev": 120,
      "calibration": 80,
      "anchor": 320,
      "reserve": 160
    },
    "split_disjoint": {
      "passed": true,
      "conflicts": []
    },
    "seed": 20260801
  }
}
