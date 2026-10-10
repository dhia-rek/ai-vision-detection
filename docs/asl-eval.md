# ASL letter classifier: evaluation

Generated with `scripts/train_asl.py`; full data provenance and protocol are in [asl-datasets.md](asl-datasets.md). The evaluation JSON is [asl-eval.json](asl-eval.json).
Checkpoint: `backend/app/models/asl_letters.npz`, feature version 3, SHA-256 `80261399df23d664c5f83d716cf3164f4a5ab9e7e8ac8b3fbfa02cc1068338e1`.

## Summary: before → after

| Test collection | Detected hands | Before (v1) | After (v3) | After, static only (no J/Z) | After, confidence ≥ 0.6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Roboflow 100 sign-language-sokdr, independent collection | 672/720 | 47.3% | **76.0%** (95% CI 72.7–79.1%) | 76.8% | 84.2% on 83% of detected hands |
| Cabana `Test/`, same collection as training | 683/720 | 81.4% | **94.9%** (95% CI 93.0–96.3%) | 94.9% | 96.9% on 94% of detected hands |

Accuracy excludes photos where MediaPipe found no hand. Counting the 48 Roboflow misses as incorrect gives 70.97% end-to-end accuracy (511/720). The before/after classifier comparison uses exactly the same 672 detected hands. The original v1 model's own random 80/20 sample split scored 96.5%, demonstrating that such a split was optimistic.

**Independence caveat:** the test photos are from a separate collection excluded from v3 training and the current leave-one-source-out comparison, but neither collection publishes signer IDs. Thus person-level disjointness cannot be proven. An earlier project iteration used these photos to choose recipes, so this is not a pristine never-seen benchmark for the project as a whole. Treat 76.0% as *collection-held-out for v3 training*, not a verified per-signer or blind generalization estimate. Cabana `Test/` may contain the same signers as Cabana `Train/`; its 94.9% is not independent.

## Model selection

Three CPU recipes were compared by leave-one-training-source-out evaluation (mean of three folds):

| Recipe | `asl_now` held out | `kp` held out | Cabana Train held out | Mean | Mean per-letter accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 71.1% | 88.3% | 87.0% | 82.1% | 84.3% |
| **MLP (selected)** | **76.2%** | **90.8%** | **92.3%** | **86.4%** | **88.4%** |
| Histogram gradient boosting | 71.4% | 89.8% | 89.0% | 83.4% | 85.2% |

The MLP has 287 input features (wrist-relative, scale/aspect- and in-plane-rotation-normalised coordinates, palm orientation, and pairwise distances), left/right mirroring and jitter/rotation augmentation. These are practical recipes with different compute budgets: logistic regression uses four fixed augmented copies at 500 samples/class, gradient boosting three at 400/class, and the MLP 40 fresh-augmentation epochs at 500/class. This compares deployable recipes, not isolated architecture quality. Full fold results: [asl-compare.json](asl-compare.json). The disjoint test collection was not part of this comparison.

## Held-out Roboflow collection

### After (v3) on the held-out Roboflow collection

n = 672, accuracy **76.0 %** (95 % CI 72.7-79.1), per-letter mean 76.9 %, static letters only (no J/Z, n=608) 76.8 %; with confidence >= 0.6: 84.2 % accurate on 83 % of hands

| Letter | A | B | C | D | E | F | G | H | I | J | K | L | M | N | O | P | Q | R | S | T | U | V | W | X | Y | Z |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Accuracy % | 59 | 100 | 94 | 100 | 82 | 100 | 48 | 8 | 13 | 62 | 88 | 100 | 84 | 88 | 83 | 76 | 95 | 84 | 97 | 25 | 68 | 93 | 100 | 88 | 88 | 77 |

Confusion matrix (rows = true letter, columns = predicted):

```
      A   B   C   D   E   F   G   H   I   J   K   L   M   N   O   P   Q   R   S   T   U   V   W   X   Y   Z  |   n
  A  17   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .  12   .   .   .   .   .   .   .  |  29
  B   .  23   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .  |  23
  C   .   1  15   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .  |  16
  D   .   .   .  24   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .  |  24
  E   .   .   .   1  18   .   .   .   .   .   .   .   .   .   1   .   .   .   2   .   .   .   .   .   .   .  |  22
  F   .   .   .   .   .  29   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .  |  29
  G   .   .   .   4   .   .  13   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   6   .   4  |  27
  H   .   .   .   .   .   .   .   2   .   .   1   .   .   .   .   .   .  13   .   .  10   .   .   .   .   .  |  26
  I   .   .   .   .   .   .   .   .   4  26   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .  |  30
  J   .   .   .   .   .   .   1   .   4  21   .   .   .   .   .   2   2   .   3   .   .   .   .   .   .   1  |  34
  K   .   .   .   .   .   .   .   .   .   .  23   .   .   .   .   .   .   .   .   .   .   3   .   .   .   .  |  26
  L   .   .   .   .   .   .   .   .   .   .   .  28   .   .   .   .   .   .   .   .   .   .   .   .   .   .  |  28
  M   3   .   .   .   .   .   .   .   .   .   .   .  21   .   .   .   .   .   1   .   .   .   .   .   .   .  |  25
  N   .   .   .   .   .   .   .   .   .   .   .   .   1  22   .   .   .   .   2   .   .   .   .   .   .   .  |  25
  O   .   .   .   .   3   .   .   .   .   .   .   .   .   .  19   .   .   .   1   .   .   .   .   .   .   .  |  23
  P   .   .   .   .   .   .   5   .   .   .   1   .   .   .   .  19   .   .   .   .   .   .   .   .   .   .  |  25
  Q   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   1  21   .   .   .   .   .   .   .   .   .  |  22
  R   .   .   1   .   .   .   .   2   .   .   .   .   .   .   1   .   .  21   .   .   .   .   .   .   .   .  |  25
  S   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .  29   1   .   .   .   .   .   .  |  30
  T   .   .   .   .   .   .   .   .   .   .   .   .   8   .   1   .   .   .   9   6   .   .   .   .   .   .  |  24
  U   .   .   .   .   .   .   .   2   .   .   .   .   .   .   .   .   .   6   .   .  17   .   .   .   .   .  |  25
  V   .   .   .   .   .   .   .   1   .   .   .   .   .   .   .   1   .   .   .   .   .  26   .   .   .   .  |  28
  W   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .  24   .   .   .  |  24
  X   .   .   .   2   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .  23   .   1  |  26
  Y   .   .   .   .   .   .   .   .   3   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .  23   .  |  26
  Z   .   .   .   .   .   .   .   .   .   .   .   .   .   .   1   5   1   .   .   .   .   .   .   .   .  23  |  30
```


### Before (v1) on `eitan`

n = 672, accuracy **47.3 %** (95 % CI 43.6-51.1), per-letter mean 46.7 %, static letters only (no J/Z, n=608) 45.7 %; with confidence >= 0.6: 49.2 % accurate on 92 % of hands

| Letter | A | B | C | D | E | F | G | H | I | J | K | L | M | N | O | P | Q | R | S | T | U | V | W | X | Y | Z |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Accuracy % | 59 | 91 | 50 | 12 | 45 | 76 | 19 | 8 | 20 | 32 | 85 | 96 | 92 | 20 | 35 | 4 | 9 | 20 | 50 | 0 | 28 | 82 | 100 | 0 | 85 | 97 |

Confusion matrix (rows = true letter, columns = predicted):

```
      A   B   C   D   E   F   G   H   I   J   K   L   M   N   O   P   Q   R   S   T   U   V   W   X   Y   Z  |   n
  A  17   2   .   .   .   .   .   .   .   2   .   .   1   .   .   .   .   .   6   1   .   .   .   .   .   .  |  29
  B   .  21   1   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   1   .   .   .   .   .   .   .  |  23
  C   .   2   8   3   .   2   .   .   .   .   .   1   .   .   .   .   .   .   .   .   .   .   .   .   .   .  |  16
  D   .   .   .   3   .   .   .   .   .   .   .   1   .   .   .   .   .   1   .   .   .   1   .   .   .  18  |  24
  E   .   .   .   .  10   .   .   .   .   1   .   .   7   .   .   .   .   .   2   .   .   .   .   .   .   2  |  22
  F   .   .   .   .   .  22   .   .   .   .   1   .   .   .   .   .   6   .   .   .   .   .   .   .   .   .  |  29
  G   .   .   .   .   .   .   5   .   .   .   1   .   .   .   .   .   .   .   .   .   .   .   .   .   .  21  |  27
  H   .   .   .   .   .   .   .   2   .   .   .   .  15   .   .   .   .   3   .   .   6   .   .   .   .   .  |  26
  I   .   .   .   .   .   .   .   .   6  18   .   .   2   .   .   .   .   .   .   .   .   .   .   .   1   3  |  30
  J   .   .   .   3   1   .   .   .   3  11   .   .   .   .   .   .   1   .   1   .   .   .   .   .   .  14  |  34
  K   .   .   .   .   .   .   .   .   .   .  22   .   .   .   .   .   .   .   .   .   .   4   .   .   .   .  |  26
  L   .   .   .   .   .   .   1   .   .   .   .  27   .   .   .   .   .   .   .   .   .   .   .   .   .   .  |  28
  M   .   .   .   .   .   .   .   .   .   1   .   .  23   .   1   .   .   .   .   .   .   .   .   .   .   .  |  25
  N   .   .   .   .   1   .   .   1   1   1   .   .  15   5   .   .   .   .   1   .   .   .   .   .   .   .  |  25
  O   .   1   .   3   4   .   .   1   1   .   .   .   3   .   8   .   1   .   .   .   .   .   .   .   1   .  |  23
  P   .   .   .   1   .   .   .   1   .   .   1   .   .   .   .   1   1   .   .   .   .   5   .   .   1  14  |  25
  Q   .   .   .   .   1   .   3   1   .   .   .   .   .   .   .   .   2   .   .   .   .   3   .   .   .  12  |  22
  R   .   1   .   .   .   1   .   2   .   .   .   .   1   .   .   .   .   5   .   .  11   3   .   .   .   1  |  25
  S   .   .   .   .   .   .   .   1   .   1   .   .   5   4   .   .   .   .  15   .   .   .   .   .   .   4  |  30
  T   .   1   .   .   .   .   .   .   .   1   .   .  18   .   .   .   .   .   4   .   .   .   .   .   .   .  |  24
  U   .   .   .   .   .   .   .   2   .   .   .   .   .   .   .   .   .   .   .   .   7  11   .   .   .   5  |  25
  V   .   .   .   .   .   .   .   3   .   .   .   .   .   .   .   .   1   .   .   .   .  23   .   .   .   1  |  28
  W   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .  24   .   .   .  |  24
  X   .   .   .   1   .   .  21   .   .   .   .   1   .   1   .   .   .   .   .   .   .   .   .   .   .   2  |  26
  Y   1   .   .   .   .   .   .   .   2   1   .   .   .   .   .   .   .   .   .   .   .   .   .   .  22   .  |  26
  Z   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   1   .   .   .   .   .   .   .   .  29  |  30
```



### After (v3) on Cabana `Test/` (same collection as training)

n = 683, accuracy **94.9 %** (95 % CI 93.0-96.3), per-letter mean 93.7 %, static letters only (no J/Z, n=683) 94.9 %; with confidence >= 0.6: 96.9 % accurate on 94 % of hands

| Letter | A | B | C | D | E | F | G | H | I | K | L | M | N | O | P | Q | R | S | T | U | V | W | X | Y |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Accuracy % | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 21 | 90 | 100 | 100 | 100 | 100 | 86 | 100 | 100 | 90 | 86 | 100 | 100 | 100 | 79 | 97 |

Confusion matrix (rows = true letter, columns = predicted):

```
      A   B   C   D   E   F   G   H   I   J   K   L   M   N   O   P   Q   R   S   T   U   V   W   X   Y   Z  |   n
  A  30   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .  |  30
  B   .  30   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .  |  30
  C   .   .  28   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .  |  28
  D   .   .   .  28   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .  |  28
  E   .   .   .   .  29   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .  |  29
  F   .   .   .   .   .  30   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .  |  30
  G   .   .   .   .   .   .  30   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .  |  30
  H   .   .   .   .   .   .   .  30   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .  |  30
  I   .   .   .   .   .   .   .   .   4  15   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .  |  19
  K   .   .   .   .   .   .   2   .   .   .  27   1   .   .   .   .   .   .   .   .   .   .   .   .   .   .  |  30
  L   .   .   .   .   .   .   .   .   .   .   .  30   .   .   .   .   .   .   .   .   .   .   .   .   .   .  |  30
  M   .   .   .   .   .   .   .   .   .   .   .   .  29   .   .   .   .   .   .   .   .   .   .   .   .   .  |  29
  N   .   .   .   .   .   .   .   .   .   .   .   .   .  30   .   .   .   .   .   .   .   .   .   .   .   .  |  30
  O   .   .   .   .   .   .   .   .   .   .   .   .   .   .  25   .   .   .   .   .   .   .   .   .   .   .  |  25
  P   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .  18   3   .   .   .   .   .   .   .   .   .  |  21
  Q   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .  27   .   .   .   .   .   .   .   .   .  |  27
  R   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .  30   .   .   .   .   .   .   .   .  |  30
  S   .   .   .   .   1   .   .   .   .   .   .   .   .   2   .   .   .   .  27   .   .   .   .   .   .   .  |  30
  T   1   .   .   .   .   .   .   .   .   .   .   .   1   2   .   .   .   .   .  24   .   .   .   .   .   .  |  28
  U   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .  30   .   .   .   .   .  |  30
  V   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .  30   .   .   .   .  |  30
  W   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .  30   .   .   .  |  30
  X   .   .   .   5   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .  23   .   1  |  29
  Y   1   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .  29   .  |  30
```

## Why the 80% target was not met

The measured 76.0% is below target; even the upper 95% Wilson bound is 79.1%. I/J share a static hand shape, and J/Z need a motion sequence, which the available labelled training data lacks. The remaining major confusions are fist-family thumb placement (A/S/T/M/N), H/U/R, and G/X; MediaPipe's single-frame landmarks are not always precise enough to separate those poses. The independent collection has 48/720 hand-detection misses, which the classifier cannot repair. More genuinely diverse, signer-labelled images and sequence data are required. The Record mode collects only landmarks for consented future training; it does not itself improve the shipped model.

## Reproduce

```bash
uv run --with scikit-learn python scripts/train_asl.py compare --lm /tmp/asl-lm --out docs/asl-compare.json
nice -n 15 env OMP_NUM_THREADS=2 uv run --with scikit-learn python scripts/train_asl.py fit --lm /tmp/asl-lm --model mlp --out backend/app/models/asl_letters.npz
uv run python scripts/train_asl.py eval --lm /tmp/asl-lm --model-file backend/app/models/asl_letters.npz --out /tmp/asl-eval-reproduced.md
```

`eval` generates the detailed sections and JSON; compare them with this contextual report. See [asl-datasets.md](asl-datasets.md) for the exact input sources and extraction commands.
