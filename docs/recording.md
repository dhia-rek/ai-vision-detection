# Recording ASL hand landmarks

Choose **Sign language → Record**, choose the letter being shown, then start recording. The recorder works with webcam, video, and image inputs. It captures at most one sample every 125 ms and skips near-identical consecutive hand poses. Stop recording at any time; use **Download JSON** to export the stored samples or **Clear** to remove them. The `R` key toggles recording while Record mode is selected.

Frames with zero or multiple detected hands are skipped; keep only the intended hand in view. Only the 21 hand landmarks, handedness, selected letter, and source image dimensions are stored. Images and video frames are never stored by the recorder. Samples remain in this browser's local storage under `avd.recordings.v1` until downloaded or cleared; clearing browser storage also removes them. The recorder stops at 5,000 samples (the browser's storage limit); download and clear to continue.

Hand geometry can in principle identify someone. Share recordings only with the person's consent, and never include other personal data in a recording or its accompanying files. These downloads have no verified signer IDs; keep contributors' files separate and do not treat a split of them as a signer-independent evaluation.

## Convert recordings to training data

Download the JSON file, then convert it to the same `.npz` layout used by the other extractors:

```sh
uv run python scripts/asl_extract.py recorded --root file.json --out /tmp/asl-lm/rec_0.npz
uv run --with scikit-learn python scripts/train_asl.py fit ...
```

`--root` can also be a folder containing downloaded recording JSON files. The training workflow will automatically pick up `rec_*.npz` files; the training script does so. Keep recordings for each contributor or session in separate downloaded files to retain group-aware split boundaries.

## Licence

Contributions to this project, including code and documentation here, are made under the project licence in [`LICENSE`](../LICENSE).
