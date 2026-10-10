# ASL letter data: sources, licences, protocol

The letter classifier is trained and evaluated on **hand landmarks** only (21 points per hand, from
MediaPipe). Pixels are never stored in this repository. This page records where every sample comes
from, under which licence, how the data is split, and which candidates were rejected.

## Sources used

| Key | Dataset | Licence (primary source) | What it is | Role |
| --- | --- | --- | --- | --- |
| `asl_now` | [sid220/asl-now-fingerspelling](https://huggingface.co/datasets/sid220/asl-now-fingerspelling) ("ASLNow!") | MIT | 2,122 MediaPipe-Web landmark samples, A-Z, several participants, webcam | train |
| `kp` | [om192006/sign_language_keypoints](https://huggingface.co/datasets/om192006/sign_language_keypoints) (Om Pradip Chougule) | MIT | ~65k MediaPipe landmark CSVs (video frames), 25 letters (no Z); every 5th frame used (11,755) | train |
| `cab_train` | Cabana (2025), [*American Sign Language Alphabet Dataset*](https://data.mendeley.com/datasets/jdyksv2jhh/1), `Train/` | CC BY 4.0 | 177k 192x192 video frames, 24 static letters (no J, Z); every 8th frame run through MediaPipe (10,641 hands found, 3.4 % misses) | train |
| `cab_test` | same collection, `Test/` | CC BY 4.0 | 720 separate 200x200 photos, 30 per static letter (683 hands found) | **test, same collection** |
| `eitan` | [Roboflow 100 "sign-language-sokdr"](https://universe.roboflow.com/roboflow-100/sign-language-sokdr), created by David Lee ([original project](https://universe.roboflow.com/david-lee-d0rhs/american-sign-language-letters)); full-resolution originals via [EitanG98/asl_letters](https://huggingface.co/datasets/EitanG98/asl_letters) | CC BY 4.0 on Roboflow Universe and on the [LibreYOLO mirror](https://huggingface.co/datasets/LibreYOLO/sign-language-sokdr) (the EitanG98 mirror is tagged Unlicense; we follow the stricter original) | 720 phone photos of letters A-Z (incl. J, Z as poses), varied backgrounds; MediaPipe found a hand in 672 | **test, separate collection** |

Citations:

- Cabana, E. (2025). *American Sign Language Alphabet Dataset*. Mendeley Data, V1. doi:10.17632/jdyksv2jhh.1. Paper: Cabana, E. (2025). *Advancing Accessible AI: A Comprehensive Dataset and Neural Models for Real-Time American Sign Language Alphabet Classification.* IntelliSys 2025, LNNS 1567, Springer. doi:10.1007/978-3-032-00071-2_15.
- Lee, D. *American Sign Language Letters*, Roboflow Universe; Ciaglia et al., *Roboflow 100: A Rich, Multi-Domain Object Detection Benchmark*, arXiv:2211.13523.
- sid220, *ASLNow!* (asl-now.vercel.app); Chougule, O. P., *Sign Language MediaPipe Keypoints*.

Only derived landmark coordinates and aggregate numbers are kept. Nothing is redistributed. CC BY
attribution is satisfied by this page, the README and the credits.

## Protocol

* **Training sources:** `asl_now`, `kp`, `cab_train` (and any `rec_*.npz` made with the Record mode).
* **Model selection:** leave-one-source-out (train on two sources, test on the third), averaged over
  the three folds. The test sources are never touched during selection.
* **Final test:** `eitan` is a different collection (different people, camera, backgrounds) that is
  never used for training or selection. It is the signer-independent number. `cab_test` is reported
  too, but it comes from the same collection as `cab_train`, so signers may overlap: it is *not*
  signer-independent.
* **Caveat:** none of these datasets publishes signer IDs. "Signer-independent" therefore means
  "held-out collection, presumably different people", not a verified per-person split. Frames of one
  video are near-duplicates, so within-source random splits are optimistic; they are not used.
* The original (v1) model was evaluated on exactly the same hands, so before/after numbers are
  comparable.
* Hands are found with the same MediaPipe model the app uses. Images without a detected hand
  (about 5 %) are excluded from the accuracy and reported as such.
* Static letters are the 24 letters without J and Z (which are motion letters). J and Z are scored too
  (as single poses) in the headline number and separately excluded in the "static letters" column.

## Rejected or unusable candidates

| Candidate | Why not |
| --- | --- |
| Google "ASL Fingerspelling Recognition" / FSboard (Kaggle, CC BY 4.0, 147 Deaf signers, MediaPipe landmarks) | Licence fits and it is the best signer variety available, but it contains **continuous phrases**, not isolated letters: letter labels would need forced alignment. It is also a large Kaggle download that needs a login. Not used; a good follow-up. |
| Kaggle "ASL Alphabet" (grassknoted) | GPL-2 licence on the data: not compatible with an MIT project without legal review. Also only a few signers. |
| Sign Language MNIST (CC0) | 28x28 greyscale crops; MediaPipe cannot find hands reliably at that size. |
| HF mirrors `asthalochan/American_Sign_Language`, `suneetpaul/American_Sign_Language` (Apache-2.0 tag) | A tag on a re-upload with no card or provenance does not establish rights to the images. |
| HF `Marxulia/asl_sign_languages_alphabets_v03`, `atalaydenknalbant/asl-dataset`, `hnam25/asl-hand-gesture-images`, `dyllanesl/ASL_Dataset` | No licence stated. |
| `AxelS27/asl-now-fingerspelling` | Copy of `asl_now` (would leak into the test). |
| `fhswf/wholebody-pose-estimation-fingerspelling` (MIT) | German Sign Language fingerspelling, not ASL. |
| ASL Finger Spelling dataset, Pugeault & Bowden 2011 (Exeter) | No explicit licence on the page. |
| Mendeley ASL-Homework-RGBD, Chicago FSWild | CC BY-NC (non-commercial) or access-restricted. |
