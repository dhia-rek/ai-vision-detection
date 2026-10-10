# Sign-language test fixtures

Hand-only photos of ASL fingerspelling letters, taken from the Hugging Face dataset
[EitanG98/asl_letters](https://huggingface.co/datasets/EitanG98/asl_letters) (licence: **Unlicense**, public domain).
Resized to max 480 px side, JPEG quality 85. No faces are visible.
Each is correctly classified (prob > 0.95) by `backend/app/models/asl_letters.npz`.

| File | Source file in dataset |
| --- | --- |
| letter_A.jpg | https://huggingface.co/datasets/EitanG98/asl_letters/blob/main/train_val/A/A28_jpg.rf.dbe1a8df3c9bbaf02e2cebda0cbb46c0.jpg |
| letter_B.jpg | https://huggingface.co/datasets/EitanG98/asl_letters/blob/main/train_val/B/B11_jpg.rf.690f5a7816485b5178fe69aaaaac2d7b.jpg |
| letter_C.jpg | https://huggingface.co/datasets/EitanG98/asl_letters/blob/main/train_val/C/C6_jpg.rf.6fd6534e3d31450c18d3f20c6e9fc623.jpg |
| letter_L.jpg | https://huggingface.co/datasets/EitanG98/asl_letters/blob/main/train_val/L/L25_jpg.rf.e365d617303e4f5f5e95f6df8af610a9.jpg |
| letter_V.jpg | https://huggingface.co/datasets/EitanG98/asl_letters/blob/main/train_val/V/V18_jpg.rf.ed994ef915feeef4bee12c9bd1056485.jpg |
| letter_Y.jpg | https://huggingface.co/datasets/EitanG98/asl_letters/blob/main/train_val/Y/Y14_jpg.rf.ddf87faa58cff7eb0ac1f69d481e9aef.jpg |

## Gesture photos

From Wikimedia Commons. Hands only, no identifiable faces.

| File | Expected (MediaPipe gesture) | Source | Author | Licence |
| --- | --- | --- | --- | --- |
| gesture_thumbs_up.jpg | Thumb_Up | [Hitchhiker's gesture](https://commons.wikimedia.org/wiki/File:Hitchhiker%27s_gesture.jpg) | User:Drozd | Public domain |
| gesture_thumbs_up_2.jpg | Thumb_Up | [Thumbs up (5762088099)](https://commons.wikimedia.org/wiki/File:Thumbs_up_(5762088099).jpg) | Morpheus Lander | Public domain |
| gesture_i_love_you.jpg | ILoveYou | [I love you in Sign Language or the number 19 in Finger Binary](https://commons.wikimedia.org/wiki/File:I_love_you_in_Sign_Language_or_the_number_19_in_Finger_Binary.jpg) | Wikimedia user | CC0 |
