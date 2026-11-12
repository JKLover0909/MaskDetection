# CLAUDE.md

## Overview
Face-mask and PPE (personal protective equipment) detection with Ultralytics YOLO: a YOLO-format face-mask dataset (`with_mask`, `without_mask`, `mask_weared_incorrect`), real-time webcam detection, and image/video PPE detection scripts.

## Layout
- `mask_detection_realtime.py` — real-time mask detection (webcam).
- `ppe_detect_image.py`, `ppe_detect_video.py` — PPE detection on a single image / a video file (argparse CLI).
- `code.ipynb` — experiments: dataset conversion, YOLO training, Hugging Face / ultralyticsplus PPE model trials.
- `yolo_dataset/` — `images/{train,val}` (682 / 171 images), `labels/{train,val}`, `dataset.yaml` (`nc: 3`; its `path:` is absolute on the original server, `/home/jkl0909/MaskDetection/yolo_dataset` — change it before training elsewhere).
- `yolov8n.pt`, `yolo11n.pt` — pretrained Ultralytics weights.
- `input.mp4`, `out.mp4`, `*.png` — sample input/output media.
- `Face-Mask-Detection-Model-CNN` — gitlink (mode 160000) to a separate CNN project; no `.gitmodules`, so it is empty after a plain clone.

## Commands
- Install: `pip install ultralytics opencv-python torch ultralyticsplus transformers`.
- Train: `yolo detect train data=yolo_dataset/dataset.yaml model=yolo11n.pt epochs=50 imgsz=640`.
- Run: `python mask_detection_realtime.py`; `python ppe_detect_image.py --help`; `python ppe_detect_video.py --help`.

## Rules
- Do not commit training output (`runs/`), logs or secrets (gitignored).
- Keep dataset image/label filenames paired (`maksssksksssN.png` ↔ `.txt`); do not rename files in `yolo_dataset/`.
- Do not "fix" the `Face-Mask-Detection-Model-CNN` gitlink without being asked.
- Repo is ~440MB (dataset + media); cloning is slow — prefer a shallow or partial clone.
