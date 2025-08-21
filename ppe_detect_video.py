import argparse
import time
import cv2
import numpy as np
import torch

# Monkey patch torch.load to fix the weights_only issue
original_torch_load = torch.load

def patched_torch_load(f, map_location=None, pickle_module=None, **pickle_load_args):
    # Force weights_only=False to allow loading the model
    if 'weights_only' not in pickle_load_args:
        pickle_load_args['weights_only'] = False
    return original_torch_load(f, map_location=map_location, pickle_module=pickle_module, **pickle_load_args)

# Replace torch.load with our patched version
torch.load = patched_torch_load

from ultralyticsplus import YOLO

parser = argparse.ArgumentParser(description="PPE detection (glove/mask/...) on a video using YOLOv8 from Hugging Face")
parser.add_argument("--input", type=str, required=True, help="Path to input video")
parser.add_argument("--output", type=str, required=True, help="Path to output video")
parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
parser.add_argument("--device", type=str, default=None, help="Device: e.g., 'cuda', 'cuda:0', or 'cpu'. Default: auto (CUDA if available)")
parser.add_argument("--no-half", action="store_true", help="Disable half-precision (FP16) on CUDA")
args = parser.parse_args()

# ---------------------------
# Device selection (prefer CUDA)
# ---------------------------
if args.device is not None:
    device = args.device
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

if device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARN] CUDA được chỉ định nhưng không khả dụng. Tự động chuyển sang CPU.")
    device = "cpu"

# ---------------------------
# Load model from Hugging Face
# ---------------------------
model_name = "keremberke/yolov8s-protective-equipment-detection"
print(f"Loading model: {model_name} on device: {device}")
model = YOLO(model_name)

# Set predict parameters
model.overrides["conf"] = args.conf
model.overrides["iou"]  = args.iou
model.overrides["device"] = device

# Bật FP16 khi chạy CUDA để tăng tốc (có thể tắt bằng --no-half)
# Lưu ý: FP16 chỉ áp dụng khi device là CUDA.
if device.startswith("cuda") and not args.no_half:
    model.overrides["half"] = True
else:
    model.overrides["half"] = False

# ---------------------------
# Color palette per class
# ---------------------------
PRESET_COLORS = {
    "glove": (0, 165, 255),
    "mask": (0, 255, 0),
    "helmet": (255, 0, 0),
    "goggles": (255, 255, 0),
    "shoes": (255, 0, 255),
    "no_glove": (0, 0, 255),
    "no_mask": (0, 0, 200),
    "no_helmet": (0, 50, 255),
    "no_goggles": (30, 30, 255),
    "no_shoes": (20, 20, 255)
}

def get_color(label: str):
    if label in PRESET_COLORS:
        return PRESET_COLORS[label]
    rng = np.random.default_rng(abs(hash(label)) % (2**32))
    return tuple(int(x) for x in rng.integers(low=50, high=255, size=3))

# ---------------------------
# Video IO setup
# ---------------------------
cap = cv2.VideoCapture(args.input)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {args.input}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0 or np.isnan(fps):
    fps = 25.0
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
if not out.isOpened():
    raise RuntimeError(f"Cannot open output video writer: {args.output}")

print(f"Input:  {args.input} ({width}x{height} @ {fps:.2f}fps)")
print(f"Output: {args.output}")
print(f"Conf: {args.conf} | IoU: {args.iou} | Device: {device} | FP16: {model.overrides['half']}")

# ---------------------------
# Inference loop
# ---------------------------
frame_idx = 0
t0_global = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        # Lưu ý: device/half đã set qua model.overrides, nên chỉ cần gọi predict
        results = model.predict(frame, verbose=False)
        infer_time = (time.time() - t0)

        if len(results) > 0:
            r = results[0]
            for box in r.boxes:
                cls_id = int(box.cls.item())
                label = r.names.get(cls_id, str(cls_id))
                conf  = float(box.conf.item()) if hasattr(box, "conf") and box.conf is not None else 0.0
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                color = get_color(label)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                caption = f"{label} {conf:.2f}"
                (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                y_text = max(y1, th + 5)
                cv2.rectangle(frame, (x1, y_text - th - 4), (x1 + tw + 4, y_text + baseline), color, -1)
                cv2.putText(frame, caption, (x1 + 2, y_text - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        fps_inst = 1.0 / max(infer_time, 1e-6)
        cv2.putText(frame, f"FPS: {fps_inst:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 220, 50), 2, cv2.LINE_AA)

        out.write(frame)
        frame_idx += 1

        # # Nếu muốn xem realtime:
        # cv2.imshow("PPE Detection", frame)
        # if cv2.waitKey(1) & 0xFF == 27:  # ESC
        #     break

finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()

elapsed = time.time() - t0_global
print(f"Done. Processed {frame_idx} frames in {elapsed:.2f}s ({frame_idx/max(elapsed,1e-6):.2f} FPS avg).")