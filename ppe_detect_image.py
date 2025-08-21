import argparse
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

parser = argparse.ArgumentParser(description="PPE detection (glove/mask/...) on images using YOLOv8 from Hugging Face")
parser.add_argument("--input", type=str, required=True, help="Path to input image (supports jpg, png, bmp, tiff, webp)")
parser.add_argument("--output", type=str, required=True, help="Path to output image (supports jpg, png, bmp, tiff, webp)")
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
# Load and process image
# ---------------------------
print(f"Loading image: {args.input}")

# Load image with different flags to handle various formats including PNG with transparency
image = cv2.imread(args.input, cv2.IMREAD_COLOR)
if image is None:
    # Try loading with different flags if the first attempt fails
    image = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Cannot load image: {args.input}")
    
    # Convert RGBA to RGB if image has alpha channel (PNG with transparency)
    if len(image.shape) == 3 and image.shape[2] == 4:
        print("Image has alpha channel (RGBA), converting to RGB...")
        # Create white background
        background = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 255
        # Extract alpha channel
        alpha = image[:, :, 3] / 255.0
        # Blend image with white background
        for i in range(3):
            background[:, :, i] = background[:, :, i] * (1 - alpha) + image[:, :, i] * alpha
        image = background
    elif len(image.shape) == 2:
        # Convert grayscale to BGR
        print("Converting grayscale image to BGR...")
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

print(f"Image size: {image.shape[1]}x{image.shape[0]}")
print(f"Conf: {args.conf} | IoU: {args.iou} | Device: {device} | FP16: {model.overrides['half']}")

# ---------------------------
# Inference
# ---------------------------
print("Running inference...")
results = model.predict(image, verbose=False)

if len(results) > 0:
    r = results[0]
    detection_count = 0
    
    for box in r.boxes:
        cls_id = int(box.cls.item())
        label = r.names.get(cls_id, str(cls_id))
        conf  = float(box.conf.item()) if hasattr(box, "conf") and box.conf is not None else 0.0
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        color = get_color(label)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        caption = f"{label} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_text = max(y1, th + 5)
        cv2.rectangle(image, (x1, y_text - th - 4), (x1 + tw + 4, y_text + baseline), color, -1)
        cv2.putText(image, caption, (x1 + 2, y_text - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        
        detection_count += 1
        print(f"Detected: {label} (confidence: {conf:.3f}) at [{x1}, {y1}, {x2}, {y2}]")

    print(f"Total detections: {detection_count}")
else:
    print("No detections found")

# ---------------------------
# Save output image
# ---------------------------
# Determine output format based on file extension
output_ext = args.output.lower().split('.')[-1]
if output_ext in ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp']:
    # Set compression parameters for different formats
    if output_ext == 'png':
        # PNG compression level (0-9, where 9 is maximum compression)
        cv2.imwrite(args.output, image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    elif output_ext in ['jpg', 'jpeg']:
        # JPEG quality (0-100, where 100 is maximum quality)
        cv2.imwrite(args.output, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        cv2.imwrite(args.output, image)
else:
    # Default to PNG if extension is not recognized
    print(f"Unrecognized output format '{output_ext}', saving as PNG...")
    args.output = args.output.rsplit('.', 1)[0] + '.png'
    cv2.imwrite(args.output, image, [cv2.IMWRITE_PNG_COMPRESSION, 6])

print(f"Output saved to: {args.output}")

# ---------------------------
# Display image (optional)
# ---------------------------
print("Press any key to close the image window...")
cv2.imshow("PPE Detection Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Done!")
