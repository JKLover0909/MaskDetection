# mask_detection_realtime.py

import cv2
import torch
from transformers import YolosForObjectDetection, YolosImageProcessor
from PIL import Image
import numpy as np

# -----------------------------
# Cấu hình model Hugging Face
# -----------------------------
MODEL_NAME = "username/mask-detection-model"  # Thay bằng model thực tế

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

# Load model và processor
model = YolosForObjectDetection.from_pretrained(MODEL_NAME).to(device)
processor = YolosImageProcessor.from_pretrained(MODEL_NAME)

# -----------------------------
# Khởi động webcam
# -----------------------------
cap = cv2.VideoCapture(0)  # 0 là camera mặc định

if not cap.isOpened():
    print("Không thể mở camera")
    exit()

# -----------------------------
# Hàm vẽ bounding box
# -----------------------------
def draw_boxes(frame, boxes, labels, scores, threshold=0.5):
    for box, label, score in zip(boxes, labels, scores):
        if score < threshold:
            continue
        x_min, y_min, x_max, y_max = map(int, box)
        color = (0, 255, 0) if label == 1 else (0, 0, 255)  # Mask=Green, No Mask=Red
        text = f"{'Mask' if label==1 else 'No Mask'}: {score:.2f}"
        
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(frame, text, (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

# -----------------------------
# Loop realtime
# -----------------------------
print("Starting webcam, press 'q' to quit...")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc frame từ camera")
        break

    # Convert frame sang PIL Image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Chuẩn bị input cho model
    inputs = processor(images=img, return_tensors="pt").to(device)

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Lấy bounding boxes, labels, scores
    boxes = outputs.pred_boxes[0].cpu().numpy()      # shape: [num_boxes, 4]
    scores = outputs.pred_logits[0].softmax(-1).max(-1).values.cpu().numpy()  # confidence
    labels = outputs.pred_logits[0].softmax(-1).argmax(-1).cpu().numpy()      # 1=Mask, 0=No Mask

    # Vẽ boxes lên frame
    frame = draw_boxes(frame, boxes, labels, scores, threshold=0.5)

    # Hiển thị frame
    cv2.imshow("Mask Detection Realtime", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
