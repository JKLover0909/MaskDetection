# mask_detection_realtime.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the Keras model
model = load_model('/home/jkl0909/MaskDetection/Face-Mask-Detection-Model-CNN/mask_detector_model.h5')

# Function to preprocess the input frame for the model
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (224, 224))  # Resize to model's expected input size
    normalized_frame = resized_frame / 255.0  # Normalize pixel values
    return np.expand_dims(normalized_frame, axis=0)  # Add batch dimension

# Function to draw bounding boxes and labels on the frame
def draw_boxes(frame, predictions):
    for prediction in predictions:
        x_min, y_min, x_max, y_max, label, score = prediction
        color = (0, 255, 0) if label == 'Mask' else (0, 0, 255)
        text = f"{label}: {score:.2f}"
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Starting webcam, press 'q' to quit...")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the frame
    input_data = preprocess_frame(frame)

    # Perform inference
    predictions = model.predict(input_data)

    # Process predictions (adjust based on model output format)
    # Example: predictions = [[x_min, y_min, x_max, y_max, label, score], ...]

    # Draw bounding boxes and labels
    frame = draw_boxes(frame, predictions)

    # Display the frame
    cv2.imshow("Mask Detection Realtime", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
