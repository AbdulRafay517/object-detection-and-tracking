import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# Initialize variables
confidence_threshold = 0.5
skip_frames = 3
frame_count = 0
smooth_factor = 3
last_boxes = deque(maxlen=smooth_factor)

# Initialize YOLO model
model = YOLO('yolov8n.pt')

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide video file path
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % skip_frames != 0:
        continue

    # Perform object detection
    results = model(frame)

    # Process results
    current_boxes = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            conf = box.conf[0]
            cls = int(box.cls[0])

            if conf > confidence_threshold:
                current_boxes.append((x1, y1, x2, y2, conf, cls))

    # Apply temporal smoothing
    last_boxes.append(current_boxes)
    smoothed_boxes = []
    if len(last_boxes) == smooth_factor:
        for i in range(len(current_boxes)):
            avg_box = np.mean([box[i][:4] for box in last_boxes if i < len(box)], axis=0)
            smoothed_boxes.append((*avg_box, current_boxes[i][4], current_boxes[i][5]))

    # Draw bounding boxes and labels
    for x1, y1, x2, y2, conf, cls in smoothed_boxes:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f'{model.names[cls]} {conf:.2f}'
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Object Detection and Tracking', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()