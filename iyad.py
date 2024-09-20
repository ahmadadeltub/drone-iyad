import cv2
from ultralytics import YOLO
import numpy as np
import time

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Path to YOLOv8 model

# Open the video file or drone camera stream
video_path = '1.mp4'  # Provide the correct video path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

previous_bboxes = []
last_motion_times = []
motionless_threshold = 2  # Threshold for small motion (distance in pixels)
motionless_duration_required = 2  # Duration in seconds to consider a person as motionless

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO Detection
    results = model(frame)
    current_bboxes = []
    person_detected = False

    for result in results[0].boxes:
        if result.cls == 0:  # Class 0 is 'person' in YOLOv8
            bbox = result.xyxy[0].cpu().numpy()
            current_bboxes.append(bbox)
            person_detected = True

    if person_detected:
        for i, current_bbox in enumerate(current_bboxes):
            if i < len(previous_bboxes):
                previous_bbox = previous_bboxes[i]
                last_motion_time = last_motion_times[i]

                # Calculate changes in position and size
                delta_pos = np.linalg.norm(previous_bbox[:2] - current_bbox[:2])
                delta_size = np.linalg.norm(previous_bbox[2:4] - current_bbox[2:4])

                # Check if the person is motionless
                if delta_pos < motionless_threshold and delta_size < motionless_threshold:
                    time_elapsed = time.time() - last_motion_time
                    if time_elapsed >= motionless_duration_required:
                        color = (0, 0, 255)  # Red for not moving (presumed dead)
                        label = 'Not Moving (Dead)'
                    else:
                        color = (0, 255, 0)  # Green for moving
                        label = 'Moving'
                else:
                    # Person is moving
                    color = (0, 255, 0)  # Green for moving
                    label = 'Moving'
                    last_motion_times[i] = time.time()  # Reset the timer

            else:
                # First frame or new detection, assume moving
                color = (0, 255, 0)
                label = 'Moving'
                last_motion_times.append(time.time())  # Reset the timer

            # Draw the bounding box and label
            x1, y1, x2, y2 = current_bbox[:4].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Update previous bounding boxes
        previous_bboxes = current_bboxes

    else:
        # No person detected but was present in the previous frame
        for i, previous_bbox in enumerate(previous_bboxes):
            # Check if the last detected person is still motionless
            if time.time() - last_motion_times[i] >= motionless_duration_required:
                color = (0, 0, 255)  # Red for presumed dead (motionless)
                label = 'Not Moving (Dead)'
                
                # Draw a bounding box around the previous person
                x1, y1, x2, y2 = previous_bbox[:4].astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            else:
                color = (0, 255, 0)  # Green for moving (though technically no person detected)
                label = 'Moving'
                
                # Draw a bounding box around the previous person
                x1, y1, x2, y2 = previous_bbox[:4].astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Show the frame with the bounding boxes and labels
    cv2.imshow('Dead/Alive Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release video capture object and close windows
cap.release()
cv2.destroyAllWindows()
