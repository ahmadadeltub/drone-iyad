import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
import cvzone

model = YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap = cv2.VideoCapture(1)  # 0 represents the default camera (usually the built-in webcam)

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0
tracker = Tracker()
counter = 0
person_info = {}
total_entered = 0
total_exited = 0

cy1 = 100
cy2 = 400

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    list = []
    for index, row in px.iterrows():
        x1, y1, x2, y2, d = int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[5])
        c = class_list[d]
        if 'person' in c:
            list.append([x1, y1, x2, y2])

    bbox_id = tracker.update(list)

    for bbox in bbox_id:
        x3, y3, x4, y4, track_id = bbox
        cx, cy = int((x3 + x4) / 2), int((y3 + y4) / 2)

        if track_id not in person_info:
            counter += 1
            person_info[track_id] = {"id": counter, "entered": False, "exited": False}

        if cy >= cy1 and not person_info[track_id]["entered"]:
            total_entered += 1
            person_info[track_id]["entered"] = True

        elif cy <= cy2 and not person_info[track_id]["exited"]:
            total_exited += 1
            person_info[track_id]["exited"] = True

        cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)
        cv2.putText(frame, f"Person {person_info[track_id]['id']}", (cx - 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.line(frame, (3, cy1), (1018, cy1), (0, 255, 0), 2)
    cv2.line(frame, (5, cy2), (1019, cy2), (0, 255, 255), 2)

    # Add title to the video at the top
    cv2.putText(frame, "IES QSTSS School", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 10, 0), 2)

    # Add people counter information at the bottom
    cv2.putText(frame, f"Total Entered: {total_entered}", (10, frame.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Total Exited: {total_exited}", (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
