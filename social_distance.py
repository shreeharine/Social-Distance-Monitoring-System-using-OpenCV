import cv2
import numpy as np
from scipy.spatial import distance as dist

# -----------------------------
# Model Paths
# -----------------------------
prototxt = "SSD_MobileNet_prototxt.txt"
model = "SSD_MobileNet.caffemodel"

# Load MobileNet SSD model
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# COCO class labels
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Video input
cap = cv2.VideoCapture("pedestrians.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]

    # Prepare frame for model
    blob = cv2.dnn.blobFromImage(
        frame, 0.007843, (300, 300), 127.5
    )

    net.setInput(blob)
    detections = net.forward()

    centroids = []
    boxes = []

    # Detect persons
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.4:
            idx = int(detections[0, 0, i, 1])

            if CLASSES[idx] != "person":
                continue

            box = detections[0, 0, i, 3:7] * \
                  np.array([w, h, w, h])

            (startX, startY, endX, endY) = box.astype("int")

            cX = int((startX + endX) / 2)
            cY = int((startY + endY) / 2)

            centroids.append((cX, cY))
            boxes.append((startX, startY, endX, endY))

    # -----------------------------
    # Distance Calculation
    # -----------------------------
    risk_high = set()
    risk_medium = set()

    if len(centroids) >= 2:
        D = dist.cdist(centroids, centroids, metric="euclidean")

        for i in range(D.shape[0]):
            for j in range(i + 1, D.shape[1]):

                if D[i, j] < 75:
                    risk_high.add(i)
                    risk_high.add(j)

                elif D[i, j] < 150:
                    risk_medium.add(i)
                    risk_medium.add(j)

    # -----------------------------
    # Draw Boxes
    # -----------------------------
    for i, (startX, startY, endX, endY) in enumerate(boxes):

        color = (0, 255, 0)  # Safe

        if i in risk_high:
            color = (0, 0, 255)
        elif i in risk_medium:
            color = (0, 255, 255)

        cv2.rectangle(
            frame,
            (startX, startY),
            (endX, endY),
            color,
            2,
        )

    # -----------------------------
    # Display Info
    # -----------------------------
    cv2.putText(
        frame,
        f"High Risk: {len(risk_high)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Medium Risk: {len(risk_medium)}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Detected Persons: {len(boxes)}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Social Distance Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()