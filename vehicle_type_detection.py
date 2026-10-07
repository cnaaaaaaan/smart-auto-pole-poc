import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo26n.pt")

# Vehicle classes
vehicle_classes = [2, 3, 5, 7]

# Vehicle names
vehicle_names = {
    2: "CAR",
    3: "MOTORCYCLE",
    5: "BUS",
    7: "TRUCK"
}

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Vehicle type detection started.")
print("Press Q to stop.")

while True:

    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read frame.")
        break

    # Run YOLO detection
    results = model(
        frame,
        classes=vehicle_classes,
        conf=0.25,
        imgsz=640,
        verbose=False
    )

    result = results[0]

    # Draw detections
    annotated_frame = result.plot()

    # Process detected vehicles
    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            vehicle_name = vehicle_names.get(
                class_id,
                "UNKNOWN"
            )

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            label = f"{vehicle_name} {confidence:.2f}"

            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            print(
                f"Vehicle: {vehicle_name} | "
                f"Confidence: {confidence:.2f}"
            )

    # Show video
    cv2.imshow(
        "Smart Auto Pole - Vehicle Type Detection",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("Vehicle type detection completed.")