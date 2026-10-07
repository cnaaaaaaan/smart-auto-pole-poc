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

# Heavy vehicle classes
# 5 = Bus
# 7 = Truck
heavy_vehicle_classes = [5, 7]

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Heavy vehicle detection started.")
print("Press Q to stop.")

while True:

    # Capture frame
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

    # Draw ONLY bounding boxes.
    # YOLO's default labels are disabled to avoid overlapping text.
    annotated_frame = result.plot(
        labels=False,
        conf=False
    )

    # Process detected vehicles
    if result.boxes is not None:

        for box in result.boxes:

            # Get class ID
            class_id = int(box.cls[0])

            # Get confidence
            confidence = float(box.conf[0])

            # Get vehicle name
            vehicle_name = vehicle_names.get(
                class_id,
                "UNKNOWN"
            )

            # Get bounding box coordinates
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # Vehicle label
            vehicle_label = (
                f"{vehicle_name} {confidence:.2f}"
            )

            # Display vehicle type
            cv2.putText(
                annotated_frame,
                vehicle_label,
                (x1, max(y1 - 40, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # Check whether vehicle is heavy
            if class_id in heavy_vehicle_classes:

                vehicle_status = "HEAVY VEHICLE"

                # Display heavy vehicle warning
                cv2.putText(
                    annotated_frame,
                    vehicle_status,
                    (x1, max(y1 - 10, 55)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    3
                )

            else:

                vehicle_status = "NORMAL VEHICLE"

            # Print information to terminal
            print(
                f"Vehicle: {vehicle_name} | "
                f"Confidence: {confidence:.2f} | "
                f"Status: {vehicle_status}"
            )

    # Display video
    cv2.imshow(
        "Smart Auto Pole - Heavy Vehicle Detection",
        annotated_frame
    )

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release camera
camera.release()
cv2.destroyAllWindows()

print("Heavy vehicle detection completed.")