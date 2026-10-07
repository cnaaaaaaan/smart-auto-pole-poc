import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo26n.pt")

# Vehicle classes in COCO
vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Vehicle detection started.")
print("Press Q to stop.")

while True:
    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read frame.")
        break

    # Detect only vehicles
    results = model(
    frame,
    classes=vehicle_classes,
    conf=0.25,
    imgsz=640,
    verbose=False
)
    

    # Draw detections
    annotated_frame = results[0].plot()

    # Display
    cv2.imshow("Smart Auto Pole - Vehicle Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("Vehicle detection completed.")

