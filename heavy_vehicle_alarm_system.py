import cv2
import subprocess
import os
import signal
from ultralytics import YOLO


# ==========================================
# LOAD YOLO MODEL
# ==========================================

model = YOLO("yolo26n.pt")


# ==========================================
# VEHICLE CLASSES
# ==========================================

vehicle_classes = [2, 3, 5, 7]

vehicle_names = {
    2: "CAR",
    3: "MOTORCYCLE",
    5: "BUS",
    7: "TRUCK"
}


# ==========================================
# HEAVY VEHICLE CLASSES
# ==========================================

heavy_vehicle_classes = [5, 7]


# ==========================================
# ALARM SETTINGS
# ==========================================

alarm_process = None

# Number of consecutive frames required
# before turning the alarm ON
DETECTION_FRAMES_REQUIRED = 8

# Number of consecutive missed frames
# required before turning the alarm OFF
MISS_FRAMES_REQUIRED = 15

heavy_detection_count = 0
heavy_miss_count = 0

alarm_is_on = False


# ==========================================
# START ALARM
# ==========================================

def start_alarm():

    global alarm_process
    global alarm_is_on

    if not alarm_is_on:

        print("🚨 HEAVY VEHICLE DETECTED!")
        print("🔊 ALARM ON")

        alarm_process = subprocess.Popen(
            [
                "sh",
                "-c",
                "while true; do afplay /System/Library/Sounds/Glass.aiff; done"
            ],
            start_new_session=True
        )

        alarm_is_on = True


# ==========================================
# STOP ALARM
# ==========================================

def stop_alarm():

    global alarm_process
    global alarm_is_on

    if alarm_is_on:

        print("🔇 ALARM OFF")

        if alarm_process is not None:

            try:
                os.killpg(
                    os.getpgid(alarm_process.pid),
                    signal.SIGTERM
                )
            except ProcessLookupError:
                pass

            alarm_process = None

        alarm_is_on = False


# ==========================================
# OPEN WEBCAM
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("ERROR: Could not open webcam.")
    exit()


print("Heavy vehicle alarm system started.")
print("Press Q to stop.")


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    # --------------------------------------
    # CAPTURE FRAME
    # --------------------------------------

    success, frame = camera.read()

    if not success:

        print("ERROR: Could not read frame.")
        break


    # --------------------------------------
    # YOLO DETECTION
    # --------------------------------------

    results = model(
        frame,
        classes=vehicle_classes,
        conf=0.25,
        imgsz=640,
        verbose=False
    )

    result = results[0]


    # --------------------------------------
    # DRAW DETECTIONS
    # --------------------------------------

    annotated_frame = result.plot(
        labels=False,
        conf=False
    )


    # --------------------------------------
    # CHECK FOR HEAVY VEHICLE
    # --------------------------------------

    heavy_vehicle_detected = False


    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0])

            confidence = float(box.conf[0])

            vehicle_name = vehicle_names.get(
                class_id,
                "UNKNOWN"
            )


            # Bounding box coordinates

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # Vehicle label

            vehicle_label = (
                f"{vehicle_name} {confidence:.2f}"
            )


            cv2.putText(
                annotated_frame,
                vehicle_label,
                (x1, max(y1 - 40, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


            # ----------------------------------
            # HEAVY VEHICLE DETECTED
            # ----------------------------------

            if class_id in heavy_vehicle_classes:

                heavy_vehicle_detected = True

                cv2.putText(
                    annotated_frame,
                    "HEAVY VEHICLE",
                    (x1, max(y1 - 10, 55)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    3
                )


    # ==========================================
    # DETECTION COUNTERS
    # ==========================================

    if heavy_vehicle_detected:

        # Heavy vehicle detected
        heavy_detection_count += 1

        # Reset missed-frame counter
        heavy_miss_count = 0

        # Limit detection counter
        if heavy_detection_count > DETECTION_FRAMES_REQUIRED:

            heavy_detection_count = DETECTION_FRAMES_REQUIRED


    else:

        # No heavy vehicle detected
        heavy_detection_count = 0

        # Increase missed-frame counter
        heavy_miss_count += 1

        # Limit missed-frame counter
        if heavy_miss_count > MISS_FRAMES_REQUIRED:

            heavy_miss_count = MISS_FRAMES_REQUIRED


    # ==========================================
    # TURN ALARM ON
    # ==========================================

    if not alarm_is_on:

        if heavy_detection_count >= DETECTION_FRAMES_REQUIRED:

            start_alarm()


    # ==========================================
    # TURN ALARM OFF
    # ==========================================

    else:

        if heavy_miss_count >= MISS_FRAMES_REQUIRED:

            stop_alarm()


    # ==========================================
    # DISPLAY ALARM STATUS
    # ==========================================

    if alarm_is_on:

        cv2.putText(
            annotated_frame,
            "ALARM: ON",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3
        )

    else:

        cv2.putText(
            annotated_frame,
            "ALARM: OFF",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            3
        )


    # ==========================================
    # DISPLAY COUNTERS
    # ==========================================

    cv2.putText(
        annotated_frame,
        f"Detection: {heavy_detection_count}/{DETECTION_FRAMES_REQUIRED}",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Missed: {heavy_miss_count}/{MISS_FRAMES_REQUIRED}",
        (30, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ==========================================
    # SHOW CAMERA
    # ==========================================

    cv2.imshow(
        "Smart Auto Pole - Heavy Vehicle Alarm System",
        annotated_frame
    )


    # ==========================================
    # PRESS Q TO STOP
    # ==========================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==========================================
# CLEANUP
# ==========================================

stop_alarm()

camera.release()

cv2.destroyAllWindows()

print("Heavy vehicle alarm system stopped.")