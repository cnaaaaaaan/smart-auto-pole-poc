import cv2
import numpy as np
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
# CAMERA
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    exit()


# ==========================================
# REFERENCE RESOLUTION
# ==========================================

REF_WIDTH = 1600
REF_HEIGHT = 900


# ==========================================
# PARKING SLOT COORDINATES
# ==========================================

slots = {
    "S1": [
        (55, 510),
        (300, 510),
        (305, 695),
        (55, 680)
    ],

    "S2": [
        (305, 510),
        (615, 505),
        (620, 695),
        (305, 695)
    ],

    "S3": [
        (615, 505),
        (960, 495),
        (965, 700),
        (620, 695)
    ],

    "S4": [
        (960, 495),
        (1300, 485),
        (1305, 705),
        (965, 700)
    ],

    "S5": [
        (1300, 485),
        (1540, 475),
        (1540, 700),
        (1305, 705)
    ]
}


print("Parking occupancy detection started.")
print("Press Q to stop.")


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    # --------------------------------------
    # READ CAMERA
    # --------------------------------------

    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read frame.")
        break


    # --------------------------------------
    # GET CAMERA SIZE
    # --------------------------------------

    height, width = frame.shape[:2]

    scale_x = width / REF_WIDTH
    scale_y = height / REF_HEIGHT


    # --------------------------------------
    # CREATE SCALED SLOT POLYGONS
    # --------------------------------------

    scaled_slots = {}

    for slot_name, points in slots.items():

        scaled_points = [
            (
                int(x * scale_x),
                int(y * scale_y)
            )
            for x, y in points
        ]

        scaled_slots[slot_name] = scaled_points


    # --------------------------------------
    # YOLO VEHICLE DETECTION
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
    # DRAW YOLO DETECTIONS
    # --------------------------------------

    annotated_frame = result.plot(
        labels=False,
        conf=False
    )


    # --------------------------------------
    # INITIAL SLOT STATUS
    # --------------------------------------

    slot_status = {}

    for slot_name in slots:
        slot_status[slot_name] = {
            "occupied": False,
            "vehicle": None
        }


    # --------------------------------------
    # CHECK DETECTED VEHICLES
    # --------------------------------------

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            vehicle_name = vehicle_names.get(
                class_id,
                "UNKNOWN"
            )


            # Bounding box
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # ----------------------------------
            # FIND VEHICLE CENTER
            # ----------------------------------

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)


            # ----------------------------------
            # DISPLAY VEHICLE
            # ----------------------------------

            vehicle_label = (
                f"{vehicle_name} {confidence:.2f}"
            )

            cv2.putText(
                annotated_frame,
                vehicle_label,
                (
                    x1,
                    max(y1 - 10, 30)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


            # Draw center point
            cv2.circle(
                annotated_frame,
                (center_x, center_y),
                6,
                (255, 0, 0),
                -1
            )


            # ----------------------------------
            # CHECK WHICH SLOT CONTAINS VEHICLE
            # ----------------------------------

            for slot_name, points in scaled_slots.items():

                polygon = np.array(
                    points,
                    dtype=np.int32
                )

                inside = cv2.pointPolygonTest(
                    polygon,
                    (center_x, center_y),
                    False
                )


                if inside >= 0:

                    slot_status[slot_name]["occupied"] = True

                    slot_status[slot_name]["vehicle"] = vehicle_name

                    break


    # ==========================================
    # DRAW PARKING SLOTS
    # ==========================================

    occupied_count = 0

    for slot_name, points in scaled_slots.items():

        polygon = np.array(
            points,
            dtype=np.int32
        )


        occupied = slot_status[slot_name]["occupied"]

        vehicle = slot_status[slot_name]["vehicle"]


        # --------------------------------------
        # SLOT COLOR
        # --------------------------------------

        if occupied:

            # RED = OCCUPIED
            line_color = (0, 0, 255)

            occupied_count += 1

            status_text = "OCCUPIED"

        else:

            # GREEN = VACANT
            line_color = (0, 255, 0)

            status_text = "VACANT"


        # --------------------------------------
        # DRAW SLOT
        # --------------------------------------

        cv2.polylines(
            annotated_frame,
            [polygon],
            True,
            line_color,
            3
        )


        # --------------------------------------
        # SLOT LABEL
        # --------------------------------------

        label_x = points[0][0] + 10
        label_y = points[0][1] - 15


        cv2.putText(
            annotated_frame,
            slot_name,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            line_color,
            2
        )


        # --------------------------------------
        # STATUS LABEL
        # --------------------------------------

        status_y = points[0][1] + 30


        if vehicle:

            display_status = (
                f"{status_text} - {vehicle}"
            )

        else:

            display_status = status_text


        cv2.putText(
            annotated_frame,
            display_status,
            (label_x, status_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            line_color,
            2
        )


    # ==========================================
    # PARKING SUMMARY
    # ==========================================

    total_slots = len(slots)

    vacant_count = total_slots - occupied_count


    cv2.putText(
        annotated_frame,
        f"Occupied: {occupied_count}/{total_slots}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )


    cv2.putText(
        annotated_frame,
        f"Vacant: {vacant_count}/{total_slots}",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    # ==========================================
    # DISPLAY
    # ==========================================

    cv2.imshow(
        "Smart Auto Pole - Parking Occupancy",
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

camera.release()

cv2.destroyAllWindows()

print("Parking occupancy detection completed.")