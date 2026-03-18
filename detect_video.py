import cv2
import subprocess
import os
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

model = YOLO("yolov8n.pt")

VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

SLOW_MOTION_FACTOR = 3

# Instead of fixed pixel threshold, we use a fraction of the box size
# If a vehicle moves less than this fraction of its own width, treat as stationary
# 0.02 = must move at least 2% of its bounding box width per frame to count as moving
MIN_MOVEMENT_RATIO = 0.02

# Meters per pixel is also now relative to box size
# We estimate real-world scale using the box width
# Average car width = 1.8 meters — adjust if using bus/truck focused video
AVG_CAR_WIDTH_METERS = 1.8

def get_meters_per_pixel(box_width_pixels):
    """Dynamically calculate scale based on how big the vehicle appears in frame."""
    if box_width_pixels == 0:
        return 0.05
    return AVG_CAR_WIDTH_METERS / box_width_pixels

def estimate_speed(prev_center, curr_center, box_width, fps, meters_per_pixel):
    dx = curr_center[0] - prev_center[0]
    dy = curr_center[1] - prev_center[1]
    pixel_dist = np.sqrt(dx**2 + dy**2)

    # Threshold is relative to box size — small distant cars get smaller threshold
    min_movement_pixels = max(1.0, box_width * MIN_MOVEMENT_RATIO)

    if pixel_dist < min_movement_pixels:
        return 0.0, False  # stationary

    meters    = pixel_dist * meters_per_pixel
    speed_mps = meters * fps
    speed_kmh = speed_mps * 3.6
    return round(speed_kmh, 1), True

def detect_vehicles(video_path, output_path):
    print("Processing started...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 20

    temp_path = output_path.replace("result.mp4", "temp_result.mp4")

    out = cv2.VideoWriter(
        temp_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    prev_centers  = defaultdict(lambda: None)
    speed_history = defaultdict(list)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=list(VEHICLE_CLASSES.keys()),
            conf=0.3
        )

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes   = results[0].boxes.xyxy.cpu().numpy()
            ids     = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, obj_id, cls_id in zip(boxes, ids, classes):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                curr_center  = (cx, cy)
                box_width    = x2 - x1

                label_name = VEHICLE_CLASSES.get(cls_id, "vehicle")

                # Dynamic scale: bigger box = closer = more meters per pixel movement
                meters_per_pixel = get_meters_per_pixel(box_width)

                speed_kmh = 0.0
                is_moving = False

                if prev_centers[obj_id] is not None:
                    speed_kmh, is_moving = estimate_speed(
                        prev_centers[obj_id],
                        curr_center,
                        box_width,
                        fps,
                        meters_per_pixel
                    )

                    if is_moving:
                        speed_history[obj_id].append(speed_kmh)
                        # Keep last 7 frames for rolling average
                        if len(speed_history[obj_id]) > 7:
                            speed_history[obj_id].pop(0)
                        speed_kmh = round(
                            sum(speed_history[obj_id]) / len(speed_history[obj_id]), 1
                        )
                    else:
                        speed_history[obj_id].clear()
                        speed_kmh = 0.0

                prev_centers[obj_id] = curr_center

                # Color: gray if stationary, type color if moving
                colors = {
                    "car":        (0, 255, 0),
                    "truck":      (0, 165, 255),
                    "bus":        (255, 0, 0),
                    "motorcycle": (255, 0, 255),
                    "bicycle":    (0, 255, 255),
                }
                color = colors.get(label_name, (255, 255, 255))
                if not is_moving:
                    color = (160, 160, 160)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                speed_text = f"{speed_kmh} km/h" if is_moving else "stationary"
                label = f"{label_name} #{obj_id} | {speed_text}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
                cv2.rectangle(frame, (x1, y1 - 22), (x1 + text_size[0] + 4, y1), color, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

                cv2.circle(frame, curr_center, 4, color, -1)

        out.write(frame)

    cap.release()
    out.release()

    pts_multiplier = float(SLOW_MOTION_FACTOR)

    if os.path.exists(output_path):
        os.remove(output_path)

    subprocess.run([
        "ffmpeg",
        "-i", temp_path,
        "-vf", f"setpts={pts_multiplier}*PTS",
        "-r", str(fps),
        "-vcodec", "libx264",
        "-crf", "23",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        output_path
    ], check=True)

    os.remove(temp_path)
    print("Processing completed!")