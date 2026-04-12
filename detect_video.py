import cv2
import subprocess
import os
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# ── cuDF with pandas fallback ────────────────────────────────────────────────
try:
    import cudf as pd
    BACKEND = "cuDF (GPU)"
    print("✅ cuDF loaded — GPU-accelerated dataframe analysis enabled.")
except ImportError:
    import pandas as pd
    BACKEND = "pandas (CPU fallback)"
    print("⚠️  cuDF not found — falling back to pandas.")
# ─────────────────────────────────────────────────────────────────────────────

model = YOLO("yolov8n.pt")

VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

SLOW_MOTION_FACTOR  = 3
MIN_MOVEMENT_RATIO  = 0.02
AVG_CAR_WIDTH_METERS = 1.8


def get_meters_per_pixel(box_width_pixels):
    if box_width_pixels == 0:
        return 0.05
    return AVG_CAR_WIDTH_METERS / box_width_pixels


def estimate_speed(prev_center, curr_center, box_width, fps, meters_per_pixel):
    dx = curr_center[0] - prev_center[0]
    dy = curr_center[1] - prev_center[1]
    pixel_dist = np.sqrt(dx**2 + dy**2)

    min_movement_pixels = max(1.0, box_width * MIN_MOVEMENT_RATIO)
    if pixel_dist < min_movement_pixels:
        return 0.0, False

    meters    = pixel_dist * meters_per_pixel
    speed_mps = meters * fps
    speed_kmh = min(speed_mps * 3.6, 200.0)   # clamp to realistic max
    return round(speed_kmh, 1), True


def analyze_with_dataframe(speed_history):
    """
    Run summary analytics using cuDF (GPU) or pandas (CPU).
    The API is identical — same code works for both.
    """
    print(f"\n{'='*50}")
    print(f"  Vehicle Speed Analysis  [{BACKEND}]")
    print(f"{'='*50}")

    data = []
    for obj_id, speeds in speed_history.items():
        if speeds:
            data.append({
                "vehicle_id": int(obj_id),
                "avg_speed_kmh": round(sum(speeds) / len(speeds), 1),
                "max_speed_kmh": round(max(speeds), 1),
                "min_speed_kmh": round(min(speeds), 1),
                "samples":       len(speeds),
            })

    if not data:
        print("No vehicle movement data collected.")
        return

    df = pd.DataFrame(data)

    print("\n📊 Per-Vehicle Speed Summary:")
    print(df.to_string(index=False))

    print("\n🏎️  Top 5 Fastest Vehicles:")
    top = df.sort_values(by="max_speed_kmh", ascending=False).head(5)
    print(top.to_string(index=False))

    print("\n📈 Fleet Statistics:")
    print(f"   Total vehicles tracked : {len(df)}")
    print(f"   Fleet avg speed        : {df['avg_speed_kmh'].mean():.1f} km/h")
    print(f"   Highest recorded speed : {df['max_speed_kmh'].max():.1f} km/h")
    print(f"{'='*50}\n")


def detect_vehicles(video_path, output_path):
    print("Processing started...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video '{video_path}'")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20.0

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temp_path = output_path.replace("result.mp4", "temp_result.mp4")

    out = cv2.VideoWriter(
        temp_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    prev_centers  = defaultdict(lambda: None)
    speed_history = defaultdict(list)

    colors = {
        "car":        (0, 255, 0),
        "truck":      (0, 165, 255),
        "bus":        (255, 0, 0),
        "motorcycle": (255, 0, 255),
        "bicycle":    (0, 255, 255),
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=list(VEHICLE_CLASSES.keys()),
            conf=0.3,
            verbose=False
        )

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes   = results[0].boxes.xyxy.cpu().numpy()
            ids     = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, obj_id, cls_id in zip(boxes, ids, classes):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                curr_center = (cx, cy)
                box_width   = max(x2 - x1, 1)
                label_name  = VEHICLE_CLASSES.get(cls_id, "vehicle")
                meters_per_pixel = get_meters_per_pixel(box_width)

                speed_kmh = 0.0
                is_moving = False

                if prev_centers[obj_id] is not None:
                    speed_kmh, is_moving = estimate_speed(
                        prev_centers[obj_id], curr_center,
                        box_width, fps, meters_per_pixel
                    )
                    if is_moving:
                        speed_history[obj_id].append(speed_kmh)
                        if len(speed_history[obj_id]) > 7:
                            speed_history[obj_id].pop(0)
                        speed_kmh = round(
                            sum(speed_history[obj_id]) / len(speed_history[obj_id]), 1
                        )
                    else:
                        speed_history[obj_id].clear()
                        speed_kmh = 0.0

                prev_centers[obj_id] = curr_center

                color = colors.get(label_name, (255, 255, 255))
                if not is_moving:
                    color = (160, 160, 160)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                speed_text = f"{speed_kmh} km/h" if is_moving else "stationary"
                label      = f"{label_name} #{obj_id} | {speed_text}"

                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                label_y = max(y1 - 22, 0)
                cv2.rectangle(frame, (x1, label_y), (x1 + tw + 4, label_y + 22), color, -1)
                cv2.putText(frame, label, (x1 + 2, label_y + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
                cv2.circle(frame, curr_center, 4, color, -1)

        out.write(frame)

    cap.release()
    out.release()

    # ── cuDF / pandas analytics ───────────────────────────────────────────────
    analyze_with_dataframe(speed_history)

    # ── slow-motion export via ffmpeg ─────────────────────────────────────────
    if not os.path.exists(temp_path):
        print("Error: temp video missing, skipping ffmpeg.")
        return

    if os.path.exists(output_path):
        os.remove(output_path)

    result = subprocess.run([
        "ffmpeg", "-y",
        "-i", temp_path,
        "-vf", f"setpts={float(SLOW_MOTION_FACTOR)}*PTS",
        "-r", str(fps),
        "-vcodec", "libx264",
        "-crf", "23",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        output_path
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print("ffmpeg error:", result.stderr)
    else:
        print("Slow-motion export complete.")

    os.remove(temp_path)
    print("Processing completed!")
