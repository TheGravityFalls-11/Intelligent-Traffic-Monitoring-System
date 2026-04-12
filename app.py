from flask import Flask, render_template, request, jsonify
import os
import threading
from detect_video import detect_vehicles

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static/output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Global job state
job_status = {
    "state": "idle",   # idle | processing | done | error
    "error": ""
}


def run_detection(video_path, output_path):
    global job_status
    try:
        job_status["state"] = "processing"
        job_status["error"] = ""
        detect_vehicles(video_path, output_path)
        job_status["state"] = "done"
    except Exception as e:
        job_status["state"] = "error"
        job_status["error"] = str(e)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    global job_status

    if job_status["state"] == "processing":
        return jsonify({"error": "Already processing a video."}), 429

    video = request.files.get("video")
    if not video or video.filename == "":
        return jsonify({"error": "No video file selected."}), 400

    video_path  = os.path.join(UPLOAD_FOLDER, video.filename)
    output_path = os.path.join(OUTPUT_FOLDER, "result.mp4")
    video.save(video_path)

    # Run detection in background thread so request returns immediately
    thread = threading.Thread(target=run_detection, args=(video_path, output_path), daemon=True)
    thread.start()

    return jsonify({"message": "Processing started."})


@app.route("/status", methods=["GET"])
def status():
    return jsonify(job_status)


if __name__ == "__main__":
    from waitress import serve
    print("Server running on http://127.0.0.1:5000")
    serve(app, host="127.0.0.1", port=5000, threads=4)
