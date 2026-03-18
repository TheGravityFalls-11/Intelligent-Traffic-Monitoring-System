from flask import Flask, render_template, request
import os
from detect_video import detect_vehicles

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static/output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        video = request.files["video"]
        video_path = os.path.join(UPLOAD_FOLDER, video.filename)
        output_path = os.path.join(OUTPUT_FOLDER, "result.mp4")  # changed to .mp4

        video.save(video_path)
        detect_vehicles(video_path, output_path)

        return render_template("index.html", video="output/result.mp4")  # changed to .mp4

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)