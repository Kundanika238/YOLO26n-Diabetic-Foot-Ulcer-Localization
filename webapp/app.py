from flask import (
    Flask, render_template, request,
    jsonify, send_file, url_for, abort
)

from ultralytics import YOLO
from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError
from io import BytesIO
from collections import OrderedDict

import torch
import secrets
import threading
import time
import sys

# ==================================================
# APPLICATION CONFIGURATION
# ==================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# Project paths
BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR.parent

# Final trained YOLO26n model
MODEL_PATH = ROOT / "models" / "best.pt"

# Optional local dataset (not included in GitHub)
DATASET_PATH = ROOT / "data" / "DFU_Research_SCREENED_DRAFT"


IMAGE_SIZE = 512
CONFIDENCE_THRESHOLD = 0.20
NMS_IOU = 0.70

DEVICE = 0 if torch.cuda.is_available() else "cpu"

# Uploaded images are processed in memory.
# Generated images are kept temporarily, not permanently.
RESULT_CACHE = OrderedDict()
CACHE_LOCK = threading.Lock()
MODEL_LOCK = threading.Lock()

CACHE_LIFETIME = 15 * 60
CACHE_LIMIT = 12

# ==================================================
# LOAD THE CORRECT TRAINED MODEL
# ==================================================

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Research model not found: {MODEL_PATH}"
    )

print("=" * 60)
print("DFUVISION - RESEARCH WEB APPLICATION")
print("=" * 60)
print("Model:", MODEL_PATH.name)
print("Model directory:", MODEL_PATH.parent.parent.name)
print("Image size:", IMAGE_SIZE)
print("Confidence:", CONFIDENCE_THRESHOLD)
print("Device:", DEVICE)

model = YOLO(str(MODEL_PATH))

# ==================================================
# TEMPORARY RESULT STORAGE
# ==================================================

def save_temporary_result(image_bytes):

    token = secrets.token_urlsafe(24)
    now = time.monotonic()

    with CACHE_LOCK:

        expired = [
            key for key, value in RESULT_CACHE.items()
            if now - value[0] > CACHE_LIFETIME
        ]

        for key in expired:
            RESULT_CACHE.pop(key, None)

        while len(RESULT_CACHE) >= CACHE_LIMIT:
            RESULT_CACHE.popitem(last=False)

        RESULT_CACHE[token] = (now, image_bytes)

    return token


def get_temporary_result(token):

    with CACHE_LOCK:

        entry = RESULT_CACHE.get(token)

        if entry is None:
            return None

        created_at, image_bytes = entry

        if time.monotonic() - created_at > CACHE_LIFETIME:
            RESULT_CACHE.pop(token, None)
            return None

        return image_bytes


# ==================================================
# BLOCK OLD PUBLIC IMAGE DIRECTORIES
# ==================================================

@app.before_request
def protect_old_images():

    if request.path.startswith("/static/uploads/"):
        abort(404)

    if request.path.startswith("/static/results/"):
        abort(404)


# ==================================================
# HOME PAGE
# ==================================================

@app.route("/")
def home():

    return render_template("index.html")


# ==================================================
# IMAGE UPLOAD AND DFU DETECTION
# ==================================================

@app.route("/detect", methods=["POST"])
def detect():

    if "image" not in request.files:
        return jsonify({
            "success": False,
            "message": "Please upload a foot image."
        }), 400

    uploaded_file = request.files["image"]

    if not uploaded_file.filename:
        return jsonify({
            "success": False,
            "message": "Please select an image."
        }), 400

    try:

        raw = uploaded_file.read()

        if not raw:
            return jsonify({
                "success": False,
                "message": "The uploaded file is empty."
            }), 400

        # Validate the actual image contents.
        image = Image.open(BytesIO(raw))

        if image.format not in ("JPEG", "PNG"):
            return jsonify({
                "success": False,
                "message": "Only JPG, JPEG and PNG images are supported."
            }), 400

        if image.width * image.height > 24000000:
            return jsonify({
                "success": False,
                "message": "Image resolution is too large."
            }), 400

        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        threshold = CONFIDENCE_THRESHOLD
        # Run inference using the research model.
        # No new model training is performed.

        with MODEL_LOCK:

            results = model.predict(
                source=image,
                imgsz=IMAGE_SIZE,
                conf=threshold,
                iou=NMS_IOU,
                device=DEVICE,
                verbose=False
            )

        result = results[0]

        detections = []

        for box in result.boxes:

            confidence = float(box.conf[0])

            x1, y1, x2, y2 = [
                round(float(v), 1)
                for v in box.xyxy[0].tolist()
            ]

            detections.append({
                "class": "DFU",
                "confidence": round(
                    confidence * 100, 2
                ),
                "bounding_box": [x1, y1, x2, y2]
            })

        count = len(detections)

        highest_confidence = max(
            (
                item["confidence"]
                for item in detections
            ),
            default=0
        )

        # Plot the detected bounding boxes.
        # Ultralytics returns an image in BGR order.
        annotated_bgr = result.plot(
            labels=True,
            conf=True
        )

        annotated_rgb = Image.fromarray(
            annotated_bgr[:, :, ::-1]
        )

        output_buffer = BytesIO()

        annotated_rgb.save(
            output_buffer,
            format="JPEG",
            quality=90
        )

        token = save_temporary_result(
            output_buffer.getvalue()
        )

        if count > 0:
            status = "Potential DFU Region Detected"
        else:
            status = "No DFU Region Detected"

        return jsonify({
            "success": True,
            "status": status,
            "confidence": highest_confidence,
            "detections": detections,
            "detection_count": count,
            "result_image": url_for(
                "view_result",
                token=token
            ),
            "download_url": url_for(
                "download_result",
                token=token
            ),
            "threshold": threshold,
            "model": "YOLO26n"
        })

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError
    ):

        return jsonify({
            "success": False,
            "message": "Invalid or damaged image file."
        }), 400

    except Exception:

        app.logger.exception(
            "Detection failed"
        )

        return jsonify({
            "success": False,
            "message": "Detection failed. Please try another image."
        }), 500


# ==================================================
# DISPLAY DETECTION IMAGE
# ==================================================

@app.route("/result/<token>")
def view_result(token):

    image_bytes = get_temporary_result(token)

    if image_bytes is None:
        abort(404)

    response = send_file(
        BytesIO(image_bytes),
        mimetype="image/jpeg"
    )

    response.headers["Cache-Control"] = "no-store"

    return response


# ==================================================
# DOWNLOAD DETECTION IMAGE
# ==================================================

@app.route("/download/<token>")
def download_result(token):

    image_bytes = get_temporary_result(token)

    if image_bytes is None:
        abort(404)

    response = send_file(
        BytesIO(image_bytes),
        mimetype="image/jpeg",
        as_attachment=True,
        download_name="DFU_detection_result.jpg"
    )

    response.headers["Cache-Control"] = "no-store"

    return response


# ==================================================
# UPLOAD SIZE ERROR
# ==================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "success": False,
        "message": "Image is too large. Maximum upload size is 10 MB."
    }), 413


# ==================================================
# AUTOMATED BACKEND TEST
# ==================================================

def run_self_test():

    print("\n========== WEB APPLICATION SELF-TEST ==========")

    sample_dir = DATASET_PATH / "val" / "images"

    samples = sorted(
        p for p in sample_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not samples:
        raise RuntimeError("No sample images found.")

    sample = samples[0]

    print("Sample image:", sample.name)

    with app.test_client() as client:

        home_response = client.get("/")

        assert home_response.status_code == 200, (
            "Home page failed."
        )

        print("Home page: PASS")

        missing_response = client.post("/detect")

        assert missing_response.status_code == 400, (
            "Missing file validation failed."
        )

        print("Missing file validation: PASS")

        with sample.open("rb") as image_file:

            response = client.post(
                "/detect",
                data={
                    "image": (
                        image_file,
                        sample.name
                    )
                },
                content_type="multipart/form-data"
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Detection request failed: {response.get_json()}"
            )

        data = response.get_json()

        assert data["success"] is True

        print("Model prediction: PASS")
        print("Detected regions:", data["detection_count"])

        result_response = client.get(
            data["result_image"]
        )

        assert result_response.status_code == 200

        assert result_response.mimetype == "image/jpeg"

        print("Result image: PASS")

        download_response = client.get(
            data["download_url"]
        )

        assert download_response.status_code == 200

        print("Download endpoint: PASS")

    print("\n======================================")
    print("BACKEND SELF-TEST PASSED")
    print("======================================")
    print("Model weights were not modified.")
    print("No patient image was saved to disk.")


# ==================================================
# START APPLICATION
# ==================================================


@app.route("/inspect", methods=["POST"])
def inspect_selected_region():
    import math

    upload = request.files.get("image")

    if upload is None or not upload.filename:
        return jsonify(
            success=False,
            message="Please choose an image."
        ), 400

    try:
        raw = upload.read()

        if not raw:
            return jsonify(
                success=False,
                message="The image is empty."
            ), 400

        image = Image.open(BytesIO(raw))

        if image.format not in ("JPEG", "PNG"):
            return jsonify(
                success=False,
                message="Only JPG or PNG supported."
            ), 400

        if image.width * image.height > 24_000_000:
            return jsonify(
                success=False,
                message="Image resolution too large."
            ), 400

        image = ImageOps.exif_transpose(image).convert("RGB")

        width, height = image.size

        coords = [
            float(request.form[k])
            for k in ("x1", "y1", "x2", "y2")
        ]

        if not all(math.isfinite(v) for v in coords):
            raise ValueError("Invalid coordinates")

        ax, bx = sorted((coords[0], coords[2]))
        ay, by = sorted((coords[1], coords[3]))

        ax = max(0, min(width, ax))
        bx = max(0, min(width, bx))
        ay = max(0, min(height, ay))
        by = max(0, min(height, by))

        if bx - ax < 24 or by - ay < 24:
            return jsonify(
                success=False,
                message="Please select a larger area."
            ), 400

        # Keep surrounding visual context.
        mx = (bx - ax) * 0.15
        my = (by - ay) * 0.15

        crop = image.crop((
            max(0, int(ax - mx)),
            max(0, int(ay - my)),
            min(width, int(bx + mx)),
            min(height, int(by + my))
        ))

        # Add border space for edge regions.
        border = max(
            12,
            int(min(crop.size) * 0.12)
        )

        crop = ImageOps.expand(
            crop,
            border=border,
            fill=(232, 237, 242)
        )

        # Independent zoomed inference.
        # Original detections remain unchanged.
        with MODEL_LOCK:
            result = model.predict(
                source=crop,
                imgsz=768,
                conf=0.12,
                iou=0.70,
                device=DEVICE,
                verbose=False
            )[0]

        candidates = [
            round(float(box.conf[0]) * 100, 2)
            for box in result.boxes
        ]

        visual = Image.fromarray(
            result.plot()[:, :, ::-1]
        )

        output = BytesIO()

        visual.save(
            output,
            format="JPEG",
            quality=90
        )

        token = save_temporary_result(
            output.getvalue()
        )

        return jsonify(
            success=True,
            candidate_count=len(candidates),
            candidate_confidences=candidates,
            inspect_image=url_for(
                "view_result",
                token=token
            ),
            note=(
                "Zoomed proposals only; "
                "not confirmed ulcers or a diagnosis."
            )
        )

    except (
        ValueError,
        KeyError,
        OSError,
        Image.DecompressionBombError
    ):
        return jsonify(
            success=False,
            message="Invalid image or selected area."
        ), 400

    except Exception:
        app.logger.exception(
            "Region inspection failed"
        )

        return jsonify(
            success=False,
            message="Inspection could not be completed."
        ), 500


if __name__ == "__main__":

    if "--self-test" in sys.argv:

        run_self_test()

    else:

        app.run(
            host="127.0.0.1",
            port=5050,
            debug=False
        )

