"""Flask web application that predicts hand-sign digits from uploaded images."""

import os

from flask import Flask, render_template, request
from PIL import UnidentifiedImageError

from model import preprocess_img, predict_result

app = Flask(__name__)

ERROR_MESSAGE = "File cannot be processed."


@app.route("/")
def main():
    """Render the home page containing the image upload form."""
    return render_template("index.html")


@app.route("/prediction", methods=["POST"])
def predict_image_file():
    """Preprocess the uploaded image, run the model and render the predicted digit.

    Any request without a readable image (missing "file" field, empty upload,
    non-image or corrupt file) renders the result page with an error message.
    """
    try:
        img = preprocess_img(request.files["file"].stream)
        pred = predict_result(img)
    except (KeyError, UnidentifiedImageError, OSError, ValueError):
        return render_template("result.html", err=ERROR_MESSAGE)

    return render_template("result.html", predictions=str(pred))


def debug_enabled():
    """Return True only when FLASK_DEBUG is explicitly set to "1" or "true".

    The Werkzeug debugger allows arbitrary code execution, so it must never be
    switched on by default (SonarQube rule python:S4507).
    """
    return os.environ.get("FLASK_DEBUG", "").strip().lower() in ("1", "true")


if __name__ == "__main__":
    app.run(port=9000, debug=debug_enabled())
