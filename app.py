"""Flask web application that predicts hand-sign digits from uploaded images."""

import os
import secrets

from flask import Flask, render_template, request
from flask_wtf.csrf import CSRFProtect
from PIL import UnidentifiedImageError

from model import preprocess_img, predict_result

app = Flask(__name__)
# CSRF tokens are signed with SECRET_KEY. Use a fixed key from the environment in
# deployment; fall back to a random per-process key so nothing secret is in the repo.
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
# Reject state-changing requests that lack a valid CSRF token (SonarQube python:S4502).
csrf = CSRFProtect(app)

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
