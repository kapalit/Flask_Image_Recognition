# Hand Digit Image Recognition

[![CI](https://github.com/kapalit/Flask_Image_Recognition/actions/workflows/ci.yml/badge.svg)](https://github.com/kapalit/Flask_Image_Recognition/actions/workflows/ci.yml)

This web app is made using Flask framework and is all about predicting the hand sign digit images based on the deep learning model trained using transfer learning.

![Python](https://img.shields.io/badge/python-8338ec)
![Keras](https://img.shields.io/badge/keras-219ebc)
![Numpy](https://img.shields.io/badge/numpy-ffafcc)
![Pillow](https://img.shields.io/badge/pillow-00bbf9)
![Flask](https://img.shields.io/badge/flask-f72585)
![Bootstrap](https://img.shields.io/badge/bootstrap-57cc99)

# Glimpse

https://user-images.githubusercontent.com/72191416/201943098-c8f5fd8b-ec7d-4e5d-883d-8b69109b946f.mp4

# Dependencies

- Flask
- TensorFlow (Keras)
- Numpy
- Pillow (PIL)

## UI Framework

- Bootstrap

# Get Started

Requires **Python 3.11 (64-bit)** and Git. The pinned versions in `requirements.txt`
(TensorFlow/Keras 2.15) support Python 3.9-3.11.

```commandline
git clone https://github.com/kapalit/Flask_Image_Recognition.git
cd Flask_Image_Recognition
py -3.11 -m venv venv
venv\Scriptsctivate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

`requirements.txt` holds the runtime dependencies; `requirements-dev.txt` adds
Pylint, pre-commit and pytest-cov for development.

# Run

```commandline
python -m flask --app app run
```

Open http://127.0.0.1:5000, choose an image from the `test_images` folder, check the
preview and click **Submit** to see the predicted digit.

# Code Quality

```commandline
pylint app.py model.py conftest.py test_*.py
pre-commit install          # once per clone: run hooks on every commit
pre-commit run --all-files  # run every hook manually
```

Pylint settings live in `.pylintrc` (fails below a 9.0 score). Pre-commit runs
whitespace/end-of-file/YAML/large-file checks and Pylint (`.pre-commit-config.yaml`).

# Tests

```commandline
pytest
```

- `test_routes.py` - Flask routes with the model mocked (valid, missing and invalid uploads)
- `test_preprocessing.py` - image preprocessing and prediction functions
- `test_model.py`, `test_integration_*.py`, `test_acceptance_*.py` - earlier
  model, integration and acceptance tests

# Continuous Integration

`.github/workflows/ci.yml` runs on every push and pull request to `main`: it installs
the dependencies on Python 3.11 and runs Pylint followed by pytest.
