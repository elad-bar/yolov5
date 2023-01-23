# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run a Flask REST API exposing one or more YOLOv5s models
"""

import argparse
import io
import logging
import os
import sys

import torch
from flask import Flask, request, abort, jsonify
from PIL import Image
from pathlib import Path

app = Flask(__name__)
models = {}

DETECTION_URL = "/v1/detect"
TRAIN_URL = "/v1/train"

DEFAULT_DATASETS_PATH = "/usr/src/datasets"

DEBUG = str(os.environ.get('DEBUG', False)).lower() == str(True).lower()

log_level = logging.DEBUG  # if DEBUG else logging.INFO

root = logging.getLogger()
root.setLevel(log_level)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(log_level)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')
stream_handler.setFormatter(formatter)
root.addHandler(stream_handler)

_LOGGER = logging.getLogger(__name__)


class API:
    _datasets_path: str

    def __init__(self, datasets_path: str):
        self.models = {}
        self._datasets_path = datasets_path

        self.initialize()

    def initialize(self):
        path = Path(self._datasets_path)

        model_files = list(path.glob('**/last.pt'))

        for model_file in model_files:
            file_name = str(model_file).replace(self._datasets_path, "")
            file_parts = file_name.split("/")
            model_name = file_parts[1]

            self.models[model_name] = str(model_file)

        _LOGGER.info(f"Loaded models: {self.models}")

    def detect(self, model, media_file, size):
        model_path = self.models.get(model)

        model_file = Path(model_path)

        media_file_provided = media_file is not None

        if not model_file.is_file():
            _LOGGER.error(f"Model {model} not found, Path: {model_path}")

            abort(404, "Model not found")

        elif not media_file_provided:
            _LOGGER.error("Media file is missing")

            abort(400, "Media file is missing")

        else:
            try:
                _LOGGER.info(f"Processing model {model}")

                im_bytes = media_file.read()
                im = Image.open(io.BytesIO(im_bytes))

                model_initializer = torch.load(model_path)
                results = model_initializer(im, size=size)

                result = results.pandas().xyxy[0].to_json(orient="records")

                return result
            except Exception as ex:
                exc_type, exc_obj, exc_tb = sys.exc_info()

                _LOGGER.error(f"Failed to process model {model} [{model_path}], error: {ex}, Line: {exc_tb.tb_lineno}")

                abort(500, f"Failed to process model {model}")


@app.route(f"{DETECTION_URL}/list", methods=["GET"])
def list_models():
    return api.models.keys()


@app.route(f"{DETECTION_URL}/<model>", methods=["POST"])
def detect(model):
    media_file = request.files.get("image")
    size = request.form.get("size", 640)

    api.detect(model, media_file, size)


if __name__ == "__main__":
    _env_datasets_path = os.environ.get("DATASETS_PATH", DEFAULT_DATASETS_PATH)

    api = API(_env_datasets_path)

    app.run(host="0.0.0.0", port=50000)  # debug=True causes Restarting with stat
