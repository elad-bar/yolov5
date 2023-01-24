from __future__ import annotations

import logging
import os
import sys

from flask import Flask, abort, request, url_for

from startup.helpers.consts import *
from startup.managers.api import API
from startup.models.exceptions import APIException

app = Flask(__name__)

DEBUG = str(os.environ.get('DEBUG', False)).lower() == str(True).lower()

log_level = logging.DEBUG if DEBUG else logging.INFO

root = logging.getLogger()
root.setLevel(log_level)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(log_level)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')
stream_handler.setFormatter(formatter)
root.addHandler(stream_handler)

_LOGGER = logging.getLogger(__name__)


@app.route(f"{MODEL_URL}/<model>", methods=["POST"])
def create_model(model):
    try:
        labels_str = request.form.get("labels")
        labels = [LABEL_LOADING, LABEL_LOADED] if not labels_str else labels_str.split(",")

        result = api.create(model, labels)

        return result

    except APIException as api_ex:
        error_message = api_ex.error
        if api_ex.inner_exception is not None:
            error_message = f"{error_message}, Error: {api_ex.inner_exception}, Line: {api_ex.line}"

        _LOGGER.error(error_message)

        abort(api_ex.status, api_ex.error)

    except Exception as ex:
        exc_type, exc_obj, exc_tb = sys.exc_info()

        _LOGGER.error(f"Failed to process model {model}, error: {ex}, Line: {exc_tb.tb_lineno}")

        abort(500, f"Failed to handle request for model {model}")


@app.route(f"{MODEL_URL}", methods=["GET"])
def list_models():
    return list(api.models.keys())


@app.route(f"{MODEL_URL}/<model>/detect", methods=["POST"])
def detect(model):
    try:
        media_file = request.files.get("image")
        expected_keys_str = request.form.get("expectedKeys")
        compare_str = request.form.get("compare", "")

        compare_by_confidence = compare_str == ATTR_CONFIDENCE

        expected_keys = [] if not expected_keys_str else expected_keys_str.split(",")

        result = api.detect(model, media_file, expected_keys, compare_by_confidence)

        return result

    except APIException as api_ex:
        error_message = api_ex.error
        if api_ex.inner_exception is not None:
            error_message = f"{error_message}, Error: {api_ex.inner_exception}, Line: {api_ex.line}"

        _LOGGER.error(error_message)

        abort(api_ex.status, api_ex.error)

    except Exception as ex:
        exc_type, exc_obj, exc_tb = sys.exc_info()

        _LOGGER.error(f"Failed to process model {model}, error: {ex}, Line: {exc_tb.tb_lineno}")

        abort(500, f"Failed to handle request for model {model}")


@app.route(f"{MODEL_URL}/<model>/train", methods=["GET"])
def get_train_status(model):
    try:
        result = {ATTR_STATUS: api.get_training_status(model)}

        return result

    except APIException as api_ex:
        error_message = api_ex.error
        if api_ex.inner_exception is not None:
            error_message = f"{error_message}, Error: {api_ex.inner_exception}, Line: {api_ex.line}"

        _LOGGER.error(error_message)

        abort(api_ex.status, api_ex.error)

    except Exception as ex:
        exc_type, exc_obj, exc_tb = sys.exc_info()

        _LOGGER.error(f"Failed to get training status for model {model}, error: {ex}, Line: {exc_tb.tb_lineno}")

        abort(500, f"Failed to handle request for model {model}")


@app.route(f"{MODEL_URL}/<model>/train", methods=["POST"])
def train(model):
    try:
        result = api.train(model)

        return result

    except APIException as api_ex:
        error_message = api_ex.error
        if api_ex.inner_exception is not None:
            error_message = f"{error_message}, Error: {api_ex.inner_exception}, Line: {api_ex.line}"

        _LOGGER.error(error_message)

        abort(api_ex.status, api_ex.error)

    except Exception as ex:
        exc_type, exc_obj, exc_tb = sys.exc_info()

        _LOGGER.error(f"Failed to train model {model}, error: {ex}, Line: {exc_tb.tb_lineno}")

        abort(500, f"Failed to handle request for model {model}")


@app.route(f"{MODEL_URL}/<model>/pre-train/images", methods=["POST"])
def upload(model):
    try:
        media_file = request.files.get("image")

        result = api.upload(model, media_file)

        return result

    except APIException as api_ex:
        error_message = api_ex.error
        if api_ex.inner_exception is not None:
            error_message = f"{error_message}, Error: {api_ex.inner_exception}, Line: {api_ex.line}"

        _LOGGER.error(error_message)

        abort(api_ex.status, api_ex.error)

    except Exception as ex:
        exc_type, exc_obj, exc_tb = sys.exc_info()

        _LOGGER.error(f"Failed to process model {model}, error: {ex}, Line: {exc_tb.tb_lineno}")

        abort(500, f"Failed to handle request for model {model}")


if __name__ == "__main__":
    _env_datasets_path = os.environ.get("DATASETS_PATH", DEFAULT_DATASETS_PATH)
    _env_tmp_path = os.environ.get("TEMP_PATH", DEFAULT_TEMP_PATH)

    api = API(_env_datasets_path, _env_tmp_path)

    api.initialize()

    links = []
    for rule in app.url_map.iter_rules():
        methods = []
        for method in rule.methods:
            if method not in ["OPTIONS", "HEAD"]:
                methods.append(method)

        _LOGGER.info(f"{methods[0]} {rule.rule}")

    app.run(host="0.0.0.0", port=50000)
