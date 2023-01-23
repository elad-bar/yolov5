from __future__ import annotations

import uuid
import logging
import os
import sys

from pathlib import Path

from datetime import datetime

import yaml

import train
from startup.helpers.consts import *
from startup.managers.media_processor import MediaProcessor
from startup.models.exceptions import APIException
from startup.models.label_statistics import LabelStatistics
from utils.general import (cv2)
from utils.torch_utils import smart_inference_mode

_LOGGER = logging.getLogger(__name__)


class API:
    _models: dict[str, dict[str, Path]]
    _datasets_path: str
    _temp_path: str

    def __init__(self, datasets_path: str, temp_path: str):
        self._models = {}
        self._datasets_path = datasets_path
        self._temp_path = temp_path

    @property
    def models(self):
        return self._models

    def initialize(self):
        _LOGGER.debug("Initializing API")
        path = Path(self._datasets_path)

        self._models = {}

        _LOGGER.debug(f"Datasets directory: {path}")

        for item in path.glob("*"):
            _LOGGER.debug(f"Processing directory: {item}")

            try:

                if item.is_dir():
                    model_dir = str(item)
                    model_name = model_dir.replace(self._datasets_path, "")[1:]

                    model_path = Path(model_dir)

                    model_weights_path = self._get_model_path(model_path, MODEL_WEIGHTS_PATH_PATTERN, True)
                    model_data_path = self._get_model_path(model_path, f"{model_name}.yaml")
                    model_pre_train_path = self._get_model_path(model_path, MODEL_PRE_TRAIN_KEY)

                    self._models[model_name] = {
                        MODEL_WEIGHTS_KEY: model_weights_path,
                        MODEL_DATA_KEY: model_data_path,
                        MODEL_PRE_TRAIN_KEY: model_pre_train_path,
                        ATTR_OK: model_weights_path is not None
                    }

                    _LOGGER.debug(f"Model details: {self._models[model_name]}")

            except Exception as ex:
                exc_type, exc_obj, exc_tb = sys.exc_info()

                _LOGGER.error(f"Failed to load model directory {str(item)}, Error: {ex}, Line: {exc_tb.tb_lineno}")

        _LOGGER.info(f"Loaded models: {self._models}")

    @staticmethod
    def _get_model_path(model_path, pattern, optional: bool = False) -> Path | None:
        file_list = list(model_path.glob(pattern))
        file = None

        if len(file_list) == 0:
            if not optional:
                raise Exception(f"'{pattern}' was not found'")
        else:
            file = file_list[0]

        return file

    def create(self, model_name, labels: list[str]):
        model_info = self._models.get(model_name)

        if model_info is not None:
            raise APIException(400, f"Model {model_name} already exists")

        try:
            model_path = os.path.join(self._datasets_path, model_name)

            objects_created = []
            start_processing = datetime.now().timestamp()

            for directory in MODEL_BASE_DIRECTORIES:
                directory_path = os.path.join(model_path, *directory.split("/"))

                os.makedirs(directory_path, exist_ok=True)

                objects_created.append(directory_path)

            data = {
                "path": None,
                "train": os.path.join(model_path, "images", "train"),
                "val": os.path.join(model_path, "images", "val"),
                "test": None,
                "nc": labels,
                "names": len(labels)
            }

            data_yaml_path = os.path.join(model_path, f"{model_name}.yaml")

            with open(data_yaml_path, 'a+') as file:
                content = yaml.dump(data).replace("null", "")

                file.write(content)

                objects_created.append(data_yaml_path)

            stop_processing = datetime.now().timestamp()

            request_duration = (stop_processing - start_processing) * MILLISECONDS

            result = {
                ATTR_OK: True,
                ATTR_CREATED: objects_created,
                ATTR_DURATION: request_duration
            }

            self.initialize()

        except Exception as ex:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"Failed to create model {model_name}"

            raise APIException(500, error_message, ex, exc_tb.tb_lineno)

        return result

    def upload(self, model_name, media_file):
        media_path = None

        try:
            original_file_name = media_file.filename

            media_path = self._save_file_to_disk(media_file)

            _LOGGER.info(f"Uploading to model {model_name}")

            result = self._upload(model_name, media_path, original_file_name)

            self._remove_file_to_disk(media_path)

            return result

        except APIException as api_ex:
            self._remove_file_to_disk(media_path)

            raise api_ex

        except Exception as ex:
            self._remove_file_to_disk(media_path)

            exc_type, exc_obj, exc_tb = sys.exc_info()

            error_message = f"Failed to upload media to model {model_name}"

            raise APIException(500, error_message, ex, exc_tb.tb_lineno)

    def train(self, model_name):
        try:
            _LOGGER.info(f"Training model {model_name}")

            model_info = self._models.get(model_name)
            model_weights_path = model_info[MODEL_WEIGHTS_KEY]
            model_data_path = model_info[MODEL_DATA_KEY]

            project_path = os.path.join(self._datasets_path, model_name)

            if model_info is None:
                raise APIException(404, f"Model {model_name} not found, Path: {model_weights_path}")

            result = train.run(name="models",
                               project=project_path,
                               data=model_data_path,
                               weights=model_weights_path,
                               epochs=1200,
                               exist_ok=True)

            return result

        except APIException as api_ex:
            raise api_ex

        except Exception as ex:
            exc_type, exc_obj, exc_tb = sys.exc_info()

            error_message = f"Failed to train model {model_name}"

            raise APIException(500, error_message, ex, exc_tb.tb_lineno)

    def detect(self, model_name, media_file, expected_keys: list[str], compare_by_confidence: bool):
        media_path = None

        try:
            media_path = self._save_file_to_disk(media_file)

            _LOGGER.info(f"Processing model {model_name}")

            result = self._process(model_name, media_path, expected_keys, compare_by_confidence)

            self._remove_file_to_disk(media_path)

            return result

        except APIException as api_ex:
            self._remove_file_to_disk(media_path)

            raise api_ex

        except Exception as ex:
            self._remove_file_to_disk(media_path)

            exc_type, exc_obj, exc_tb = sys.exc_info()

            error_message = f"Failed to detect model {model_name}"

            raise APIException(500, error_message, ex, exc_tb.tb_lineno)

    @smart_inference_mode()
    def _process(self, model_name: str, media_path: str, expected_keys: list[str], compare_by_confidence: bool):
        try:
            start_processing = datetime.now().timestamp()

            model_info = self._models.get(model_name)
            model_weights_path = model_info[MODEL_WEIGHTS_KEY]
            model_data_path = model_info[MODEL_DATA_KEY]
            is_ready = model_info[ATTR_OK]

            if model_info is None:
                raise APIException(404, f"Model {model_name} not found, Path: {model_weights_path}")

            elif not is_ready:
                raise APIException(400, f"Model {model_name} is not trained")

            processor = MediaProcessor(media_path, str(model_weights_path), str(model_data_path))

            media_details = {}

            labels = LabelStatistics()

            for path, im, im0s, vid_cap, s in processor.dataset:
                media_details = self._get_media_details(vid_cap)
                current_timestamp = media_details.get(ATTR_TIMESTAMP, 0)

                media_type = media_details.get(ATTR_TYPE)

                labels.set_media_type(media_type)

                predicates = processor.get_predictions(im)

                # Process predictions
                for i, det in enumerate(predicates):  # per image
                    if len(det):
                        for *xyxy, conf, cls in reversed(det):
                            label = processor.labels[int(cls)]

                            labels.add(label, float(conf), current_timestamp)

                # TODO: Report to Prometheus
                if media_type == MEDIA_VIDEO:
                    _LOGGER.debug(
                        f"Processed video frame of {(current_timestamp / 1000):.3f}s, "
                        f"Duration: {processor.last_frame_processing_time:.3f}ms"
                    )
                else:
                    _LOGGER.debug(
                        f"Processed image, "
                        f"Duration: {processor.last_frame_processing_time:.3f}ms"
                    )

            stop_processing = datetime.now().timestamp()

            request_duration = (stop_processing - start_processing) * MILLISECONDS

            optimized_labels = labels.get_highest_confidence() if compare_by_confidence else labels.get_first()

            if len(expected_keys) == 0:
                expected_keys = processor.labels.values()

            status = optimized_labels.keys_status(expected_keys).name.capitalize()

            if ATTR_TIMESTAMP in media_details:
                del media_details[ATTR_TIMESTAMP]

            result = {
                ATTR_OK: True,
                ATTR_MEDIA: media_details,
                ATTR_LABELS: optimized_labels.to_dict(),
                ATTR_STATUS: status,
                ATTR_DURATION: request_duration
            }

            _LOGGER.debug(result)

        except Exception as ex:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"Failed to process model {model_name}"

            raise APIException(500, error_message, ex, exc_tb.tb_lineno)

        return result

    def _upload(self, model_name: str, media_path: str, original_file_name: str):
        try:
            start_processing = datetime.now().timestamp()

            model_info = self._models.get(model_name)
            model_pre_train_path = model_info[MODEL_PRE_TRAIN_KEY]

            if not model_pre_train_path.is_dir():
                raise APIException(404, f"Model {model_name} not found, Path: {model_pre_train_path}")

            vid_cap = cv2.VideoCapture(media_path)

            new_file_prefix = os.path.join(model_pre_train_path, original_file_name)

            # frame
            currentframe = 1

            while True:
                # reading from frame
                ret, frame = vid_cap.read()

                if not ret:
                    break

                name = f"{new_file_prefix}.{currentframe}.jpg"
                _LOGGER.info(f"Extracting frame #{currentframe} to ${name}")

                # writing the extracted images
                cv2.imwrite(name, frame)

                # increasing counter so that it will
                # show how many frames are created
                currentframe += 1

            # Release all space and windows once done
            vid_cap.release()
            cv2.destroyAllWindows()

            stop_processing = datetime.now().timestamp()

            request_duration = (stop_processing - start_processing) * MILLISECONDS

            status = (
                f"Extracted {currentframe - 1} image{'s' if currentframe > 2 else ''}, "
                f"Path: {new_file_prefix}.*.jpg"
            )

            result = {
                ATTR_OK: True,
                ATTR_STATUS: status,
                ATTR_DURATION: request_duration
            }

            _LOGGER.debug(result)

        except Exception as ex:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"Failed to pre-train images for model {model_name}"

            raise APIException(500, error_message, ex, exc_tb.tb_lineno)

        return result

    @staticmethod
    def _get_media_details(video_capture) -> dict:
        file_type = MEDIA_IMAGE if video_capture is None else MEDIA_VIDEO

        result = {
            ATTR_TYPE: file_type
        }

        if file_type == MEDIA_VIDEO:
            video_fps = video_capture.get(cv2.CAP_PROP_FPS)
            video_total_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
            current_timestamp = video_capture.get(cv2.CAP_PROP_POS_MSEC)

            video_duration = video_total_frames / video_fps

            result[ATTR_DURATION] = video_duration
            result[ATTR_TIMESTAMP] = current_timestamp

        return result

    def _save_file_to_disk(self, file_stream):
        if file_stream is None:
            raise APIException(400, "Media file is missing")

        file_name = file_stream.filename
        file_name_parts = file_name.split(".")
        file_ext = file_name_parts[-1]

        source = os.path.join(self._temp_path, f".{uuid.uuid4()}.{file_ext}")
        file_stream.save(source)

        return source

    @staticmethod
    def _remove_file_to_disk(file_path):
        if file_path is not None and os.path.exists(file_path):
            os.remove(file_path)
