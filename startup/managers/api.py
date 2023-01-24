from __future__ import annotations

import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from threading import Thread

import yaml

import train
from startup.helpers.consts import *
from startup.helpers.enums import ModelFile, ModelStatus
from startup.managers.media_processor import MediaProcessor
from startup.models.exceptions import APIException
from startup.models.label_statistics import LabelStatistics
from startup.models.model_details import ModelDetails
from utils.general import cv2
from utils.torch_utils import smart_inference_mode

_LOGGER = logging.getLogger(__name__)


class API:
    _models: list[ModelDetails]
    _datasets_path: str
    _temp_path: str

    def __init__(self, datasets_path: str, temp_path: str):
        self._models = []
        self._datasets_path = datasets_path
        self._temp_path = temp_path

    def initialize(self):
        _LOGGER.debug("Initializing API")
        path = Path(self._datasets_path)

        self._models.clear()

        _LOGGER.debug(f"Datasets directory: {path}")

        for item in path.glob("*"):
            _LOGGER.debug(f"Processing directory: {item}")

            try:

                if item.is_dir():
                    model_dir = str(item)
                    model_name: str = model_dir.replace(self._datasets_path, "")[1:]

                    model = ModelDetails(model_name, self._datasets_path)

                    self._models.append(model)

            except Exception as ex:
                exc_type, exc_obj, exc_tb = sys.exc_info()

                _LOGGER.error(f"Failed to load model directory {str(item)}, Error: {ex}, Line: {exc_tb.tb_lineno}")

        _LOGGER.info(f"Loaded models: {self.get_models()}")

    @staticmethod
    def _get_model_path(model_path: str, pattern: str, optional: bool = False) -> str | None:
        path = os.path.join(model_path, *pattern.split("/"))

        file_list = list(Path(model_path).glob(pattern))

        if not optional and len(file_list) == 0:
            raise Exception(f"'{pattern}' was not found'")

        return path

    def create(self, model_name, labels: list[str]):
        model_info = self.get_model(model_name)

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
                "names": len(labels)}

            data_yaml_path = os.path.join(model_path, f"{model_name}.yaml")

            with open(data_yaml_path, 'a+') as file:
                content = yaml.dump(data).replace("null", "")

                file.write(content)

                objects_created.append(data_yaml_path)

            stop_processing = datetime.now().timestamp()

            request_duration = (stop_processing - start_processing) * MILLISECONDS

            result = {ATTR_OK: True, ATTR_CREATED: objects_created, ATTR_DURATION: request_duration}

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

    def get_models(self):
        models = []

        for model in self._models:
            models.append(model.as_dict())

        return models

    def get_model(self, model_name: str) -> ModelDetails | None:
        results = [model for model in self._models if model.name == model_name]

        result = None if len(results) == 0 else results[0]

        return result

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

    def train(self, model_name):
        model_info = self.get_model(model_name)

        try:
            if model_info is None:
                raise APIException(404, f"Model {model_name} not found")

            if model_info.is_training:
                return {"ok": False, "status": model_info.status.name.lower()}

            model_info.set_status(ModelStatus.TRAINING)

            _LOGGER.info(f"Training model {model_name}")

            thread = Thread(target=self._train, args=(model_name,))
            thread.start()

            return {"ok": True, "status": model_info.status.name.lower()}

        except APIException as api_ex:
            if model_info is not None:
                model_info.restore_status()

            raise api_ex

        except Exception as ex:
            if model_info is not None:
                model_info.restore_status()

            exc_type, exc_obj, exc_tb = sys.exc_info()

            error_message = f"Failed to train model {model_name}"

            raise APIException(500, error_message, ex, exc_tb.tb_lineno)

    def _train(self, model_name):
        model_info = self.get_model(model_name)

        try:
            train.run(name="models",
                      project=model_info.get_file(ModelFile.PROJECT),
                      data=model_info.get_file(ModelFile.DATA),
                      epochs=1200,
                      patience=0,
                      exist_ok=True)

            _LOGGER.info(f"Training model {model_name} is done")

        except Exception as ex:
            exc_type, exc_obj, exc_tb = sys.exc_info()

            _LOGGER.error(f"Failed to train model {model_name}, Error: {ex}, Line: {exc_tb.tb_lineno}")

        model_info.restore_status()

    @smart_inference_mode()
    def _process(self, model_name: str, media_path: str, expected_keys: list[str], compare_by_confidence: bool):
        try:
            start_processing = datetime.now().timestamp()

            model_info = self.get_model(model_name)

            if model_info is None:
                raise APIException(404, f"Model {model_name} not found")

            elif model_info.status != ModelStatus.READY:
                raise APIException(400, f"Model {model_name} cannot process request, Status: {model_info.status}")

            processor = MediaProcessor(media_path, model_info)

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
                    _LOGGER.debug(f"Processed video frame of {(current_timestamp / 1000):.3f}s, "
                                  f"Duration: {processor.last_frame_processing_time:.3f}ms")
                else:
                    _LOGGER.debug(f"Processed image, "
                                  f"Duration: {processor.last_frame_processing_time:.3f}ms")

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
                ATTR_DURATION: request_duration}

            _LOGGER.debug(result)

        except Exception as ex:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"Failed to process model {model_name}"

            raise APIException(500, error_message, ex, exc_tb.tb_lineno)

        return result

    def _upload(self, model_name: str, media_path: str, original_file_name: str):
        try:
            start_processing = datetime.now().timestamp()

            model_info = self.get_model(model_name)

            if model_info is None:
                raise APIException(404, f"Model {model_name} not found")

            elif not model_info.is_valid:
                raise APIException(400, f"Model {model_name} cannot accept images")

            vid_cap = cv2.VideoCapture(media_path)

            model_pre_train_path = model_info.get_file(ModelFile.PRE_TRAIN)
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

            status = (f"Extracted {currentframe - 1} image{'s' if currentframe > 2 else ''}, "
                      f"Path: {new_file_prefix}.*.jpg")

            result = {ATTR_OK: True, ATTR_STATUS: status, ATTR_DURATION: request_duration}

            _LOGGER.debug(result)

        except Exception as ex:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"Failed to pre-train images for model {model_name}"

            raise APIException(500, error_message, ex, exc_tb.tb_lineno)

        return result

    @staticmethod
    def _get_media_details(video_capture) -> dict:
        file_type = MEDIA_IMAGE if video_capture is None else MEDIA_VIDEO

        result = {ATTR_TYPE: file_type}

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
