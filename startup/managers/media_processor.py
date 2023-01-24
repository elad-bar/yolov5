from __future__ import annotations

import logging
import sys

import torch

from models.common import DetectMultiBackend
from startup.helpers.consts import *
from startup.helpers.enums import ModelFile
from startup.models.model_details import ModelDetails
from utils.dataloaders import LoadImages
from utils.general import Profile, check_img_size, non_max_suppression
from utils.torch_utils import select_device

_LOGGER = logging.getLogger(__name__)


class MediaProcessor:

    def __init__(self, media_path: str, model_info: ModelDetails):
        # Load model
        device = select_device('')

        weights_path = model_info.get_file(ModelFile.WEIGHTS)
        data_path = model_info.get_file(ModelFile.DATA)

        self._model = DetectMultiBackend(weights_path, device=device, dnn=False, data=data_path, fp16=False)

        stride, pt = self._model.stride, self._model.pt

        img_size = check_img_size((640, 640), s=stride)  # check image size

        self.dataset = LoadImages(media_path, img_size=img_size, stride=stride, auto=pt, vid_stride=VIDEO_FRAMES)

        # Run inference
        self._model.warmup(imgsz=(1 if pt or self._model.triton else BS, 3, *img_size))  # warmup

        self._profiles = (Profile(), Profile(), Profile())

    @property
    def labels(self) -> dict[int, str]:
        return self._model.names

    @property
    def last_frame_processing_time(self) -> float:
        value = self._profiles[1].dt * MILLISECONDS

        return value

    def get_predictions(self, im):
        predicates = None

        try:
            with self._profiles[0]:
                im = torch.from_numpy(im).to(self._model.device)
                im = im.half() if self._model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with self._profiles[1]:
                predicates = self._model(im, augment=False, visualize=False)

            # NMS
            with self._profiles[2]:
                predicates = non_max_suppression(predicates,
                                                 CONFIDENCE_THRESHOLD,
                                                 NMS_IOU_THRESHOLD,
                                                 None,
                                                 False,
                                                 max_det=MAX_DETECTION)

        except Exception as ex:
            exc_type, exc_obj, exc_tb = sys.exc_info()

            _LOGGER.error(f"Failed to get predictions ({self._model.pt}), Error: {ex}, Line: {exc_tb.tb_lineno}")

        return predicates
