import logging

from startup.helpers.consts import *
from startup.models.label_details import LabelDetails
from startup.models.label_statistics_result import LabelStatisticsResult

_LOGGER = logging.getLogger(__name__)


class LabelStatistics:
    labels: dict[str, list[LabelDetails]]

    def __init__(self):
        self.labels = {}
        self._media_type = MEDIA_IMAGE

    def set_media_type(self, media_type):
        self._media_type = media_type

    def add(self, label: str, conf: float, timestamp: float):
        message = f"'{label}' was detected ({conf:.3%})"

        if self._media_type == MEDIA_VIDEO:
            _LOGGER.info(f"{message} at {(timestamp / MILLISECONDS):.2f}s")
        else:
            _LOGGER.info(message)

        if label not in self.labels:
            self.labels[label] = []

        label_details = LabelDetails(conf, timestamp)

        self.labels[label].append(label_details)

    def get_highest_confidence(self) -> LabelStatisticsResult:
        result = LabelStatisticsResult(self._media_type)

        for label in self.labels:
            label_items = self.labels[label]

            for label_item in label_items:
                if not result.has_label(label) or label_item.confidence > result.get_confidence(label):
                    result.set(label, label_item)

        return result

    def get_first(self) -> LabelStatisticsResult:
        result = LabelStatisticsResult(self._media_type)

        for label in self.labels:
            label_items = self.labels[label]

            for label_item in label_items:
                if not result.has_label(label):
                    result.set(label, label_item)
                    break

        return result
