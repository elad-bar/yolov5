from startup.helpers.consts import *
from startup.helpers.enums import StatusMapping
from startup.models.label_details import LabelDetails


class LabelStatisticsResult:
    _media_type: str
    labels: dict[str, LabelDetails]

    def __init__(self, media_type):
        self.labels = {}
        self._media_type = media_type

    def set(self, label: str, details: LabelDetails):
        self.labels[label] = details

    def has_label(self, label: str) -> bool:
        return label in self.labels

    def get_confidence(self, label: str) -> float:
        label_details = self.labels.get(label, LabelDetails(0, 0))

        return label_details.confidence

    def to_dict(self) -> dict:
        result = {}

        for label in self.labels:
            data_item = self.labels[label]

            result[label] = {
                ATTR_CONFIDENCE: data_item.confidence
            }

            if self._media_type == MEDIA_VIDEO:
                result[label][ATTR_TIMESTAMP] = data_item.timestamp

        return result

    def keys_status(self, keys: list[str]) -> StatusMapping:
        result = StatusMapping.NONE
        found = []

        for key in keys:
            if key in self.labels:
                found.append(key)

                result = StatusMapping.PARTIAL

        if len(found) == len(keys):
            result = StatusMapping.FULL

        return result
