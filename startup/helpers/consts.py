LABEL_LOADING = "loading"
LABEL_LOADED = "loaded"

DETECTION_URL = "/v1/detect"
TRAIN_URL = "/v1/train"

DEFAULT_DATASETS_PATH = "/usr/src/datasets"
DEFAULT_TEMP_PATH = "/tmp"

MODEL_WEIGHTS_PATH_PATTERN = '**/last.pt'
MODEL_DATA_PATH_PATTERN = '*.yaml'

MODEL_WEIGHTS_KEY = "weights"
MODEL_DATA_KEY = "data"

MEDIA_VIDEO = "video"
MEDIA_IMAGE = "image"

MILLISECONDS = 1E3

VIDEO_FRAMES = 10
MAX_FRAME_PER_SECOND = 2

BS = 1

CONFIDENCE_THRESHOLD = 0.40  # confidence threshold
NMS_IOU_THRESHOLD = 0.45  # NMS IOU threshold
MAX_DETECTION = 1000  # maximum detections per image

STATUS_MAPPING = {
    "True_True": "full",
    "True_False": "loading_only",
    "False_True": "loaded_only",
    "False_False": "none",
}

ATTR_CONFIDENCE = "confidence"
ATTR_TIMESTAMP = "timestamp"
ATTR_DURATION = "duration"
ATTR_TYPE = "type"
ATTR_OK = "ok"
ATTR_MEDIA = "media"
ATTR_LABELS = "labels"
ATTR_STATUS = "status"
