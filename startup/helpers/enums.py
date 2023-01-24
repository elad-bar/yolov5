from enum import Enum


class StatusMapping(Enum):
    NONE = -1,
    PARTIAL = 0,
    FULL = 1


class ModelStatus(Enum):
    INVALID = -1,
    NOT_READY = 0,
    READY = 1,
    TRAINING = 2


class ModelFile(Enum):
    DATA = 0,
    PRE_TRAIN = 1,
    WEIGHTS = 2,
    PROJECT = 3
