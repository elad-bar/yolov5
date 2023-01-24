import os

from startup.helpers.consts import *
from startup.helpers.enums import *


class ModelDetails:
    name: str
    status: ModelStatus

    _model_dir: str
    _path: dict[ModelFile, str]
    _path_expectation: dict[ModelFile, bool]

    def __init__(self, name, datasets_path: str | None = DEFAULT_DATASETS_PATH):
        self.name = name
        self.status = ModelStatus.INVALID

        self._model_dir = os.path.join(datasets_path, self.name)

        self._path = {}

        self._path_expectation = {
            ModelFile.DATA: True,
            ModelFile.PRE_TRAIN: True,
            ModelFile.WEIGHTS: False,
            ModelFile.PROJECT: True}

        self._set_initial_status()

    @property
    def is_valid(self):
        return self.status != ModelStatus.INVALID

    @property
    def is_training(self):
        return self.status == ModelStatus.TRAINING

    def get_file(self, file: ModelFile) -> str | None:
        return self._path.get(file)

    def _set_initial_status(self):
        self._add_path_item(ModelFile.PROJECT, "")
        self._add_path_item(ModelFile.DATA, f"{self.name}.yaml")
        self._add_path_item(ModelFile.PRE_TRAIN, MODEL_PRE_TRAIN_KEY)
        self._add_path_item(ModelFile.WEIGHTS, MODEL_WEIGHTS_PATH_PATTERN)

        self.restore_status()

    def _add_path_item(self, file: ModelFile, path: str):
        path = os.path.join(self._model_dir, *path.split("/"))

        self._path[file] = path

    def set_status(self, status: ModelStatus):
        self.status = status

    def restore_status(self):
        mandatory_availability = [self.is_exists(file) for file in self._path if self._path_expectation[file]]
        optional_availability = [self.is_exists(file) for file in self._path if not self._path_expectation[file]]

        status = ModelStatus.READY

        if False in optional_availability:
            status = ModelStatus.NOT_READY

        if False in mandatory_availability:
            status = ModelStatus.INVALID

        self.status = status

    def as_dict(self):
        item = {
            ATTR_NAME: self.name,
            ATTR_STATUS: self.status.name.lower(),
            ATTR_PATH: {path.name: self._path[path]
                        for path in self._path}}

        return item

    def is_exists(self, file: ModelFile) -> bool:
        path = self.get_file(file)

        return os.path.exists(path)
