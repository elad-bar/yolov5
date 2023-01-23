
class LabelDetails:
    confidence: float
    timestamp: float

    def __init__(self, conf: float, timestamp: float):
        self.confidence = conf
        self.timestamp = timestamp
