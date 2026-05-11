import json

from .base import BaseExtractor


class EncodingsExtractor(BaseExtractor):
    def __init__(self, source_path):
        super().__init__(source_path)
        with open(source_path, "r", encoding="utf-8") as handle:
            self.encodings = json.load(handle)

    def collect_encodings(self):
        return self.encodings, [], []
