import abc


class BaseExporter(abc.ABC):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    @abc.abstractmethod
    def export(self, output_path):
        raise NotImplementedError
