import abc



class BaseFieldExporter(abc.ABC):
    def __init__(self, field):
        self.field = field

    @abc.abstractmethod
    def export(self, output_prefix):
        raise NotImplementedError
    
