from nerfstudio.fields.base_field import Field
from .base import BaseFieldExporter
from .density_fields import HashMLPDensityFieldExporter
from .nerfacto_field import NerfactoFieldExporter


FIELD_EXPORTER_MAPPING = {
    "HashMLPDensityField": HashMLPDensityFieldExporter,
    "NerfactoField": NerfactoFieldExporter,
}

def get_field_exporter(field: Field) -> BaseFieldExporter:
    exporter = FIELD_EXPORTER_MAPPING[field.__class__.__name__](field)
    return exporter


