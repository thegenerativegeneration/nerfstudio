from nerfstudio.utils.eval_utils import eval_setup
from pathlib import Path
from .nerfacto import NerfactoExporter
from .base import BaseExporter


MODEL_EXPORTER_MAPPING = {
    "NerfactoModel": NerfactoExporter
}



def get_exporter(config_path: Path) -> BaseExporter:
    test_mode = "inference"
    eval_num_rays_per_chunk = 1024
    _, pipeline , _, _ = eval_setup(
        config_path, eval_num_rays_per_chunk, test_mode=test_mode)
    
    exporter = MODEL_EXPORTER_MAPPING[pipeline.model.__class__.__name__](pipeline)
    return exporter