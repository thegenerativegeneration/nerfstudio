import abc
import os
from pathlib import Path
from .base import BaseExporter

from nerfstudio.exporter.onnx.fields import get_field_exporter


class NerfactoExporter(BaseExporter):

    def export(self, output_path: Path):
        field = self.pipeline._model.field
        proposal_networks = self.pipeline._model.proposal_networks

        field_base_mlp_output_path = os.path.join(output_path, "field")
        proposal_networks_mlp_output_path = os.path.join(output_path, "_proposal_networks_{}_field")

        field_exporter = get_field_exporter(field)
        field_exporter.export(field_base_mlp_output_path)


        for i, proposal_network in enumerate(proposal_networks):

            proposal_network_exporter = get_field_exporter(proposal_network)
            proposal_network_exporter.export(proposal_networks_mlp_output_path.format(i))