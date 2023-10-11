import os
import torch
from nerfstudio.exporter.onnx.fields import BaseFieldExporter
from nerfstudio.data.scene_box import SceneBox

from nerfstudio.exporter.onnx.utils import get_field_input

class HashMLPDensityFieldExporter(BaseFieldExporter):

    def export(self, output_prefix):

        ray_samples, aabb = get_field_input()

                
        if self.field.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.field.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), aabb)
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        self.field._sample_locations = positions
        if not self.field._sample_locations.requires_grad:
            self.field._sample_locations.requires_grad = True
        positions_flat = positions.view(-1, 3)

        if not self.field.use_linear:
            field_base = self.field.mlp_base[-1]

        else:
            field_base = self.field.linear
        encoding = self.field.encoding

        field_base_output_path = output_prefix + "_field_base.onnx"
        encoding_output_path = output_prefix + "_encoding.onnx"


        torch.onnx.export(
            encoding,
            positions_flat,
            encoding_output_path,
            export_params=True,
            opset_version=10,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )

        torch.onnx.export(
            field_base,
            ray_samples.frustums if self.field.use_linear else encoding(positions_flat),
            field_base_output_path,
            export_params=True,
            opset_version=10,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )



    
