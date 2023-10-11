import argparse
import os
from pathlib import Path
from nerfstudio.exporter.onnx.models import get_exporter



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("output_path", type=str)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    config_path = Path(args.config_path)
    output_path = Path(args.output_path)

    os.makedirs(output_path, exist_ok=True)

    exporter = get_exporter(config_path)

    exporter.export(output_path)