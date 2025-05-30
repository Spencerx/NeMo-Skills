# SPDX-License-Identifier: Apache-2.0
"""
Saves each worker's model state dict directly to a checkpoint, which enables a
fast load path for large tensor-parallel models where each worker only needs to
read its own shard rather than the entire checkpoint.

Example usage:

python save_sharded_state.py \
    --model-path /path/to/load \
    --quantization deepspeedfp \
    --tensor-parallel-size 8 \
    --output /path/to/save

Then, the model can be loaded with

llm = Engine(
    model_path="/path/to/save",
    load_format="sharded_state",
    quantization="deepspeedfp",
    tensor_parallel_size=8,
)
"""

# copied from https://github.com/sgl-project/sglang/blob/main/examples/runtime/engine/save_sharded_state.py

import dataclasses
import os
import shutil
from argparse import ArgumentParser
from pathlib import Path

from sglang import Engine, ServerArgs

parser = ArgumentParser()
ServerArgs.add_cli_args(parser)

parser.add_argument("--output", "-o", required=True, type=str, help="path to output checkpoint")
parser.add_argument("--file-pattern", type=str, help="string pattern of saved filenames")
parser.add_argument(
    "--max-file-size",
    type=str,
    default=5 * 1024**3,
    help="max size (in bytes) of each safetensors file",
)


def main(args):
    engine_args = ServerArgs.from_cli_args(args)
    model_path = engine_args.model_path
    if not Path(model_path).is_dir():
        raise ValueError("model path must be a local directory")
    # Create LLM instance from arguments
    llm = Engine(**dataclasses.asdict(engine_args))
    Path(args.output).mkdir(exist_ok=True)
    llm.save_sharded_model(path=args.output, pattern=args.file_pattern, max_size=args.max_file_size)

    # Copy metadata files to output directory
    for file in os.listdir(model_path):
        if os.path.splitext(file)[1] not in (".bin", ".pt", ".safetensors"):
            if os.path.isdir(os.path.join(model_path, file)):
                shutil.copytree(os.path.join(model_path, file), os.path.join(args.output, file))
            else:
                shutil.copy(os.path.join(model_path, file), args.output)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
