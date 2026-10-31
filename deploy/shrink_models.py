"""Halve the size of the browser models, so the first visit downloads less.

    python deploy/shrink_models.py

Large weights are stored as 16-bit numbers and turned back into 32-bit ones when the model loads, so
nothing else in the browser changes. The result goes to models/web/, which the build publishes.
"""
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

MODELS = Path(__file__).resolve().parent.parent / "models"
NAMES = ["person.onnx", "close.onnx", "water.onnx", "fire.onnx"]
SMALLEST = 1024  # small tables can be exact ratios or shapes, so they are left alone
LARGEST_16_BIT = 65504.0


def shrink(path, out):
    model = onnx.load(path)
    graph = model.graph
    casts = []
    for weights in graph.initializer:
        values = numpy_helper.to_array(weights)
        if weights.data_type != TensorProto.FLOAT or values.size < SMALLEST:
            continue
        name = weights.name
        small = numpy_helper.from_array(np.clip(values, -LARGEST_16_BIT, LARGEST_16_BIT).astype(np.float16), f"{name}_16")
        weights.CopyFrom(small)
        casts.append(helper.make_node("Cast", [small.name], [name], to=TensorProto.FLOAT, name=f"{name}_to_32"))
    nodes = casts + list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes)
    onnx.checker.check_model(model)
    onnx.save(model, out)
    print(f"{path.name}: {path.stat().st_size / 1e6:.0f} MB to {out.stat().st_size / 1e6:.0f} MB")


def main():
    (MODELS / "web").mkdir(exist_ok=True)
    for name in NAMES:
        shrink(MODELS / name, MODELS / "web" / name)


if __name__ == "__main__":
    main()
