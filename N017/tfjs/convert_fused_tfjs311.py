"""Convert N017 with bias normalization before TF.js 3.11 graph fusion.

onnx2tf emits channel-bias operations as AddV2. TensorFlow.js 3.11's
conversion passes only fuse Conv2D/DepthwiseConv2dNative/MatMul when the bias
operation is BiasAdd. This script narrowly rewrites eligible AddV2 nodes after
the converter's first Grappler pass, then lets the unmodified converter finish
its normal remap, depthwise-fusion, validation, and weight extraction passes.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

if not hasattr(np, "object"):
    np.object = object
if not hasattr(np, "bool"):
    np.bool = bool

from tensorflowjs.converters import converter
from tensorflowjs.converters import tf_saved_model_conversion_v2


CONTRACTION_OPS = {"Conv2D", "DepthwiseConv2dNative", "MatMul"}


def base_name(name: str) -> str:
    return name.removeprefix("^").split(":", 1)[0]


def tensor_shape(node) -> list[int] | None:
    if node is None or node.op != "Const" or "value" not in node.attr:
        return None
    dims = node.attr["value"].tensor.tensor_shape.dim
    if any(dim.size < 0 for dim in dims):
        return None
    return [int(dim.size) for dim in dims]


def contraction_width(contraction, node_map) -> int | None:
    if len(contraction.input) < 2:
        return None
    weight_shape = tensor_shape(node_map.get(base_name(contraction.input[1])))
    if not weight_shape:
        return None
    if contraction.op == "Conv2D" and len(weight_shape) == 4:
        return weight_shape[3]
    if contraction.op == "DepthwiseConv2dNative" and len(weight_shape) == 4:
        return weight_shape[2] * weight_shape[3]
    if contraction.op == "MatMul" and len(weight_shape) == 2:
        transpose_b = contraction.attr.get("transpose_b")
        return weight_shape[0] if transpose_b and transpose_b.b else weight_shape[1]
    return None


def normalize_channel_bias_adds(graph_def) -> Counter:
    node_map = {node.name: node for node in graph_def.node}
    consumers: Counter = Counter()
    for node in graph_def.node:
        for input_name in node.input:
            consumers[base_name(input_name)] += 1

    rewritten: Counter = Counter()
    for node in graph_def.node:
        if node.op != "AddV2" or len(node.input) != 2:
            continue
        inputs = [node_map.get(base_name(name)) for name in node.input]
        contraction_index = next(
            (index for index, item in enumerate(inputs) if item is not None and item.op in CONTRACTION_OPS),
            None,
        )
        const_index = next(
            (index for index, item in enumerate(inputs) if item is not None and item.op == "Const"),
            None,
        )
        if contraction_index is None or const_index is None or contraction_index == const_index:
            continue

        contraction = inputs[contraction_index]
        bias = inputs[const_index]
        bias_shape = tensor_shape(bias)
        if bias_shape is None or len(bias_shape) != 1:
            continue
        if contraction_width(contraction, node_map) != bias_shape[0]:
            continue
        if consumers[contraction.name] != 1:
            continue

        node.op = "BiasAdd"
        node.input[:] = [contraction.name, bias.name]
        node.attr["data_format"].s = b"NHWC"
        rewritten[contraction.op] += 1
    return rewritten


def output_width(item) -> int:
    dimensions = item[1]["tensorShape"]["dim"]
    return int(dimensions[-1]["size"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("saved_model")
    parser.add_argument("output")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    original_run_grappler = tf_saved_model_conversion_v2._run_grappler
    grappler_pass = 0
    normalized: Counter = Counter()

    def run_grappler_with_bias_normalization(*call_args, **call_kwargs):
        nonlocal grappler_pass
        graph_def = original_run_grappler(*call_args, **call_kwargs)
        grappler_pass += 1
        if grappler_pass == 1:
            normalized.update(normalize_channel_bias_adds(graph_def))
        return graph_def

    tf_saved_model_conversion_v2._run_grappler = run_grappler_with_bias_normalization
    try:
        converter.convert(
            [
                "--input_format=tf_saved_model",
                "--output_format=tfjs_graph_model",
                "--signature_name=serving_default",
                "--saved_model_tags=serve",
                "--strip_debug_ops=True",
                args.saved_model,
                args.output,
            ]
        )
    finally:
        tf_saved_model_conversion_v2._run_grappler = original_run_grappler

    model_path = output / "model.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    ordered = sorted(
        model["signature"]["outputs"].items(),
        key=lambda item: {1: 0, 4: 1}.get(output_width(item), 2),
    )
    widths = [output_width(item) for item in ordered]
    if widths != [1, 4]:
        raise RuntimeError(f"Expected [unsafe, SQRX] output widths [1, 4], received {widths}")
    model["signature"]["outputs"] = dict(ordered)
    model_path.write_text(json.dumps(model, separators=(",", ":")), encoding="utf-8")

    op_counts = Counter(node["op"] for node in model["modelTopology"].get("node", []))
    summary = {
        "format": model.get("format"),
        "generatedBy": model.get("generatedBy"),
        "convertedBy": model.get("convertedBy"),
        "signature": model.get("signature"),
        "node_count": sum(op_counts.values()),
        "op_counts": dict(sorted(op_counts.items())),
        "normalized_bias_adds": dict(sorted(normalized.items())),
        "weight_shards": [path for group in model["weightsManifest"] for path in group["paths"]],
        "weight_count": sum(len(group["weights"]) for group in model["weightsManifest"]),
        "addon_output_order": widths,
    }
    (output / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
