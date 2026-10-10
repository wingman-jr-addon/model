# N017 TensorFlow.js graph fusion

The deployed `n017_graph_model/model.json` is a graph-fused rebuild of the
original N017 TensorFlow.js 3.11 artifact. The learned parameters did not
change: all ten weight shards and the 399-entry weight manifest are identical.
The pre-fusion graph is retained as `n017_graph_model/model.unfused.json` and
uses those same colocated shards.

`convert_fused_tfjs311.py` reproduces the graph from the packaged SavedModel.
It intercepts the converter's first Grappler result and changes an `AddV2` to
`BiasAdd` only when a rank-one constant exactly matches the output width of a
single-consumer `Conv2D`, `DepthwiseConv2dNative`, or `MatMul`. The unchanged
TensorFlow.js 3.11 converter then performs its normal fusion, validation, and
weight extraction passes; the final TensorFlow.js graph is not hand-edited.

Run the script in the original conversion environment, which supplies
TensorFlow.js Converter 3.11.0:

```powershell
python N017/tfjs/convert_fused_tfjs311.py `
  N017/tfjs_saved_model `
  path/to/output/n017_graph_model
```

The conversion normalized 49 convolution, 39 depthwise-convolution, and four
matrix-multiplication bias additions. The resulting graph contains 767 nodes
instead of 898. CPU parity through TensorFlow.js 3.11 remained within
`2.05e-8` for unsafe and `6.06e-9` for S/Q/R/X. A longer Firefox/WebGL smoke
test measured a directional warm-inference improvement from roughly 60 ms to
52 ms on the test machine.
