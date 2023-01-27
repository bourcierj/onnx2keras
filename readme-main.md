
## 0. Setup

* Git clone this repository then install `onnx2keras` locally `pip install .` in a virtualenv
* Install dependencies in `requirements.txt`
* Install PyTorch

## 1. Convert directly from PyTorch to Keras

### With variable keras input shape

```
python -m onnx2keras.pytorch2keras_main --in-file '/path/to/pytorch/checkpoint.pth.tar' --intermediate-onnx-file '/optional/path/to/intermediate/checkpoint.onnx' --out-file  '/path/to/output/checkpoint_keras_savedmodel'
```

### With fixed keras input shape

By default input shape spatial dimensions are set to `None` (variable). Add `--input-shape w h` option to fix to a constant *(w, h)* shape.

### Specify backbone architecture

By default the architecture is ResNet50. Add e.g. `--arch "resnet101"` to convert a ResNet101 backbone.


## 2. Create multi-output model for a Feature Pyramid Network


The converted keras model has a single output layer. To convert a single-output model to a multi-output model for a FPN:

### With default FPN layers for ResNet50
```
python -m onnx2keras.create_multi_output_model_for_fpn_main --in-file '/path/to/converted/checkpoint_keras_savedmodel' --out-file '/path/to/output/multi-ouput/checkpoint_keras_savedmodel'
```

The default FPN layers for ResNet50 are: `["activation_9", "activation_21", "activation_39", "activation_48"]`

> **Warning**
> These default layer names are only valid for ResNet50

### With provided FPN layers:

Add `--fpn-layers-names ["activation_a", "activation_b", ...]` to set layers for FPN.
