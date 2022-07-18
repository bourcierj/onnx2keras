# onnx2keras

ONNX to Keras deep neural network converter.

Convert an ONNX graph to a Keras model, that (hopefully) should returns the same output(s) as the
source model given the same input(s).

[![GitHub License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-2.7%2C3.6-lightgrey.svg)](https://github.com/gmalivenko/onnx2keras)
[![Downloads](https://pepy.tech/badge/onnx2keras)](https://pepy.tech/project/onnx2keras)
![PyPI](https://img.shields.io/pypi/v/onnx2keras.svg)

## Requirements

TensorFlow 2.0

## API

*function* `onnx_to_keras(onnx_model: onnx.ModelProto, input_names: Sequence[str], input_shapes: Sequence[Tuple[Optional[int]]] = None, name_policy: str = None, verbose: bool = True, change_ordering: bool = False, raise_error_on_lambda_layers: bool = False) -> tf.keras.Model`

>   Convert an ONNX graph to a Keras model.
>   * `onnx_model`: loaded ONNX model
>   * `input_names`: input names, optional
>   * `input_shapes`: input shapes to override (experimental)
>   * `name_policy`: override layer names. None, "short" or "renumerate", or "keras" (experimental):
>       - None uses the ONNX graph node output name.
>       - "short" takes the first 8 characters of the ONNX graph node.
>       - "renumerate" is the prefix 'LAYER_' followed by the node number in conversion order.
>       - "keras" uses Keras layers default names (with the advantage to give understandable and easy to process names).
>   * `verbose`: verbose output
>   * `change_ordering`: change tensor dimensions ordering, from channels-first (batch, channels, ...) to channels-last (batch, ..., channels).
>           True should be considered experimental; it applies manual tweaks for certain layers to (hopefully) get the same output at the end.
>   * `raise_error_on_lambda_layers`: raise an error if the obtained Keras model is composed of at least one `tf.keras.layers.Lambda` layer.
        Use this as a sanity check if you intend to load the converted Keras model in a different environment; indeed, deserializing a model
        with `Lambda` layers in a different environment where it was saved will results in an error when calling it. This is a limitation of `Lambda`
        layers, (according to [Keras docs on Lambda layer](https://keras.io/api/layers/core_layers/lambda/)).


## Getting started

### ONNX model
```python
import onnx
from onnx2keras import onnx_to_keras

# Load ONNX model
onnx_model = onnx.load('resnet18.onnx')

# Call the converter (input - is the main model input name, can be different for your model)
k_model = onnx_to_keras(onnx_model, ['input'])
```

The converted Keras model will be stored to the `k_model` variable. So simple, isn't it?


### PyTorch model

Using ONNX as intermediate format, you can convert PyTorch model as well.

```python
import numpy as np
import torch
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras
import torchvision.models as models

if __name__ == '__main__':
    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    input_var = Variable(torch.FloatTensor(input_np))
    model = models.resnet18()
    model.eval()
    k_model = \
        pytorch_to_keras(model, input_var, [(3, 224, 224,)], verbose=True, change_ordering=True)

    for i in range(3):
        input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)
        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(np.transpose(input_np, [0, 2, 3, 1]))
        error = np.max(pytorch_output - keras_output)
        print('error -- ', error)  # Around zero :)
```

### Deploying model as frozen graph

You can try using the snippet below to convert your ONNX / PyTorch model to frozen graph. It may be useful for deploy for TensorFlow.js / for TensorFlow for Android / for TensorFlow C-API.

```python
import numpy as np
import torch
from pytorch2keras.converter import pytorch_to_keras
from torch.autograd import Variable
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from my_module import Model  # your torch.nn.Module model

# Create and load model
model = Model()
model.load_state_dict(torch.load('model-checkpoint.pth'))
model.eval()

# Make dummy variables (and checking if the model works)
input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
input_var = Variable(torch.FloatTensor(input_np))
output = model(input_var)

# Convert the model!
k_model = \
    pytorch_to_keras(model, input_var, (3, 224, 224), 
                     verbose=True, name_policy='short',
                     change_ordering=True)

# Save model to SavedModel format
tf.saved_model.save(k_model, "./models")

# Convert Keras model to ConcreteFunction
full_model = tf.function(lambda x: k_model(x))
full_model = full_model.get_concrete_function(
    tf.TensorSpec(k_model.inputs[0].shape, k_model.inputs[0].dtype))

# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

print("-" * 50)
print("Frozen model layers: ")
for layer in [op.name for op in frozen_func.graph.get_operations()]:
    print(layer)

print("-" * 50)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)

# Save frozen graph from frozen ConcreteFunction to hard drive
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir="./frozen_models",
                  name="frozen_graph.pb",
                  as_text=False)
```

## Troubleshooting

### Model loading in different environment raises error

Todo...


## License
This software is covered by MIT License.
