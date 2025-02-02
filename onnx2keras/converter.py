"""
The ONNX to Keras converter module
"""
from typing import List, Dict, Tuple, Callable, Collection, Sequence, Any, Optional, cast
import logging
import inspect
import collections

import tensorflow as tf
from tensorflow import keras
import onnx
import onnx.numpy_helper
import numpy as np

from .layers import AVAILABLE_CONVERTERS
from .exceptions import LambdaLayerError

def onnx_node_attributes_to_dict(args: Collection[onnx.AttributeProto]) -> Dict[str, Any]:
    """
    Parse ONNX attributes to Python dictionary
    :param args: ONNX attributes object
    :return: Python dictionary of attributes names mapped to processed values
    """
    def onnx_attribute_to_dict(onnx_attr: onnx.AttributeProto) -> Any:
        """
        Parse ONNX attribute
        :param onnx_attr: ONNX attribute
        :return: Python data type
        """
        if onnx_attr.HasField('t'):
            return onnx.numpy_helper.to_array(getattr(onnx_attr, 't'))

        for attr_type in ['f', 'i', 's']:
            if onnx_attr.HasField(attr_type):
                return getattr(onnx_attr, attr_type)

        for attr_type in ['floats', 'ints', 'strings']:
            if getattr(onnx_attr, attr_type):
                return list(getattr(onnx_attr, attr_type))
    return {arg.name: onnx_attribute_to_dict(arg) for arg in args}


def onnx_to_keras(onnx_model: onnx.ModelProto,
                  input_names: Sequence[str],
                  input_shapes: Sequence[Tuple[Optional[int]]] = None,
                  name_policy: str = None,
                  verbose: bool = True,
                  change_ordering: bool = False,
                  raise_error_on_lambda_layers: bool = False) -> keras.Model:
    """
    Convert an ONNX graph to a Keras model.
    :param onnx_model: loaded ONNX model
    :param input_names: input names, optional
    :param input_shapes: input shapes to override (experimental)
    :param name_policy: override layer names. None, "short" or "renumerate", or "keras" (experimental):
        - None uses the ONNX graph node output name.
        - "short" takes the first 8 characters of the ONNX graph node.
        - "renumerate" is the prefix 'LAYER_' followed by the node number in conversion order.
        - "keras" uses Keras layers default names (with the advantage to give understandable and easy to process names).
    :param verbose: verbose output
    :param change_ordering: change tensor dimensions ordering, from channels-first (batch, channels, ...) to channels-last (batch, ..., channels).
        True should be considered experimental; it applies manual tweaks for certain layers to (hopefully) get the same output at the end.
    :param raise_error_on_lambda_layers: raise an error if the obtained Keras model is composed of at least one `tf.keras.layers.Lambda` layer.
        Use this as a sanity check if you intend to load the converted Keras model in a different environment; indeed, deserializing a model
        with `Lambda` layers in a different environment where it was saved will results in an error when calling it. This is a limitation of `Lambda`
        layers, (according to [Keras docs on Lambda layer](https://keras.io/api/layers/core_layers/lambda/).
    """
    # Use channels first format by default.
    keras_fmt = keras.backend.image_data_format()
    keras.backend.set_image_data_format('channels_first')

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    logger = logging.getLogger('onnx2keras')

    logger.info('Converter is called.')

    onnx_weights = onnx_model.graph.initializer  # collection of TensorProto weights in onnx model graph
    onnx_inputs = onnx_model.graph.input
    onnx_outputs = [i.name for i in onnx_model.graph.output]
    onnx_nodes = onnx_model.graph.node

    logger.debug('List input shapes:')
    logger.debug(input_shapes)

    logger.debug('List inputs:')
    for i, input in enumerate(onnx_inputs):
        logger.debug('Input {0} -> {1}.'.format(i, input.name))

    logger.debug('List outputs:')
    for i, output in enumerate(onnx_outputs):
        logger.debug('Output {0} -> {1}.'.format(i, output))

    logger.debug('Gathering weights to dictionary.')
    weights: Dict[str, np.ndarray] = {}  # dictionary mapping weight names to values as numpy arrays
    for onnx_w in onnx_weights:
        try:
            if len(onnx_w.ListFields()) < 4:
                onnx_extracted_weights_name = onnx_w.ListFields()[1][1]
            else:
                onnx_extracted_weights_name = onnx_w.ListFields()[2][1]
            weights[onnx_extracted_weights_name] = onnx.numpy_helper.to_array(onnx_w)
        except:
            onnx_extracted_weights_name = onnx_w.ListFields()[3][1]
            weights[onnx_extracted_weights_name] = onnx.numpy_helper.to_array(onnx_w)

        logger.debug('Found weight {0} with shape {1}.'.format(
                     onnx_extracted_weights_name,
                     weights[onnx_extracted_weights_name].shape))

    layers: Dict[str, tf.Tensor] = dict()  # dictionary that maps layers/nodes names to their output
    lambda_funcs: Dict[str, Callable] = dict()  # dictionary that maps layers/nodes names of Lambda layers to the function wrapped in the `Lambda` layer
    keras_outputs: List[tf.Tensor] = []  # list of output tensors of the model/graph
    keras_inputs: List[tf.Tensor] = []  # list of input tensors of the model/graph

    for i, input_name in enumerate(input_names):
        for onnx_i in onnx_inputs:
            if onnx_i.name == input_name:
                if input_shapes:
                    input_shape = input_shapes[i]
                else:
                    input_shape = [i.dim_value for i in onnx_i.type.tensor_type.shape.dim][1:]

                layers[input_name] = keras.layers.InputLayer(
                    input_shape=input_shape, name=input_name
                ).output

                keras_inputs.append(layers[input_name])

                logger.debug('Found input {0} with shape {1}'.format(input_name, input_shape))

    # Convert every operation separable
    node_names: List[str] = []  # list of layers/node names in order of conversion
    for node_index, node in enumerate(onnx_nodes):
        node_type: str = node.op_type
        node_params: Dict[str, Any] = onnx_node_attributes_to_dict(node.attribute)

        # Add global converter info:
        node_params['change_ordering'] = change_ordering
        node_params['name_policy'] = name_policy

        node_name: str = str(node.output[0])
        keras_names: List[Optional[str]] = []
        output: str
        for output_index, output in enumerate(node.output):
            if name_policy == 'short':
                #@todo: use `keras.backend.unique_object_name(output, zero_based=True)
                keras_name = keras_name_i = output[:8]
                suffix = 1
                while keras_name_i in node_names:
                    keras_name_i = keras_name + '_' + str(suffix)
                    suffix += 1
                keras_names.append(keras_name_i)
            elif name_policy == 'renumerate':
                postfix = node_index if len(node.output) == 1 else "%s_%s" % (node_index, output_index)
                keras_names.append('LAYER_%s' % postfix)
            elif name_policy == "keras":
                keras_names.append(None)
            else:
                #@todo: verify that `output` is unique within the ONNX graph
                keras_names.append(output)

        if len(node.output) != 1:
            logger.warning('Trying to convert multi-output node')
            node_params['_outputs'] = list(node.output)
            node_names.extend(keras_names)
        else:
            keras_names = cast(str, keras_names)
            keras_names = keras_names[0]
            node_names.append(keras_names)

        logger.debug('######')
        logger.debug('...')
        logger.debug('Converting ONNX operation')
        logger.debug('type: %s', node_type)
        logger.debug('node_name: %s', node_name)
        logger.debug('node_params: %s', node_params)
        logger.debug('...')

        logger.debug('Check if all inputs are available:')
        if len(node.input) == 0 and node_type != 'Constant':
            raise AttributeError('Operation doesn\'t have an input. Aborting.')

        for i, node_input in enumerate(node.input):
            logger.debug('Check input %i (name %s).', i, node_input)
            if node_input not in layers:
                logger.debug('The input not found in layers / model inputs.')

                if node_input in weights:
                    logger.debug('Found in weights, add as a numpy constant.')
                    layers[node_input] = weights[node_input]
                else:
                    raise AttributeError('Current node is not in weights / model inputs / layers.')
        else:
            logger.debug('... found all, continue')

        keras.backend.set_image_data_format('channels_first')

        # Convert ONNX node to Keras layer -> populate the `layers` and `lambda_funcs` dicts
        AVAILABLE_CONVERTERS[node_type](
            node,
            node_params,
            layers,
            lambda_funcs,
            node_name,
            keras_names
        )
        if isinstance(keras_names, list):
            keras_names = keras_names[0]

        try:
            logger.debug('Output TF Layer -> ' + str(layers[keras_names]))
        except KeyError:
            pass

    # Check for terminal nodes
    for layer in onnx_outputs:
        if layer in layers:
            keras_outputs.append(layers[layer])

    # Create model
    model = keras.models.Model(inputs=keras_inputs, outputs=keras_outputs)

    if change_ordering:
        change_ord_axes_map = {
            3: 2,
            1: 3,
            -1: 1
        }

        conf = model.get_config()

        for layer in conf['layers']:
            if layer['config'] and 'shared_axes' in layer['config']:
                # TODO: check axes first (if it's not 4D tensor)
                layer['config']['shared_axes'] = [1, 2]

            if layer['config'] and 'batch_input_shape' in layer['config']:
                layer['config']['batch_input_shape'] = \
                    tuple(np.reshape(np.array(
                        [
                            [None] +
                            list(layer['config']['batch_input_shape'][2:][:]) +
                            [layer['config']['batch_input_shape'][1]]
                        ]), -1
                    ))
            if layer['config'] and 'target_shape' in layer['config']:
                #@fixme: transposing `target_shape` in a possibly incorrect way
                if len(list(layer['config']['target_shape'][1:][:])) > 0:
                    layer['config']['target_shape'] = \
                        tuple(np.reshape(np.array(
                                list(layer['config']['target_shape'][1:]) +
                                [layer['config']['target_shape'][0]]
                            ), -1),)

            if layer['config'] and 'data_format' in layer['config']:
                layer['config']['data_format'] = 'channels_last'
            if layer['config'] and 'axis' in layer['config']:
                axis = layer['config']['axis']
                # BatchNorm wrap axis with ListWrapper instead single INT value
                if isinstance(axis, (tuple, list)):
                    axis = axis[0]
                layer['config']['axis'] = change_ord_axes_map.get(axis, layer['config']['axis'])

        for layer in conf['layers']:
            if 'function' in layer['config'] and layer['config']['function'][1] is not None:
                # lambda layers with custom functions that have arguments
                kerasf = list(layer['config']['function'])  # function config
                dargs = list(kerasf[1])  # function arguments' list (except first arg, i.e. input tensor)
                func = lambda_funcs.get(layer['name'])

                if func:
                    # ReduceSum operation has 'axis' param as array of ints. When onnx uses ReduceSum
                    # to reproduce SoftMax - dargs become something like [[1]] (list of lists)
                    # that why we handle collections.Iterable
                    if len(dargs) > 1 or isinstance(dargs[0], (tuple, list)):
                        params = inspect.signature(func).parameters
                        # if has parameter named 'axes', change value format from NCHW to NHWC
                        i = list(params.keys()).index('axes') if ('axes' in params) else -1

                        if i > 0:
                            i -= 1
                            axes = list(range(len(dargs[i].shape)))
                            axes = axes[0:1] + axes[2:] + axes[1:2]
                            dargs[i] = np.transpose(dargs[i], axes)

                        # if has parameter named 'axis', change value format from NCHW to NHWC
                        i = list(params.keys()).index('axis') if ('axis' in params) else -1

                        if i > 0:
                            i -= 1
                            axis = np.array(dargs[i])
                            axes_map = np.array([0, 3, 1, 2])
                            # to list because some tf operations check only for core python types (e.g tf.norm)
                            dargs[i] = axes_map[axis].tolist()
                    else:
                        # if map exists will change else will remain the same
                        #@fixme: this assumes that `dargs[0]` represents an axis, but it could be
                        # anything else, if it's anything else we should not change its value !
                        dargs[0] = change_ord_axes_map.get(dargs[0], dargs[0])

                kerasf[1] = tuple(dargs)
                layer['config']['function'] = tuple(kerasf)

        keras.backend.set_image_data_format('channels_last')

        model_tf_ordering = keras.models.Model.from_config(conf)

        for dst_layer, src_layer, conf in zip(model_tf_ordering.layers, model.layers, conf['layers']):
            W = src_layer.get_weights()
            # TODO: check axes first (if it's not 4D tensor)
            if conf['config'] and 'shared_axes' in conf['config']:
                W[0] = W[0].transpose(1, 2, 0)
            dst_layer.set_weights(W)

        model = model_tf_ordering

    keras.backend.set_image_data_format(keras_fmt)

    if raise_error_on_lambda_layers:
        logger.debug("Raise an exception on the first Lambda layer found")
        for layer in model.layers:
            if isinstance(layer, keras.layers.Lambda):
                raise LambdaLayerError(layer)

        logger.debug("Them model does not contain any Lambda layer")

    return model
