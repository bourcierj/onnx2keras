import numpy as np
import pytest
import tensorflow as tf

from torchvision.models import resnet18

from test.utils import convert_and_test, convert


@pytest.mark.parametrize('change_ordering', [True, False])
def test_resnet18(change_ordering):
    if not tf.test.gpu_device_name() and not change_ordering:
        pytest.skip("Skip! Since tensorflow Conv2D op currently only supports the NHWC tensor format on the CPU")
    model = resnet18()
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)


@pytest.mark.parametrize('change_ordering', [True, False])
def test_resnet18_name_policy_keras(change_ordering):
    if not tf.test.gpu_device_name() and not change_ordering:
        pytest.skip("Skip! Since tensorflow Conv2D op currently only supports the NHWC tensor format on the CPU")
    model = resnet18()
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))

    k_model = convert(model, input_np, name_policy="keras", verbose=False, change_ordering=change_ordering)

    # assert we get the expected layer names for "keras" name policy
    expected_layer_names = ['test_in', 'zero_padding2d', 'conv2d', 'activation', 'zero_padding2d_1', 'max_pooling2d', 'zero_padding2d_2', 'conv2d_1', 'activation_1', 'zero_padding2d_3', 'conv2d_2', 'add', 'activation_2', 'zero_padding2d_4', 'conv2d_3', 'activation_3', 'zero_padding2d_5', 'conv2d_4', 'add_1', 'activation_4', 'zero_padding2d_6', 'conv2d_5', 'activation_5', 'zero_padding2d_7', 'conv2d_6', 'conv2d_7', 'add_2', 'activation_6', 'zero_padding2d_8', 'conv2d_8', 'activation_7', 'zero_padding2d_9', 'conv2d_9', 'add_3', 'activation_8', 'zero_padding2d_10', 'conv2d_10', 'activation_9', 'zero_padding2d_11', 'conv2d_11', 'conv2d_12', 'add_4', 'activation_10', 'zero_padding2d_12', 'conv2d_13', 'activation_11', 'zero_padding2d_13', 'conv2d_14', 'add_5', 'activation_12', 'zero_padding2d_14', 'conv2d_15', 'activation_13', 'zero_padding2d_15', 'conv2d_16', 'conv2d_17', 'add_6', 'activation_14', 'zero_padding2d_16', 'conv2d_18', 'activation_15', 'zero_padding2d_17', 'conv2d_19', 'add_7', 'activation_16', 'global_average_pooling2d', 'reshape', 'reshape_1', 'dense']
    layer_names = [layer.name for layer in k_model.layers]

    assert len(layer_names) == len(expected_layer_names)
    assert layer_names == expected_layer_names
