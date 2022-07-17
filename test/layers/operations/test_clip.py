import torch.nn as nn
import numpy as np
import pytest

from test.utils import convert_and_test, convert
from onnx2keras.exceptions import LambdaLayerError


class FClipTest(nn.Module):
    """
    Test for nn.functional types
    """
    def __init__(self):
        self.low = np.random.uniform(-1, 1)
        self.high = np.random.uniform(1, 2)
        super(FClipTest, self).__init__()

    def forward(self, x):
        return x.clamp(self.low, self.high)


@pytest.mark.parametrize('change_ordering', [True, False])
def test_clip(change_ordering):
    model = FClipTest()
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))

    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)


@pytest.mark.parametrize('change_ordering', [True, False])
def test_clip_raise_error_on_lambda_layers(change_ordering):
    model = FClipTest()
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    with pytest.raises(LambdaLayerError) as excinfo:
        k_model = convert(model, input_np, verbose=False, change_ordering=change_ordering, raise_error_on_lambda_layers=True)
