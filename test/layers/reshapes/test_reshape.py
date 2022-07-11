import numpy as np
import torch.nn as nn
import pytest

from test.utils import convert_and_test


class FlattenLayerTest(nn.Module):
    def __init__(self):
        super(FlattenLayerTest, self).__init__()

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return x


@pytest.mark.parametrize('change_ordering', [True, False])
def test_reshape_flatten(change_ordering):
    model = FlattenLayerTest()
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)


class UnflattenLayerTest(nn.Module):
    def __init__(self):
        super(UnflattenLayerTest, self).__init__()

    def forward(self, x):
        x = x.reshape(x.size(0), 3, 224, 224)
        return x


@pytest.mark.parametrize('change_ordering', [True, False])
def test_reshape_unflatten(change_ordering):
    model = UnflattenLayerTest()
    model.eval()
    input_np = np.random.uniform(0, 1, (1, 150528,))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)
