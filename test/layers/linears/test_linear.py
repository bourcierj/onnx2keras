import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import pytest

from test.utils import convert_and_test


class LayerTest(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(LayerTest, self).__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        x = self.fc(x)
        return x


@pytest.mark.parametrize('change_ordering', [True, False])
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('in_features', [1, 128])
@pytest.mark.parametrize('out_features', [1, 128])
def test_linear(change_ordering, bias, in_features, out_features):

    model = LayerTest(in_features, out_features, bias=bias)
    model.eval()

    input_np = np.random.uniform(0, 1, (1, in_features))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)
