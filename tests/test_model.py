from s2m6_ccex.model2 import Model
import pytest
import torch

@pytest.mark.parametrize("batch_size", [32, 64])
def test_my_model(batch_size: int):

    model = Model()

    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        model(torch.randn(1,2,3))

    with pytest.raises(ValueError, match=r'Expected each sample to have shape \[1, 28, 28\]'):
        model(torch.randn(batch_size,1,28,29))

    dummy_input = torch.randn(batch_size, 1, 28, 28)
    output = model(dummy_input)

    assert output.shape == (batch_size, 10), "Output dimension was different from 10"

    