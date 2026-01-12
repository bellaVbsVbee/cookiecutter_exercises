import torch
import pytest
from s2m6_ccex.train2 import train
from s2m6_ccex.model2 import Model


def test_my_training_loop():
    torch.manual_seed(0)

    trained_model = train()
    trained_model.eval()

    with torch.no_grad():
        dummy_input = torch.randn(5, 1, 28, 28)
        new_output = trained_model(dummy_input)
        preds = new_output.argmax(dim=1)

        assert new_output.shape == (5, 10)
        assert torch.isfinite(new_output).all()

        assert preds.min() >= 0
        assert preds.max() < 10