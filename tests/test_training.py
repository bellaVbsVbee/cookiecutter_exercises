import torch
import pytest
from s2m6_ccex.train2 import train
from s2m6_ccex.model2 import Model
import os

@pytest.mark.skip(reason="Checkpoint file not available in CI")
def test_my_training_loop():

    torch.manual_seed(0)
    model = Model()

    root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
    )
    
    models_dir = os.path.join(root, "models")
    save_path = os.path.join(models_dir, "corrupt_mnist_checkpoint.pth")
    state_dict = torch.load(save_path)
    model.load_state_dict(state_dict)

    model.eval()

    with torch.no_grad():
        dummy_input = torch.randn(5, 1, 28, 28)
        new_output = model(dummy_input)
        preds = new_output.argmax(dim=1)

        assert new_output.shape == (5, 10)
        assert torch.isfinite(new_output).all()

        assert preds.min() >= 0
        assert preds.max() < 10