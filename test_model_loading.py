"""Pure model loading keeps tensor compatibility without arbitrary object loading."""
import pickle
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

import azero as az


@pytest.mark.parametrize("input_channels", [64, 71, az.STATE_CHANNELS])
def test_weight_loading_preserves_legacy_channel_padding(tmp_path, input_channels):
    net = az.PolicyValueNet(channels=16, blocks=1)
    weights = net.state_dict()
    weights["conv.weight"] = weights["conv.weight"][:, :input_channels].clone()
    path = tmp_path / "model.pt"
    torch.save(weights, path)
    restored = az.PolicyValueNet(channels=16, blocks=1)
    with mock.patch.object(torch, "load", wraps=torch.load) as load:
        assert az.load_model_if_exists(restored, str(path), torch.device("cpu"))
    assert load.call_args.kwargs["weights_only"] is True
    for name, value in restored.state_dict().items():
        if name == "conv.weight":
            torch.testing.assert_close(value[:, :input_channels], weights[name], rtol=0, atol=0)
            assert torch.count_nonzero(value[:, input_channels:]) == 0
        else:
            torch.testing.assert_close(value, weights[name], rtol=0, atol=0)


def test_weight_loading_rejects_python_objects_before_modifying_network(tmp_path):
    path = tmp_path / "objects.pt"
    torch.save({"object": SimpleNamespace(value=1)}, path)
    net = az.PolicyValueNet(channels=16, blocks=1)
    with mock.patch.object(net, "load_state_dict") as apply:
        with pytest.raises(pickle.UnpicklingError):
            az.load_model_if_exists(net, str(path), torch.device("cpu"))
    apply.assert_not_called()


def test_missing_weights_keep_existing_optional_load_behavior(tmp_path):
    net = az.PolicyValueNet(channels=16, blocks=1)
    assert not az.load_model_if_exists(net, str(tmp_path / "missing.pt"), torch.device("cpu"))
