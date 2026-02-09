"""Tests for ppidm_pi.classifier."""

import torch
import pytest

from ppidm_pi.classifier import ResNet1d


@pytest.fixture
def resnet():
    return ResNet1d(in_channels=12, num_classes=3)


class TestResNet1d:
    def test_output_shape(self, resnet, dummy_ecg):
        out = resnet(dummy_ecg)
        assert out.shape == (4, 3)

    def test_logits_per_class(self, resnet, dummy_ecg):
        """Output should have one logit per class."""
        out = resnet(dummy_ecg)
        assert out.shape[-1] == 3

    def test_gradient_flows(self, resnet, dummy_ecg, dummy_labels):
        out = resnet(dummy_ecg)
        loss = torch.nn.functional.cross_entropy(out, dummy_labels)
        loss.backward()
        for name, p in resnet.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_different_num_classes(self, dummy_ecg):
        model = ResNet1d(in_channels=12, num_classes=5)
        out = model(dummy_ecg)
        assert out.shape == (4, 5)

    def test_different_sequence_length(self, resnet):
        x = torch.randn(2, 12, 512)
        out = resnet(x)
        assert out.shape == (2, 3)
