
import pytest
import torch

from src.predict import LABEL_NAMES, predict


class FakeTokenizer:
    """Pretends to convert text into model inputs."""

    def __call__(self, text, **kwargs):
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }


class FakeModel:
    """Pretends to be a model that predicts anger."""

    def __call__(self, **kwargs):
        logits = torch.tensor([[0.1, 0.2, 0.3, 3.0, 0.4, 0.1]])
        return type("Output", (), {"logits": logits})()


def test_prediction_returns_valid_emotion():
    emotion, probabilities = predict(
        "This makes me furious",
        FakeModel(),
        FakeTokenizer(),
        torch.device("cpu"),
    )

    assert emotion in LABEL_NAMES


def test_prediction_returns_six_probabilities():
    _, probabilities = predict(
        "This makes me furious",
        FakeModel(),
        FakeTokenizer(),
        torch.device("cpu"),
    )

    assert len(probabilities) == 6


def test_probabilities_sum_to_one():
    _, probabilities = predict(
        "This makes me furious",
        FakeModel(),
        FakeTokenizer(),
        torch.device("cpu"),
    )

    assert sum(probabilities) == pytest.approx(1.0, abs=1e-5)


def test_probabilities_are_between_zero_and_one():
    _, probabilities = predict(
        "This makes me furious",
        FakeModel(),
        FakeTokenizer(),
        torch.device("cpu"),
    )

    assert all(0.0 <= p <= 1.0 for p in probabilities)


def test_empty_text_is_rejected():
    with pytest.raises(ValueError, match="Input text cannot be empty"):
        predict(
            "   ",
            FakeModel(),
            FakeTokenizer(),
            torch.device("cpu"),
        )
