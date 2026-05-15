import sys
import types

import torch

from turboquant_vllm import build, expert_pruning, flute_build


class _FakeTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return {"input_ids": [[1, 2, 3] for _ in texts]}


def test_build_cuda_version_tuple_accepts_major_only(monkeypatch):
    monkeypatch.setattr(torch.version, "cuda", "12")
    assert build._cuda_version_tuple() == (12, 0)


def test_flute_build_cuda_version_tuple_accepts_major_only(monkeypatch):
    monkeypatch.setattr(torch.version, "cuda", "12")
    assert flute_build._cuda_version_tuple() == (12, 0)


def test_prepare_calibration_data_falls_back_when_filter_removes_all_samples(monkeypatch):
    fake_datasets = types.ModuleType("datasets")

    def fake_load_dataset(*args, **kwargs):
        return iter(
            [
                {"text": "too short"},
                {"text": "still short"},
                {"text": ""},
            ]
        )

    fake_datasets.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    tokenizer = _FakeTokenizer()
    inputs = expert_pruning._prepare_calibration_data(
        tokenizer=tokenizer,
        num_samples=3,
        max_length=8,
        dataset_name="dummy",
        device=torch.device("cpu"),
    )

    assert len(inputs) == 3
    assert len(tokenizer.calls) == 1
    texts, kwargs = tokenizer.calls[0]
    assert len(texts) == 3
    assert all(text.startswith("The quick brown fox") for text in texts)
    assert kwargs["max_length"] == 8
