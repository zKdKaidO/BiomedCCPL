"""tools/benchmark_cost.py (ADR-0001 7.3), run at a minimal size on CPU."""

import pytest
import torch

from conftest import make_cfg
from tools.benchmark_cost import BTMRI_CLASSNAMES, benchmark, build_cfg, build_model


def test_build_cfg_matches_test_config():
    cfg, lr = build_cfg(mode="image", image_layers=[3, 7, 11])
    expected = make_cfg(MODE="image", IMAGE_LAYERS=[3, 7, 11]).TRAINER.BIOMEDCCPL
    assert cfg.TRAINER.BIOMEDCCPL == expected
    assert lr == 0.0025


@pytest.mark.parametrize("mode, per_image", [("text", True), ("image", False)])
def test_benchmark_counts_text_sequences(biomedclip, mode, per_image):
    cfg, lr = build_cfg(mode=mode)
    torch.manual_seed(0)
    model = build_model(biomedclip, cfg, BTMRI_CLASSNAMES, torch.device("cpu"))
    row = benchmark(
        model, 2, torch.device("cpu"), lr=lr, steps=1, warmup=0, test_images=3
    )
    n_cls = len(BTMRI_CLASSNAMES)
    sequences = 2 * n_cls if per_image else n_cls
    assert row["text_sequences_per_step"] == [sequences, sequences]
    assert row["step_time_median_s"] > 0 and row["test_images_per_s"] > 0
    assert row["memory_bytes"] is None and row["oom"] is False
