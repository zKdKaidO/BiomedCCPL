"""ADR-0001 Phase 0 diagnostics: VGAPProbe and tools/phase0_diagnostics.py."""

import math

import pytest
import torch

from conftest import CLASSNAMES, load_original_module, make_cfg
from test_vgap_modes import ORIGINAL_TRAINABLE, build, images
from tools.phase0_diagnostics import pairwise_js_divergence, run_phase0


@pytest.fixture(scope="module")
def text_models(biomedclip):
    """(MODE "text", MODE "text_agnostic") built from the same seed."""
    from trainers import biomedccpl

    names = biomedccpl.TRAINABLE_PARAM_NAMES
    return (
        build(biomedccpl, make_cfg(), biomedclip, names).eval(),
        build(biomedccpl, make_cfg(MODE="text_agnostic"), biomedclip, names).eval(),
    )


def logits_with(model, probe, x):
    model.vgap_probe = probe
    try:
        with torch.no_grad():
            return model(x)
    finally:
        model.vgap_probe = None


def test_js_divergence_bounds():
    same = torch.softmax(torch.randn(1, 14), -1).expand(4, 14)
    assert pairwise_js_divergence(same).item() == pytest.approx(0, abs=1e-6)
    disjoint = torch.eye(4, 14)  # 4 classes, each on its own prototype
    assert pairwise_js_divergence(disjoint).item() == pytest.approx(math.log(2))
    assert pairwise_js_divergence(torch.stack([same, disjoint])).shape == (2,)


def test_probe_rejects_unknown_intervention():
    from trainers.biomedccpl import VGAPProbe

    with pytest.raises(ValueError):
        VGAPProbe("bogus")


def test_probe_without_intervention_changes_nothing(text_models):
    from trainers.biomedccpl import VGAPProbe

    model, _ = text_models
    x = images(3)
    expected = logits_with(model, None, x)
    probe = VGAPProbe(non_causal=True)
    actual = logits_with(model, probe, x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert sorted(probe.attn) == [
        (branch, layer) for branch in ("causal", "non_causal") for layer in (3, 7, 11)
    ]
    assert all(a.shape == (3, len(CLASSNAMES), 14) for a in probe.attn.values())


def test_text_only_everywhere_equals_no_vgap(biomedclip, text_models):
    """Removing the readout at every layer gives the original model without
    VGAP injection (CROSSLAYERS []), still run in lockstep."""
    from trainers.biomedccpl import VGAPProbe

    model, _ = text_models
    cfg = make_cfg()
    cfg.TRAINER.BIOMEDCCPL.CROSSLAYERS = []
    no_vgap = build(load_original_module(), cfg, biomedclip, ORIGINAL_TRAINABLE)
    x = images(2)
    with torch.no_grad():
        expected = no_vgap.eval()(x)
    actual = logits_with(model, VGAPProbe("text_only", layers=[3, 7, 11]), x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-5)

    # Only the readout's contribution is removed at layer 11, not the alpha scale.
    alpha_only = logits_with(model, VGAPProbe("text_only_alpha", layers=[11]), x)
    unscaled = logits_with(model, VGAPProbe("text_only", layers=[11]), x)
    full = logits_with(model, None, x)
    assert not torch.allclose(alpha_only, full)
    assert not torch.allclose(alpha_only, unscaled)


def test_run_phase0_text_mode(text_models):
    model, _ = text_models
    batches = [(images(2, seed=s), torch.tensor([0, 1])) for s in range(2)]
    results = run_phase0(model, batches)
    assert model.vgap_probe is None
    assert results["n_images"] == 4
    assert results["alpha"] == model.alpha
    assert set(results["accuracy"]) == {"full", "text_only_alpha", "text_only", "swap"}
    assert sorted(results["js_divergence"]) == sorted(
        f"{b}_L{layer}" for b in ("causal", "non_causal") for layer in (3, 7, 11)
    )
    assert all(0 < v <= math.log(2) for v in results["js_divergence"].values())
    assert all(v > 0 for v in results["mean_rel_logit_change"].values())
    assert results["swap_drop"] == pytest.approx(
        results["accuracy"]["full"] - results["accuracy"]["swap"]
    )


def test_run_phase0_text_agnostic_is_class_agnostic(text_models):
    from trainers.biomedccpl import VGAPProbe

    _, model = text_models
    x = images(2)
    torch.testing.assert_close(
        logits_with(model, VGAPProbe("class_mean", layers=[3, 7, 11]), x),
        logits_with(model, None, x), rtol=0, atol=1e-5,
    )
    results = run_phase0(model, [(x, torch.tensor([0, 1]))])
    assert results["swap_drop"] == 0
    assert all(v == pytest.approx(0, abs=1e-6) for v in results["js_divergence"].values())


def test_run_phase0_rejects_decoupled_modes(biomedclip):
    from trainers import biomedccpl

    model = build(biomedccpl, make_cfg(MODE="none"), biomedclip, [])
    with pytest.raises(ValueError):
        run_phase0(model, [])
