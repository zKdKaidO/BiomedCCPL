"""ADR-0001 tests for the VGAP.MODE switch (section 6.8)."""

import contextlib

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from conftest import CLASSNAMES, load_original_module, make_cfg

# The trainable-parameter selection used by the original trainer.
ORIGINAL_TRAINABLE = [
    "ctx_causal",
    "ctx_non_causal",
    "prototype_learner",
    "crossattention_layer",
]


def build(module, cfg, biomedclip, trainable, seed=0):
    torch.manual_seed(seed)
    model = module.CustomCLIP(cfg, CLASSNAMES, biomedclip)
    for name, param in model.named_parameters():
        param.requires_grad = any(t in name for t in trainable)
    return model


def images(batch_size, seed=123):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch_size, 3, 224, 224, generator=generator)


def train_step(model, image, label, seed=7):
    model.train()
    model.zero_grad(set_to_none=True)
    torch.manual_seed(seed)  # the frozen encoders' dropout is active in training
    outputs = model(image, label)
    outputs[1].backward()
    grads = {
        name: param.grad.clone()
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is not None
    }
    return [o.detach() for o in outputs], grads


@contextlib.contextmanager
def dropout_off(model):
    """Zero every dropout rate (restored on exit).

    Decoupling the encoders changes the order in which dropout draws its
    masks, so exact train-step comparisons across code paths need it off.
    """
    dropouts = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    rates = [m.p for m in dropouts]
    for m in dropouts:
        m.p = 0.0
    try:
        yield
    finally:
        for m, p in zip(dropouts, rates):
            m.p = p


@contextlib.contextmanager
def count_text_sequences(biomedclip):
    """Record the batch size of every call to the first text-encoder layer."""
    sizes = []
    layer = biomedclip.text.transformer.encoder.layer[0]
    handle = layer.register_forward_hook(lambda m, inp, out: sizes.append(inp[0].shape[0]))
    try:
        yield sizes
    finally:
        handle.remove()


@pytest.fixture(scope="module")
def models(biomedclip):
    from trainers import biomedccpl

    cfg = make_cfg()
    original = build(load_original_module(), cfg, biomedclip, ORIGINAL_TRAINABLE)
    current = build(biomedccpl, cfg, biomedclip, biomedccpl.TRAINABLE_PARAM_NAMES)
    return original, current


def test_defaults():
    vgap = make_cfg().TRAINER.BIOMEDCCPL.VGAP
    assert vgap.MODE == "text"
    assert vgap.IMAGE_LAYERS == [11]
    assert vgap.READOUT_HEADS == 12
    assert vgap.GATE_INIT == 0.0


def test_trainable_params_unchanged():
    from trainers import biomedccpl

    assert biomedccpl.TRAINABLE_PARAM_NAMES == ORIGINAL_TRAINABLE


def test_invalid_mode_rejected(biomedclip):
    from trainers import biomedccpl

    with pytest.raises(ValueError):
        biomedccpl.CustomCLIP(make_cfg(MODE="bogus"), CLASSNAMES, biomedclip)


# 6.8.1: MODE "text" reproduces the original outputs.
def test_text_mode_initial_weights_match_original(models):
    original, current = models
    original_state = original.state_dict()
    current_state = current.state_dict()
    assert original_state.keys() == current_state.keys()
    for key in original_state:
        assert torch.equal(original_state[key], current_state[key]), key


def test_text_mode_eval_logits_match_original(models):
    original, current = models
    x = images(10)
    with torch.no_grad():
        expected = original.eval()(x)
        actual = current.eval()(x)
    assert actual.shape == (10, len(CLASSNAMES))
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_text_mode_train_step_matches_original(models, batch_size):
    original, current = models
    x = images(batch_size)
    label = torch.arange(batch_size) % len(CLASSNAMES)
    expected, expected_grads = train_step(original, x, label)
    actual, actual_grads = train_step(current, x, label)

    # logits, total_loss, loss_ce, loss_ortho, loss_nem, loss_sar
    assert len(actual) == len(expected) == 6
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=0, atol=1e-6)
    assert actual_grads.keys() == expected_grads.keys()
    assert len(actual_grads) > 0
    for name in expected_grads:
        torch.testing.assert_close(
            actual_grads[name], expected_grads[name], rtol=0, atol=1e-6, msg=name
        )


# MODE "none" (R1) equals the original model with no VGAP injection.
@pytest.fixture(scope="module")
def r1_models(biomedclip):
    from trainers import biomedccpl

    original_cfg = make_cfg()
    original_cfg.TRAINER.BIOMEDCCPL.CROSSLAYERS = []
    original = build(
        load_original_module(), original_cfg, biomedclip, ORIGINAL_TRAINABLE
    )
    current = build(
        biomedccpl, make_cfg(MODE="none"), biomedclip,
        biomedccpl.TRAINABLE_PARAM_NAMES,
    )
    return original, current


def test_none_mode_has_only_prompt_params(r1_models):
    _, current = r1_models
    trainable = {n for n, p in current.named_parameters() if p.requires_grad}
    assert trainable == {"ctx_causal", "ctx_non_causal"}
    assert not hasattr(current, "prototype_learner")
    assert not hasattr(current, "crossattention_layer")


def test_none_mode_prompt_init_matches_original(r1_models):
    original, current = r1_models
    for name in ("ctx_causal", "ctx_non_causal"):
        assert torch.equal(getattr(original, name), getattr(current, name)), name


def test_decoupled_image_features_match_encode_image(biomedclip, r1_models):
    _, current = r1_models
    x = images(4)
    current.eval()
    hidden, cls_token, features = current._encode_image_keep(x, layers=(3, 7, 11))
    with torch.no_grad():
        expected = biomedclip.encode_image(x)
    torch.testing.assert_close(
        F.normalize(features, dim=-1), F.normalize(expected, dim=-1),
        rtol=0, atol=1e-6,
    )
    assert sorted(hidden) == [3, 7, 11]
    assert all(h.shape == (4, 196, 768) for h in hidden.values())
    assert cls_token.shape == (4, 768)


def assert_close_to_scale(actual, expected, rel=1e-5, name=""):
    """Max abs difference within `rel` of the largest |expected| entry.

    MODE "none" runs the text encoder on ncls instead of B * ncls sequences,
    so it matches the original only up to float rounding.
    """
    scale = expected.abs().max().clamp_min(1e-12)
    error = (actual - expected).abs().max()
    assert error <= rel * scale, f"{name}: max diff {error:.2e}, scale {scale:.2e}"


def test_none_mode_eval_logits_match_original(r1_models):
    original, current = r1_models
    x = images(10)
    with torch.no_grad():
        expected = original.eval()(x)
        actual = current.eval()(x)
    assert_close_to_scale(actual, expected, name="logits")


@pytest.mark.parametrize("batch_size", [1, 2])
def test_none_mode_train_step_matches_original(r1_models, batch_size):
    original, current = r1_models
    x = images(batch_size)
    label = torch.arange(batch_size) % len(CLASSNAMES)
    with dropout_off(original), dropout_off(current):
        expected, expected_grads = train_step(original, x, label)
        actual, actual_grads = train_step(current, x, label)

    names = ["logits", "total_loss", "loss_ce", "loss_ortho", "loss_nem", "loss_sar"]
    for name, a, e in zip(names, actual, expected):
        assert_close_to_scale(a, e, name=name)
    assert actual_grads.keys() == expected_grads.keys() == {
        "ctx_causal", "ctx_non_causal"
    }
    for name in expected_grads:
        assert_close_to_scale(actual_grads[name], expected_grads[name], name=name)


# 6.8.3: text-encoder sequences per training step.
@pytest.mark.parametrize("batch_size", [1, 4])
def test_text_encoder_sequences_per_step(biomedclip, models, r1_models, batch_size):
    _, text_model = models
    _, none_model = r1_models
    n_cls = len(CLASSNAMES)
    x = images(batch_size)
    label = torch.arange(batch_size) % n_cls

    with count_text_sequences(biomedclip) as sizes:
        train_step(none_model, x, label)
    assert sizes == [n_cls, n_cls]  # causal + non-causal, independent of B

    with count_text_sequences(biomedclip) as sizes:
        train_step(text_model, x, label)
    assert sizes == [batch_size * n_cls, batch_size * n_cls]


# 6.8.4: no gradient reaches the frozen encoders.
@pytest.mark.parametrize("fixture", ["models", "r1_models"])
def test_frozen_encoders_get_no_grad(request, biomedclip, fixture):
    _, model = request.getfixturevalue(fixture)
    train_step(model, images(1), torch.tensor([0]))
    for name, param in biomedclip.named_parameters():
        assert not param.requires_grad, name
        assert param.grad is None, name
