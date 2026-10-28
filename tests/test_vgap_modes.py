"""ADR-0001 tests for the VGAP.MODE switch (section 6.8)."""

import contextlib
import math

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


def test_trainable_params_extend_original():
    from trainers import biomedccpl

    names = biomedccpl.TRAINABLE_PARAM_NAMES
    assert names[: len(ORIGINAL_TRAINABLE)] == ORIGINAL_TRAINABLE
    assert names[len(ORIGINAL_TRAINABLE) :] == ["image_readout"]


def test_invalid_mode_rejected(biomedclip):
    from trainers import biomedccpl

    with pytest.raises(ValueError):
        biomedccpl.CustomCLIP(make_cfg(MODE="bogus"), CLASSNAMES, biomedclip)


@pytest.mark.parametrize("layers", [[], [12], [-1]])
def test_invalid_image_layers_rejected(biomedclip, layers):
    from trainers import biomedccpl

    with pytest.raises(ValueError):
        biomedccpl.CustomCLIP(
            make_cfg(MODE="image", IMAGE_LAYERS=layers), CLASSNAMES, biomedclip
        )


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


# MODE "image" (R2/R3).
@pytest.fixture(scope="module")
def r2_models(biomedclip, r1_models):
    """(MODE "none", MODE "image") built from the same seed."""
    from trainers import biomedccpl

    _, none_model = r1_models
    image_model = build(
        biomedccpl, make_cfg(MODE="image"), biomedclip,
        biomedccpl.TRAINABLE_PARAM_NAMES,
    )
    return none_model, image_model


def test_image_mode_prompt_init_matches_none(r2_models):
    none_model, image_model = r2_models
    for name in ("ctx_causal", "ctx_non_causal"):
        assert torch.equal(getattr(none_model, name), getattr(image_model, name)), name


# 6.8.2: at init (beta = 0) MODE "image" equals MODE "none".
def test_image_mode_equals_none_at_init(r2_models):
    none_model, image_model = r2_models
    x = images(10)
    with torch.no_grad():
        expected = none_model.eval()(x)
        actual = image_model.eval()(x)
    assert image_model.logs["beta"] == 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)

    x = images(2)
    label = torch.tensor([0, 1])
    expected, _ = train_step(none_model, x, label)
    actual, _ = train_step(image_model, x, label)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=0, atol=1e-6)


# 6.8.5: the gate gets a gradient first; W_o once the gate has opened.
def test_image_readout_gradients_over_two_steps(biomedclip):
    from trainers import biomedccpl

    model = build(
        biomedccpl, make_cfg(MODE="image"), biomedclip,
        biomedccpl.TRAINABLE_PARAM_NAMES,
    )
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0025)  # the shipped configs' LR
    x, label = images(1), torch.tensor([0])
    gate, w_o = "image_readout.g", "image_readout.W_o.weight"

    _, grads = train_step(model, x, label)
    assert grads[gate].abs() > 0
    assert grads[w_o].abs().max() == 0  # beta = 0 blocks it exactly
    optimizer.step()
    assert model.image_readout.g.item() != 0

    _, grads = train_step(model, x, label)
    assert grads[w_o].abs().max() > 0


# 6.8.6: shapes.
@pytest.mark.parametrize("layers", [[11], [3, 7, 11]])
def test_readout_shapes(layers):
    from trainers.biomedccpl import ImageSideReadout

    readout = ImageSideReadout(layers)
    hidden = {layer: torch.randn(2, 196, 768) for layer in layers}
    assert readout.prototypes(hidden).shape == (2, 14 * len(layers), 768)
    features, logs = readout(hidden, torch.randn(2, 768), torch.randn(2, 512))
    assert features.shape == (2, 512)
    torch.testing.assert_close(features.norm(dim=-1), torch.ones(2))
    assert 0 <= logs["attn_entropy"] <= math.log(14 * len(layers)) + 1e-5


def test_r3_train_step_runs(biomedclip):
    from trainers import biomedccpl

    model = build(
        biomedccpl, make_cfg(MODE="image", IMAGE_LAYERS=[11, 3, 7]), biomedclip,
        biomedccpl.TRAINABLE_PARAM_NAMES,
    )
    assert model.image_readout.layers == [3, 7, 11]
    outputs, grads = train_step(model, images(2), torch.tensor([0, 1]))
    assert torch.isfinite(outputs[1])
    assert "image_readout.layer_emb" in grads


# 6.8.7: trainable parameters per top-level module.
def test_trainable_param_counts(biomedclip, models, r1_models, r2_models):
    from trainers import biomedccpl

    r3 = build(
        biomedccpl, make_cfg(MODE="image", IMAGE_LAYERS=[3, 7, 11]), biomedclip,
        biomedccpl.TRAINABLE_PARAM_NAMES,
    )
    prompts = {"ctx_causal": 4 * 768, "ctx_non_causal": 4 * 768}
    prototype_learner = 768 * 256 + 2 * 256 + 256 * 14
    attention = 4 * 768 * 768 + 4 * 768
    readout = 2 * 2 * 768 + attention + 768 * 512 + 1  # 2 LayerNorms, attn, W_o, g

    count = biomedccpl.count_trainable_params
    assert count(models[1]) == {
        **prompts,
        "prototype_learner": 3 * prototype_learner,
        "crossattention_layer": 3 * attention,
        "crossattention_layer_nc": 3 * attention,
    }
    assert count(r1_models[1]) == prompts
    assert count(r2_models[1]) == {
        **prompts, "image_readout": prototype_learner + readout
    }
    assert count(r3) == {
        **prompts, "image_readout": 3 * prototype_learner + 3 * 768 + readout
    }
    # The ADR's estimates: ~14.8M (R0) and ~3M (R2).
    assert sum(count(models[1]).values()) == 14_782_464
    assert sum(count(r2_models[1]).values()) == 2_965_505


# 6.8.8: the code's SAR (element-wise MSE) is (2 / D) x the paper's form.
@pytest.mark.parametrize("fixture", ["models", "r1_models", "r2_models"])
def test_sar_matches_paper_form(request, biomedclip, fixture):
    _, model = request.getfixturevalue(fixture)
    outputs, _ = train_step(model, images(1), torch.tensor([0]))
    dim = biomedclip.visual.head.proj.out_features
    assert dim == 512
    torch.testing.assert_close(
        outputs[5], (2 / dim) * model.logs["loss_sar_cos"], rtol=1e-5, atol=1e-9
    )


def test_logs_are_scalars_per_mode(models, r1_models, r2_models):
    expected_keys = {
        "text": {"loss_sar_cos"},
        "none": {"loss_sar_cos"},
        "image": {"loss_sar_cos", "beta", "r_norm_ratio", "attn_entropy"},
    }
    for _, model in (models, r1_models, r2_models):
        train_step(model, images(1), torch.tensor([0]))
        assert set(model.logs) == expected_keys[model.vgap_mode]
        assert all(v.dim() == 0 and not v.requires_grad for v in model.logs.values())


# 6.8.3: text-encoder sequences per training step.
@pytest.mark.parametrize("batch_size", [1, 4])
def test_text_mode_sequences_per_step(biomedclip, models, batch_size):
    _, text_model = models
    n_cls = len(CLASSNAMES)
    x = images(batch_size)
    with count_text_sequences(biomedclip) as sizes:
        train_step(text_model, x, torch.arange(batch_size) % n_cls)
    assert sizes == [batch_size * n_cls, batch_size * n_cls]


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_decoupled_sequences_per_step(biomedclip, r2_models, batch_size):
    n_cls = len(CLASSNAMES)
    x = images(batch_size)
    for model in r2_models:  # MODE "none" and "image"
        with count_text_sequences(biomedclip) as sizes:
            train_step(model, x, torch.arange(batch_size) % n_cls)
        assert sizes == [n_cls, n_cls], model.vgap_mode  # independent of B


# 6.8.4: no gradient reaches the frozen encoders.
@pytest.mark.parametrize("fixture", ["models", "r1_models", "r2_models"])
def test_frozen_encoders_get_no_grad(request, biomedclip, fixture):
    _, model = request.getfixturevalue(fixture)
    train_step(model, images(1), torch.tensor([0]))
    for name, param in biomedclip.named_parameters():
        assert not param.requires_grad, name
        assert param.grad is None, name
