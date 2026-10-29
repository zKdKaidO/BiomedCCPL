"""ADR-0001 tests for the VGAP.MODE switch (section 6.8)."""

import contextlib
import math
import types

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
    assert names[len(ORIGINAL_TRAINABLE) :] == ["image_readout", "vgap_query"]


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


# MODE "text_agnostic" (R4): the original model with a shared query.
@pytest.fixture(scope="module")
def r4_models(biomedclip, models):
    """(MODE "text", MODE "text_agnostic") built from the same seed."""
    from trainers import biomedccpl

    _, text_model = models
    agnostic = build(
        biomedccpl, make_cfg(MODE="text_agnostic"), biomedclip,
        biomedccpl.TRAINABLE_PARAM_NAMES,
    )
    return text_model, agnostic


def test_text_agnostic_init_matches_text_mode(r4_models):
    text_model, agnostic = r4_models
    text_state, agnostic_state = text_model.state_dict(), agnostic.state_dict()
    assert set(agnostic_state) - set(text_state) == {"vgap_query", "vgap_query_nc"}
    for key in text_state:
        assert torch.equal(text_state[key], agnostic_state[key]), key
    assert agnostic.vgap_query.shape == (3, 1, 768)
    assert agnostic.vgap_query.std().item() == pytest.approx(0.02, rel=0.1)


@contextlib.contextmanager
def record_outputs(modules):
    outputs = []
    handles = [
        m.register_forward_hook(lambda m, inp, out: outputs.append(out[0].detach()))
        for m in modules
    ]
    try:
        yield outputs
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("batch_size", [1, 2])
def test_text_agnostic_readout_is_class_agnostic(r4_models, batch_size):
    text_model, agnostic = r4_models
    x, label = images(batch_size), torch.arange(batch_size)
    for model, agnostic_expected in ((text_model, False), (agnostic, True)):
        modules = [*model.crossattention_layer, *model.crossattention_layer_nc]
        with record_outputs(modules) as outputs:
            train_step(model, x, label)
        assert len(outputs) == 6
        for out in outputs:  # B, ncls, 768
            assert out.shape == (batch_size, len(CLASSNAMES), 768)
            same = torch.allclose(out, out[:, :1].expand_as(out), atol=1e-6)
            assert same == agnostic_expected, model.vgap_mode


def test_text_agnostic_queries_get_grad(r4_models):
    _, agnostic = r4_models
    _, grads = train_step(agnostic, images(1), torch.tensor([0]))
    assert grads["vgap_query"].abs().max() > 0
    assert grads["vgap_query_nc"].abs().max() > 0


# 6.8.7: trainable parameters per top-level module.
def test_trainable_param_counts(biomedclip, models, r1_models, r2_models, r4_models):
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
    assert count(r4_models[1]) == {
        **count(models[1]), "vgap_query": 3 * 768, "vgap_query_nc": 3 * 768
    }
    # The ADR's estimates: ~14.8M (R0) and ~3M (R2).
    assert sum(count(models[1]).values()) == 14_782_464
    assert sum(count(r2_models[1]).values()) == 2_965_505


# 6.8.8: the code's SAR (element-wise MSE) is (2 / D) x the paper's form.
@pytest.mark.parametrize("fixture", ["models", "r1_models", "r2_models", "r4_models"])
def test_sar_matches_paper_form(request, biomedclip, fixture):
    _, model = request.getfixturevalue(fixture)
    outputs, _ = train_step(model, images(1), torch.tensor([0]))
    dim = biomedclip.visual.head.proj.out_features
    assert dim == 512
    torch.testing.assert_close(
        outputs[5], (2 / dim) * model.logs["loss_sar_cos"], rtol=1e-5, atol=1e-9
    )


def test_logs_are_scalars_per_mode(models, r1_models, r2_models, r4_models):
    expected_keys = {
        "text": {"loss_sar_cos"},
        "text_agnostic": {"loss_sar_cos"},
        "none": {"loss_sar_cos"},
        "image": {"loss_sar_cos", "beta", "r_norm_ratio", "attn_entropy"},
    }
    for _, model in (models, r1_models, r2_models, r4_models):
        train_step(model, images(1), torch.tensor([0]))
        assert set(model.logs) == expected_keys[model.vgap_mode]
        assert all(v.dim() == 0 and not v.requires_grad for v in model.logs.values())


# 6.8.3: text-encoder sequences per training step.
@pytest.mark.parametrize("batch_size", [1, 4])
def test_text_mode_sequences_per_step(biomedclip, r4_models, batch_size):
    n_cls = len(CLASSNAMES)
    x = images(batch_size)
    for model in r4_models:  # MODE "text" and "text_agnostic": same cost
        with count_text_sequences(biomedclip) as sizes:
            train_step(model, x, torch.arange(batch_size) % n_cls)
        assert sizes == [batch_size * n_cls, batch_size * n_cls], model.vgap_mode


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_decoupled_sequences_per_step(biomedclip, r2_models, batch_size):
    n_cls = len(CLASSNAMES)
    x = images(batch_size)
    for model in r2_models:  # MODE "none" and "image"
        with count_text_sequences(biomedclip) as sizes:
            train_step(model, x, torch.arange(batch_size) % n_cls)
        assert sizes == [n_cls, n_cls], model.vgap_mode  # independent of B


# 6.8.4: no gradient reaches the frozen encoders.
@pytest.mark.parametrize("fixture", ["models", "r1_models", "r2_models", "r4_models"])
def test_frozen_encoders_get_no_grad(request, biomedclip, fixture):
    _, model = request.getfixturevalue(fixture)
    train_step(model, images(1), torch.tensor([0]))
    for name, param in biomedclip.named_parameters():
        assert not param.requires_grad, name
        assert param.grad is None, name


# READOUT_LR_MULT: the readout gets its own optimizer group (ADR 6.7).
def test_param_groups_only_split_off_the_readout(
    models, r1_models, r2_models, r4_models
):
    from trainers.biomedccpl import build_param_groups

    assert build_param_groups(models[1], 0.0025, 10.0) is None  # R0: unchanged
    assert build_param_groups(r1_models[1], 0.0025, 10.0) is None
    assert build_param_groups(r4_models[1], 0.0025, 10.0) is None

    model = r2_models[1]
    others, readout = build_param_groups(model, 0.0025, 10.0)
    assert "lr" not in others and readout["lr"] == pytest.approx(0.025)
    readout_ids = {id(p) for p in model.image_readout.parameters()}
    assert {id(p) for p in readout["params"]} == readout_ids
    assert {id(p) for p in others["params"]} == {
        id(model.ctx_causal), id(model.ctx_non_causal)
    }


def test_readout_lr_mult_through_dassl_schedule(r2_models):
    dassl_config = pytest.importorskip("dassl.config")
    from dassl.optim import build_lr_scheduler, build_optimizer
    from trainers.biomedccpl import build_param_groups

    model = r2_models[1]
    cfg = dassl_config.get_cfg_default()
    cfg.OPTIM.NAME = "sgd"
    cfg.OPTIM.LR = 0.0025
    cfg.OPTIM.MAX_EPOCH = 50
    cfg.OPTIM.LR_SCHEDULER = "cosine"
    cfg.OPTIM.WARMUP_EPOCH = 1
    cfg.OPTIM.WARMUP_TYPE = "constant"
    cfg.OPTIM.WARMUP_CONS_LR = 1e-5
    groups = build_param_groups(model, cfg.OPTIM.LR, 10.0)
    optimizer = build_optimizer(model, cfg.OPTIM, param_groups=groups)
    scheduler = build_lr_scheduler(optimizer, cfg.OPTIM)

    # Warmup epoch: every group at WARMUP_CONS_LR; afterwards the cosine
    # schedule keeps the readout at 10x the prompt learning rate.
    assert [g["lr"] for g in optimizer.param_groups] == [1e-5, 1e-5]
    scheduler.step()
    base_lr, readout_lr = (g["lr"] for g in optimizer.param_groups)
    assert readout_lr == pytest.approx(10 * base_lr)
    assert base_lr == pytest.approx(0.0025, rel=1e-2)


def test_key_mismatch_warning(capsys):
    from torch.nn.modules.module import _IncompatibleKeys
    from trainers.biomedccpl import warn_on_key_mismatch

    warn_on_key_mismatch(_IncompatibleKeys([], []), "ckpt")
    assert capsys.readouterr().out == ""

    missing = [f"image_readout.p{i}" for i in range(7)]
    warn_on_key_mismatch(_IncompatibleKeys(missing, ["prototype_learner.0"]), "ckpt")
    out = capsys.readouterr().out
    assert "7 missing key(s)" in out and "(+2 more)" in out
    assert "1 unexpected key(s)" in out


# 6.6: decoupled modes encode the prompts once per evaluation.
def test_eval_text_features_cached(biomedclip, r2_models):
    n_cls = len(CLASSNAMES)
    for model in r2_models:  # MODE "none" and "image"
        model.eval()
        model._text_cache = None
        batches = [images(3, seed=s) for s in range(3)]
        with torch.no_grad():
            with count_text_sequences(biomedclip) as sizes:
                cached = [model(x) for x in batches]
            assert sizes == [n_cls], model.vgap_mode
            fresh_text = model._encode_text(model.ctx_causal)
            torch.testing.assert_close(
                model._text_cache[1], fresh_text, rtol=0, atol=0
            )

            # In-place prompt changes (optimizer step, load_state_dict) invalidate it.
            state = {k: v.clone() for k, v in model.state_dict().items()}
            # (a constant shift would be undone by the embedding LayerNorm)
            generator = torch.Generator().manual_seed(0)
            noise = torch.randn(model.ctx_causal.shape, generator=generator)
            model.ctx_causal.add_(0.1 * noise)
            with count_text_sequences(biomedclip) as sizes:
                changed = model(batches[0])
            assert sizes == [n_cls]
            assert not torch.allclose(changed, cached[0])
            model.load_state_dict(state)
            with count_text_sequences(biomedclip) as sizes:
                restored = model(batches[0])
            assert sizes == [n_cls]
            torch.testing.assert_close(restored, cached[0], rtol=0, atol=0)

        train_step(model, images(1), torch.tensor([0]))
        assert model._text_cache is None


def test_text_mode_is_not_cached(biomedclip, models):
    _, text_model = models
    text_model.eval()
    with torch.no_grad(), count_text_sequences(biomedclip) as sizes:
        for seed in range(2):
            text_model(images(2, seed=seed))
    assert sizes == [2 * len(CLASSNAMES)] * 2  # image-conditioned text features


# 6.9: readout attention on fixed test images.
def test_fixed_image_indices():
    from trainers.biomedccpl import fixed_image_indices

    indices = fixed_image_indices(1000)
    assert indices == fixed_image_indices(1000) and len(set(indices)) == 20
    assert indices == sorted(indices) and all(0 <= i < 1000 for i in indices)
    assert fixed_image_indices(5) == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("layers", [[11], [3, 7, 11]])
def test_readout_attention_maps(layers):
    from trainers.biomedccpl import ImageSideReadout

    readout = ImageSideReadout(layers)
    hidden = {layer: torch.randn(2, 196, 768) for layer in layers}
    cls_token = torch.randn(2, 768)
    maps = readout.attention_maps(hidden, cls_token)
    n_proto = 14 * len(layers)
    assert maps["prototype_maps"].shape == (2, n_proto, 14, 14)
    assert maps["readout_attn"].shape == (2, n_proto)
    torch.testing.assert_close(
        maps["prototype_maps"].sum(dim=(-2, -1)), torch.ones(2, n_proto)
    )
    torch.testing.assert_close(maps["readout_attn"].sum(-1), torch.ones(2))

    # The same attention the forward pass logs the entropy of.
    _, logs = readout(hidden, cls_token, torch.randn(2, 512))
    attn = maps["readout_attn"]
    entropy = -(attn * attn.log()).sum(-1).mean()
    torch.testing.assert_close(entropy, logs["attn_entropy"])


def test_trainer_saves_readout_attention(tmp_path, r2_models):
    from trainers.biomedccpl import BIOMEDCCPL_BiomedCLIP

    _, model = r2_models
    dataset = [
        {"img": images(1, seed=i)[0], "label": i % 4, "impath": f"img{i}.jpg"}
        for i in range(25)
    ]
    trainer = BIOMEDCCPL_BiomedCLIP.__new__(BIOMEDCCPL_BiomedCLIP)
    trainer.model = model
    trainer.test_loader = types.SimpleNamespace(dataset=dataset)
    trainer.device = torch.device("cpu")
    trainer.output_dir = str(tmp_path)

    path = trainer.save_readout_attention()
    record = torch.load(path)
    assert len(record["indices"]) == len(record["impaths"]) == 20
    assert record["impaths"] == [f"img{i}.jpg" for i in record["indices"]]
    assert record["prototype_maps"].shape == (20, 14, 14, 14)
    assert record["readout_attn"].shape == (20, 14)
    assert record["preds"].shape == record["labels"].shape == (20,)
    assert record["layers"] == [11] and record["beta"] == 0.0
