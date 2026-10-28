"""ADR-0001 tests for the VGAP.MODE switch (section 6.8)."""

import pytest
import torch

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
