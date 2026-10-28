"""Config defaults for the BiomedCCPL trainer."""

from yacs.config import CfgNode as CN

# ADR-0001: where the VGAP prototype readout lives.
#   "text"          R0, original: VGAP injected into both text branches
#   "none"          R1, no VGAP
#   "image"         R2/R3, class-agnostic readout on the image side
#   "text_agnostic" R4, text-side VGAP with a shared (class-agnostic) query
VGAP_MODES = ("text", "none", "image", "text_agnostic")


def add_biomedccpl_cfg(cfg):
    cfg.TRAINER.BIOMEDCCPL = CN()
    cfg.TRAINER.BIOMEDCCPL.PREC = "fp32"
    cfg.TRAINER.BIOMEDCCPL.ALPHA = 1.0
    cfg.TRAINER.BIOMEDCCPL.PROTONUM = 14
    cfg.TRAINER.BIOMEDCCPL.LAMBDA_NEM = 0.1
    cfg.TRAINER.BIOMEDCCPL.LAMBDA_ORTHO = 0.01
    cfg.TRAINER.BIOMEDCCPL.CROSSLAYERS = [3, 7, 11]

    cfg.TRAINER.BIOMEDCCPL.VGAP = CN()
    cfg.TRAINER.BIOMEDCCPL.VGAP.MODE = "text"
    cfg.TRAINER.BIOMEDCCPL.VGAP.IMAGE_LAYERS = [11]  # R2: [11]; R3: [3, 7, 11]
    cfg.TRAINER.BIOMEDCCPL.VGAP.READOUT_HEADS = 12  # same as the original cross-attention
    cfg.TRAINER.BIOMEDCCPL.VGAP.READOUT_DROPOUT = 0.0
    cfg.TRAINER.BIOMEDCCPL.VGAP.GATE_INIT = 0.0  # g_0; beta = tanh(g)
    cfg.TRAINER.BIOMEDCCPL.VGAP.READOUT_LR_MULT = 1.0
