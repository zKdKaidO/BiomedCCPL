"""BiomedCCPL model: VGAP, SCD, and semantic anchor regularization."""

import copy
import math
import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.metrics import compute_accuracy
from trainers.config import VGAP_MODES
from trainers.prompt_templates import CUSTOM_TEMPLATES
from open_clip import create_model_from_pretrained, get_tokenizer

# Substrings of the parameter names that are trained; everything else is frozen.
TRAINABLE_PARAM_NAMES = [
    "ctx_causal",
    "ctx_non_causal",
    "prototype_learner",
    "crossattention_layer",
    "image_readout",
    "vgap_query",
]


def count_trainable_params(model):
    """Trainable parameters per top-level submodule (ADR-0001 test 6.8.7)."""
    counts = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            top = name.split(".")[0]
            counts[top] = counts.get(top, 0) + param.numel()
    return counts


def build_param_groups(model, base_lr, readout_lr_mult):
    """Optimizer param groups; the image readout trains at base_lr * READOUT_LR_MULT.

    Returns None outside VGAP.MODE "image", so Dassl optimizes
    model.parameters() exactly as the original trainer did.
    """
    readout = getattr(model, "image_readout", None)
    if readout is None:
        return None
    readout_params = [p for p in readout.parameters() if p.requires_grad]
    readout_ids = {id(p) for p in readout_params}
    other_params = [
        p for p in model.parameters() if p.requires_grad and id(p) not in readout_ids
    ]
    return [
        {"params": other_params},
        {"params": readout_params, "lr": base_lr * readout_lr_mult},
    ]


def warn_on_key_mismatch(load_result, model_path, max_names=5):
    """load_state_dict(strict=False) is silent; a checkpoint from another
    VGAP.MODE would otherwise load with its trained modules missing."""
    for kind, keys in (
        ("missing", load_result.missing_keys),
        ("unexpected", load_result.unexpected_keys),
    ):
        if keys:
            shown = ", ".join(keys[:max_names])
            more = f" (+{len(keys) - max_names} more)" if len(keys) > max_names else ""
            print(
                f"WARNING: {len(keys)} {kind} key(s) when loading '{model_path}': "
                f"{shown}{more}. Was it trained with a different VGAP.MODE?"
            )


def fixed_image_indices(n_items, n_images=20, seed=0):
    """The same `n_images` test images for every run on a dataset (ADR 6.9)."""
    generator = torch.Generator().manual_seed(seed)
    return sorted(torch.randperm(n_items, generator=generator)[:n_images].tolist())


class VGAPProbe:
    """Inference-only interventions on the text-side VGAP (ADR-0001 Phase 0).

    Set `CustomCLIP.vgap_probe` to an instance; `None` (the default) leaves the
    forward pass untouched. At the ViT layers in `layers`, `intervention`
    replaces the class feature alpha * t_c + (1 - alpha) * CrossAttn(t_c, P):
      "text_only_alpha"  alpha * t_c       (readout contribution removed)
      "text_only"        t_c               (removed and not scaled by alpha)
      "class_mean"       the readout averaged over classes, per image
    Every module's attention over the prototypes, (B, ncls, protonum), is
    stored in `attn` under (branch, layer). `non_causal` also runs the
    non-causal branch at test time so its attention can be recorded.
    """

    INTERVENTIONS = (None, "text_only_alpha", "text_only", "class_mean")

    def __init__(self, intervention=None, layers=(), non_causal=False):
        if intervention not in self.INTERVENTIONS:
            raise ValueError(f"intervention must be one of {self.INTERVENTIONS}")
        self.intervention = intervention
        self.layers = set(layers)
        self.non_causal = non_causal
        self.attn = {}

    def mix(self, branch, layer, alpha, cls_feature, readout, attn):
        self.attn[(branch, layer)] = attn.detach()
        if self.intervention is None or layer not in self.layers:
            return alpha * cls_feature + (1 - alpha) * readout
        if self.intervention == "text_only_alpha":
            return alpha * cls_feature
        if self.intervention == "text_only":
            return cls_feature
        readout = readout.mean(dim=1, keepdim=True).expand_as(readout)
        return alpha * cls_feature + (1 - alpha) * readout


class LayerNorm(nn.LayerNorm):
    """Compute LayerNorm in fp32 for stable mixed-precision training."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


def build_prototype_learner(width, protonum):
    # B, 196, width -> B, 196, protonum
    return nn.Sequential(
        nn.Linear(width, 256, bias=False),
        LayerNorm(256),
        nn.ReLU(inplace=True),
        nn.Linear(256, protonum, bias=False),
    )


def compute_prototypes(prototype_learner, patches):
    """VGAP prototypes: softmax over patches, then a weighted sum of patches."""
    w_score = prototype_learner(patches)
    w_score_softmax = F.softmax(w_score, dim=1)
    # B, protonum, 196 @ B, 196, width -> B, protonum, width
    return torch.matmul(w_score_softmax.transpose(1, 2), patches)


class ImageSideReadout(nn.Module):
    """Class-agnostic prototype readout on the image side (ADR-0001, R2/R3).

    F_I' = norm(F_I + tanh(g) * W_o CrossAttn(q = [CLS], K = V = prototypes)).
    With g = 0 at init the model starts out identical to VGAP.MODE "none".
    """

    def __init__(
        self,
        layers,
        width=768,
        embed_dim=512,
        protonum=14,
        heads=12,
        dropout=0.0,
        gate_init=0.0,
    ):
        super().__init__()
        self.layers = sorted(layers)
        # One prototype learner per layer, defined exactly as in VGAP.
        self.proto = nn.ModuleDict(
            {str(layer): build_prototype_learner(width, protonum) for layer in self.layers}
        )
        # Tells the prototypes of different layers apart (R3 only).
        self.layer_emb = (
            nn.Parameter(torch.zeros(len(self.layers), 1, width))
            if len(self.layers) > 1
            else None
        )
        self.ln_q = LayerNorm(width)
        self.ln_kv = LayerNorm(width)
        self.attn = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.W_o = nn.Linear(width, embed_dim, bias=False)
        self.g = nn.Parameter(torch.tensor(float(gate_init)))  # beta = tanh(g)

    def prototypes(self, hidden):
        prototypes = []
        for i, layer in enumerate(self.layers):
            prototype = compute_prototypes(self.proto[str(layer)], hidden[layer])
            if self.layer_emb is not None:
                prototype = prototype + self.layer_emb[i]
            prototypes.append(prototype)
        return torch.cat(prototypes, dim=1)  # B, protonum * len(layers), width

    def forward(self, hidden, cls_token, image_features):
        """
        hidden: {layer: (B, 196, width)} patch tokens; cls_token: (B, width)
        [CLS] state of the deepest layer; image_features: (B, embed_dim),
        projected but not normalized. Returns normalized (B, embed_dim) features.
        """
        kv = self.ln_kv(self.prototypes(hidden))
        query = self.ln_q(cls_token).unsqueeze(1)  # B, 1, width
        readout, attn_weights = self.attn(query, kv, kv, need_weights=True)
        readout = self.W_o(readout.squeeze(1))  # B, embed_dim

        beta = torch.tanh(self.g)
        features = F.normalize(image_features + beta * readout, dim=-1)

        with torch.no_grad():
            attn = attn_weights.squeeze(1)  # B, protonum * len(layers)
            logs = {
                "beta": beta.detach(),
                "r_norm_ratio": (
                    readout.norm(dim=-1) / image_features.norm(dim=-1)
                ).mean(),
                "attn_entropy": -(attn * attn.clamp_min(1e-12).log()).sum(-1).mean(),
            }
        return features, logs

    @torch.no_grad()
    def attention_maps(self, hidden, cls_token):
        """Maps for visualization: each prototype's softmax over the patch
        grid, (B, protonum * len(layers), 14, 14), and the readout attention
        over the prototypes, (B, protonum * len(layers))."""
        maps = torch.cat(
            [
                F.softmax(self.proto[str(layer)](hidden[layer]), dim=1).transpose(1, 2)
                for layer in self.layers
            ],
            dim=1,
        )
        side = math.isqrt(maps.shape[-1])
        kv = self.ln_kv(self.prototypes(hidden))
        query = self.ln_q(cls_token).unsqueeze(1)
        _, attn_weights = self.attn(query, kv, kv, need_weights=True)
        return {
            "prototype_maps": maps.reshape(maps.shape[0], maps.shape[1], side, side),
            "readout_attn": attn_weights.squeeze(1),
        }


class CustomCLIP(nn.Module):
    """BiomedCLIP with image-conditional causal/non-causal prompt pathways."""

    def __init__(self, cfg, classnames, biomedclip_model):
        super().__init__()
        self.cfg = cfg
        self.n_cls = len(classnames)
        self.n_ctx = 4
        self.biomedclip_model = biomedclip_model
        self.dtype = biomedclip_model.text.transformer.dtype
        self.logit_scale = biomedclip_model.logit_scale

        self.alpha = cfg.TRAINER.BIOMEDCCPL.get("ALPHA", 0.01)
        self.protonum = cfg.TRAINER.BIOMEDCCPL.get("PROTONUM", 14)
        self.lambda_nem = cfg.TRAINER.BIOMEDCCPL.get("LAMBDA_NEM", 0.1)
        self.lambda_ortho = cfg.TRAINER.BIOMEDCCPL.get("LAMBDA_ORTHO", 0.01)
        self.crosslayers = cfg.TRAINER.BIOMEDCCPL.get("CROSSLAYERS", [3, 7, 11])

        vgap_cfg = cfg.TRAINER.BIOMEDCCPL.get("VGAP", {})
        self.vgap_mode = vgap_cfg.get("MODE", "text")
        if self.vgap_mode not in VGAP_MODES:
            raise ValueError(
                f"VGAP.MODE must be one of {VGAP_MODES}, got {self.vgap_mode!r}"
            )
        self.image_layers = sorted(vgap_cfg.get("IMAGE_LAYERS", [11]))
        n_blocks = len(biomedclip_model.visual.trunk.blocks)
        if self.vgap_mode == "image" and (
            not self.image_layers
            or not all(0 <= layer < n_blocks for layer in self.image_layers)
        ):
            raise ValueError(
                f"VGAP.IMAGE_LAYERS must be ViT block indices in [0, {n_blocks}), "
                f"got {self.image_layers}"
            )
        # Detached diagnostics from the last forward pass, logged by the trainer.
        self.logs = {}
        # Test-time causal text features of the decoupled modes (ADR-0001 6.6).
        self._text_cache = None
        # Phase 0 diagnostics only; see VGAPProbe.
        self.vgap_probe = None

        clip_imsize = 224
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert (
            cfg_imsize == clip_imsize
        ), f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        with torch.no_grad():
            device = self.biomedclip_model.logit_scale.device
            tokenizer = get_tokenizer(
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            )
            anchor_model = self.biomedclip_model.float().eval().to(device)

            clean_classnames = [name.replace("_", " ") for name in classnames]

            hand_template = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
            tokenized_prompts = torch.cat(
                [tokenizer(hand_template.format(c)) for c in clean_classnames]
            ).to(device)
            hand_feas = anchor_model.encode_text(tokenized_prompts)
            # Anchors are regenerated from the active dataset class names.
            self.register_buffer(
                "hand_feas",
                hand_feas / hand_feas.norm(dim=-1, keepdim=True),
                persistent=False,
            )

            l_tokenized_prompts = torch.cat(
                [tokenizer(hand_template.format(c)) for c in clean_classnames]
            ).to(device)
            self.register_buffer(
                "l_tokenized_prompts", l_tokenized_prompts, persistent=False
            )
            x = anchor_model.text.transformer.embeddings.word_embeddings(
                self.l_tokenized_prompts
            )
            self.register_buffer("prefix", x[:, :1, :], persistent=False)
            ctx_vectors = x[0, 1 : 1 + self.n_ctx, :].clone().detach()
            self.register_buffer("suffix", x[:, 1 + self.n_ctx :, :], persistent=False)

        self.ctx_causal = nn.Parameter(ctx_vectors.clone())
        noise = torch.randn_like(ctx_vectors) * 0.01
        self.ctx_non_causal = nn.Parameter(ctx_vectors.clone() + noise)

        # Text-side VGAP modules; built after the prompts so that every MODE
        # starts from the same prompt initialization for a given seed.
        if self.vgap_mode in ("text", "text_agnostic"):
            self._build_text_vgap()
        if self.vgap_mode == "text_agnostic":
            # R4: one shared query per injection layer and branch replaces the
            # class [CLS] token, so every class reads the prototypes the same way.
            n_layers = len(self.crosslayers)
            self.vgap_query = nn.Parameter(torch.randn(n_layers, 1, 768) * 0.02)
            self.vgap_query_nc = nn.Parameter(torch.randn(n_layers, 1, 768) * 0.02)
        elif self.vgap_mode == "image":
            self.image_readout = ImageSideReadout(
                self.image_layers,
                width=biomedclip_model.visual.trunk.embed_dim,
                embed_dim=biomedclip_model.visual.head.proj.out_features,
                protonum=self.protonum,
                heads=vgap_cfg.get("READOUT_HEADS", 12),
                dropout=vgap_cfg.get("READOUT_DROPOUT", 0.0),
                gate_init=vgap_cfg.get("GATE_INIT", 0.0),
            )

    def _build_text_vgap(self):
        single_prototype_learner = build_prototype_learner(768, self.protonum)
        self.prototype_learner = nn.ModuleList(
            [
                copy.deepcopy(single_prototype_learner)
                for _ in range(len(self.crosslayers))
            ]
        )
        single_crossattention_layer = nn.MultiheadAttention(
            embed_dim=768, num_heads=768 // 64, batch_first=True
        )
        self.crossattention_layer = nn.ModuleList(
            [
                copy.deepcopy(single_crossattention_layer)
                for _ in range(len(self.crosslayers))
            ]
        )
        self.crossattention_layer_nc = nn.ModuleList(
            [
                copy.deepcopy(single_crossattention_layer)
                for _ in range(len(self.crosslayers))
            ]
        )

    def forward(self, image, label=None):
        self.logs = {}
        if self.vgap_mode in ("text", "text_agnostic"):
            return self._forward_text_vgap(image, label)
        return self._forward_decoupled(image, label)

    def _forward_text_vgap(self, image, label=None):
        """Original BiomedCCPL forward: VGAP injected into both text branches.

        MODE "text_agnostic" (R4) differs only in the cross-attention query.
        """
        logit_scale = self.logit_scale.exp()
        batch_size = image.shape[0]
        probe = self.vgap_probe
        run_non_causal = self.training or (probe is not None and probe.non_causal)

        content_causal = self.ctx_causal.unsqueeze(0).expand(self.n_cls, -1, -1)
        word_emb_causal = torch.cat([self.prefix, content_causal, self.suffix], dim=1)
        text_emb_causal = self._get_text_embedding_base(word_emb_causal)
        seq_len = text_emb_causal.shape[1]
        text_emb_causal = (
            text_emb_causal.unsqueeze(0)
            .expand(batch_size, -1, -1, -1)
            .reshape(batch_size * self.n_cls, seq_len, -1)
        )

        if run_non_causal:
            content_non_causal = self.ctx_non_causal.unsqueeze(0).expand(
                self.n_cls, -1, -1
            )
            word_emb_non_causal = torch.cat(
                [self.prefix, content_non_causal, self.suffix], dim=1
            )
            text_emb_non_causal = self._get_text_embedding_base(word_emb_non_causal)
            text_emb_non_causal = (
                text_emb_non_causal.unsqueeze(0)
                .expand(batch_size, -1, -1, -1)
                .reshape(batch_size * self.n_cls, seq_len, -1)
            )

        attention_mask_after = (self.l_tokenized_prompts != 0).long().to(image.device)
        attention_mask_after = attention_mask_after.repeat(batch_size, 1)
        extended_attention_mask = self._get_attention_mask(attention_mask_after)

        image_emb = self.biomedclip_model.visual.trunk.patch_embed(image)
        image_emb = torch.cat(
            (
                self.biomedclip_model.visual.trunk.cls_token.expand(
                    image_emb.shape[0], -1, -1
                ),
                image_emb,
            ),
            dim=1,
        )
        image_emb = image_emb + self.biomedclip_model.visual.trunk.pos_embed

        counter = 0
        for i in range(12):
            text_emb_causal = self.biomedclip_model.text.transformer.encoder.layer[i](
                text_emb_causal, attention_mask=extended_attention_mask
            )[0]
            if run_non_causal:
                text_emb_non_causal = (
                    self.biomedclip_model.text.transformer.encoder.layer[i](
                        text_emb_non_causal, attention_mask=extended_attention_mask
                    )[0]
                )
            image_emb = self.biomedclip_model.visual.trunk.blocks[i](image_emb)
            if i in self.crosslayers:
                prototype = compute_prototypes(
                    self.prototype_learner[counter], image_emb[:, 1:, :]
                )

                text_emb_causal_batch = text_emb_causal.reshape(
                    batch_size, self.n_cls, seq_len, -1
                )

                crossatt_out_c, attn_c = self.crossattention_layer[counter](
                    query=self._vgap_query(
                        "vgap_query", counter, text_emb_causal_batch
                    ),  # (B, ncls, 768)
                    key=prototype,  # (B, protonum, 768)
                    value=prototype,
                )
                if probe is None:
                    cau_class_feature = (
                        self.alpha * text_emb_causal_batch[:, :, 0, :]
                        + (1 - self.alpha) * crossatt_out_c
                    )
                else:
                    cau_class_feature = probe.mix(
                        "causal", i, self.alpha,
                        text_emb_causal_batch[:, :, 0, :], crossatt_out_c, attn_c,
                    )
                text_emb_causal = torch.cat(
                    [
                        cau_class_feature.unsqueeze(2),
                        text_emb_causal_batch[:, :, 1:, :],
                    ],
                    dim=2,
                ).reshape(batch_size * self.n_cls, seq_len, -1)

                if run_non_causal:
                    text_emb_non_causal_batch = text_emb_non_causal.reshape(
                        batch_size, self.n_cls, seq_len, -1
                    )
                    crossatt_out_u, attn_u = self.crossattention_layer_nc[counter](
                        query=self._vgap_query(
                            "vgap_query_nc", counter, text_emb_non_causal_batch
                        ),
                        key=prototype,
                        value=prototype,
                    )
                    if probe is None:
                        adv_class_feature = (
                            self.alpha * text_emb_non_causal_batch[:, :, 0, :]
                            + (1 - self.alpha) * crossatt_out_u
                        )
                    else:
                        adv_class_feature = probe.mix(
                            "non_causal", i, self.alpha,
                            text_emb_non_causal_batch[:, :, 0, :], crossatt_out_u,
                            attn_u,
                        )
                    text_emb_non_causal = torch.cat(
                        [
                            adv_class_feature.unsqueeze(2),
                            text_emb_non_causal_batch[:, :, 1:, :],
                        ],
                        dim=2,
                    ).reshape(batch_size * self.n_cls, seq_len, -1)

                counter += 1

        cau_text_feature = self._get_final_text_features(text_emb_causal).reshape(
            batch_size, self.n_cls, -1
        )
        image_features = self._get_final_image_features(image_emb)
        logits = logit_scale * torch.einsum(
            "bd,bcd->bc", image_features, cau_text_feature
        )

        if self.training:
            adv_text_features = self._get_final_text_features(
                text_emb_non_causal
            ).reshape(batch_size, self.n_cls, -1)
            return self._compute_losses(
                logits, label, image_features, cau_text_feature, adv_text_features
            )
        else:
            return logits

    @torch.no_grad()
    def readout_attention(self, image):
        """MODE "image" only: see ImageSideReadout.attention_maps."""
        hidden, cls_token, _ = self._encode_image_keep(image, self.image_layers)
        return self.image_readout.attention_maps(hidden, cls_token)

    def _vgap_query(self, name, counter, text_emb_batch):
        """Cross-attention query: the class [CLS] states, or in R4 a shared
        learned vector broadcast to every class."""
        if self.vgap_mode == "text":
            return text_emb_batch[:, :, 0, :]
        batch_size, n_cls = text_emb_batch.shape[:2]
        return getattr(self, name)[counter].expand(batch_size, n_cls, -1)

    def _forward_decoupled(self, image, label=None):
        """VGAP.MODE "none" (R1) and "image" (R2/R3).

        Text features no longer depend on the image; in MODE "image" a single
        class-agnostic readout of the prototypes updates the image feature.
        """
        logit_scale = self.logit_scale.exp()

        readout = self.vgap_mode == "image"
        hidden, cls_token, image_features = self._encode_image_keep(
            image, self.image_layers if readout else ()
        )
        if readout:
            image_features, readout_logs = self.image_readout(
                hidden, cls_token, image_features
            )
            self.logs.update(readout_logs)
        else:
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True
            )

        cau_text_feature = self._causal_text_features()  # (ncls, D)
        logits = logit_scale * self._class_scores(image_features, cau_text_feature)

        if self.training:
            adv_text_features = self._encode_text(self.ctx_non_causal)
            return self._compute_losses(
                logits, label, image_features, cau_text_feature, adv_text_features
            )
        else:
            return logits

    def _causal_text_features(self):
        """Causal text features; at test time computed once and reused.

        They do not depend on the image outside MODE "text"/"text_agnostic",
        so evaluation encodes the prompts once instead of once per batch. The
        cache is keyed on the prompt tensor and its in-place version counter,
        so optimizer steps and load_state_dict invalidate it.
        """
        if self.training:
            self._text_cache = None
            return self._encode_text(self.ctx_causal)
        ctx = self.ctx_causal
        key = (ctx.data_ptr(), ctx._version, ctx.device, ctx.dtype)
        if self._text_cache is None or self._text_cache[0] != key:
            with torch.no_grad():
                self._text_cache = (key, self._encode_text(ctx))
        return self._text_cache[1]

    def _compute_losses(
        self, logits, label, image_features, cau_text_feature, adv_text_features
    ):
        # Keep the original operation order: MODE "text" must stay bit-identical,
        # and autograd's accumulation order depends on it.
        loss_ce = F.cross_entropy(logits, label)
        loss_sar = self._compute_sar_loss(cau_text_feature)

        ortho_matrix = torch.matmul(
            F.normalize(cau_text_feature, dim=-1),
            F.normalize(adv_text_features, dim=-1).transpose(-2, -1),
        )
        loss_ortho = torch.mean(ortho_matrix**2)

        logits_adv = self.logit_scale.exp() * self._class_scores(
            image_features, adv_text_features
        )
        log_probs_adv = F.log_softmax(logits_adv, dim=-1)
        uniform_dist = torch.full_like(log_probs_adv, 1.0 / self.n_cls)
        loss_nem = F.kl_div(
            log_probs_adv, uniform_dist, reduction="batchmean"
        )  # Make the non-causal pathway non-predictive.

        total_loss = (
            loss_ce
            + self.lambda_ortho * loss_ortho
            + self.lambda_nem * loss_nem
            + loss_sar
        )
        with torch.no_grad():
            self.logs["loss_sar_cos"] = self._compute_sar_cos(cau_text_feature)
        return logits, total_loss, loss_ce, loss_ortho, loss_nem, loss_sar

    @staticmethod
    def _class_scores(image_features, text_features):
        # Text features are (B, ncls, D) in MODE "text" and (ncls, D) otherwise.
        if text_features.dim() == 3:
            return torch.einsum("bd,bcd->bc", image_features, text_features)
        return torch.einsum("bd,cd->bc", image_features, text_features)

    def _encode_text(self, ctx):
        """Run the text encoder once on the ncls prompts built from `ctx`."""
        content = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        word_emb = torch.cat([self.prefix, content, self.suffix], dim=1)
        text_emb = self._get_text_embedding_base(word_emb)

        attention_mask = (self.l_tokenized_prompts != 0).long().to(word_emb.device)
        extended_attention_mask = self._get_attention_mask(attention_mask)
        for layer in self.biomedclip_model.text.transformer.encoder.layer:
            text_emb = layer(text_emb, attention_mask=extended_attention_mask)[0]
        return self._get_final_text_features(text_emb)

    def _encode_image_keep(self, image, layers=()):
        """Run the frozen ViT once, keeping the hidden states VGAP reads.

        Returns patch tokens of each layer in `layers` (block output, before
        trunk.norm), the [CLS] state of the deepest of them, and the projected,
        unnormalized image feature.
        """
        trunk = self.biomedclip_model.visual.trunk
        hidden, cls_token = {}, None
        with torch.no_grad():
            image_emb = trunk.patch_embed(image)
            image_emb = torch.cat(
                (trunk.cls_token.expand(image_emb.shape[0], -1, -1), image_emb),
                dim=1,
            )
            image_emb = image_emb + trunk.pos_embed
            for i, block in enumerate(trunk.blocks):
                image_emb = block(image_emb)
                if i in layers:
                    hidden[i] = image_emb[:, 1:, :]
                    cls_token = image_emb[:, 0, :]
            image_features = self._project_image(image_emb)
        return hidden, cls_token, image_features

    def _get_text_embedding_base(self, word_emb):
        ncls, seq_len, _ = word_emb.shape
        position_ids = (
            torch.arange(seq_len, dtype=torch.long, device=word_emb.device)
            .unsqueeze(0)
            .expand(ncls, -1)
        )
        position_emb = (
            self.biomedclip_model.text.transformer.embeddings.position_embeddings(
                position_ids
            )
        )
        token_type_ids = torch.zeros(
            ncls, seq_len, dtype=torch.long, device=word_emb.device
        )
        token_type_emb = (
            self.biomedclip_model.text.transformer.embeddings.token_type_embeddings(
                token_type_ids
            )
        )
        embeddings = word_emb + position_emb + token_type_emb
        text_emb = self.biomedclip_model.text.transformer.embeddings.LayerNorm(
            embeddings
        )
        return self.biomedclip_model.text.transformer.embeddings.dropout(text_emb)

    def _get_attention_mask(self, attention_mask):
        extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_mask = extended_mask.to(dtype=self.dtype)
        extended_mask = (1.0 - extended_mask) * -10000.0
        return extended_mask

    def _get_final_text_features(self, text_emb):
        cls_features = text_emb[:, 0]
        text_features_proj = self.biomedclip_model.text.proj(cls_features)
        return text_features_proj / text_features_proj.norm(dim=-1, keepdim=True)

    def _get_final_image_features(self, image_emb):
        image_features_proj = self._project_image(image_emb)
        return image_features_proj / image_features_proj.norm(dim=-1, keepdim=True)

    def _project_image(self, image_emb):
        image_emb_norm = self.biomedclip_model.visual.trunk.norm(image_emb)
        return self.biomedclip_model.visual.head.proj(image_emb_norm[:, 0])

    def _compute_sar_loss(self, text_features):
        target = self.hand_feas
        if text_features.dim() == 3:
            target = target.unsqueeze(0).expand_as(text_features)
        return F.mse_loss(text_features, target)

    def _compute_sar_cos(self, text_features):
        # The paper's form, 1 - mean cosine to the anchors. For unit vectors the
        # element-wise MSE above equals (2 / D) times this (ADR-0001, 1.1 item 5).
        target = self.hand_feas
        if text_features.dim() == 3:
            target = target.unsqueeze(0).expand_as(text_features)
        return 1.0 - (text_features * target).sum(dim=-1).mean()


@TRAINER_REGISTRY.register()
class BIOMEDCCPL_BiomedCLIP(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.BIOMEDCCPL.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading BiomedCLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        biomedclip_model, preprocess = create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        if (
            cfg.TRAINER.BIOMEDCCPL.PREC == "fp32"
            or cfg.TRAINER.BIOMEDCCPL.PREC == "amp"
        ):
            biomedclip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, biomedclip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            if any(re_name in name for re_name in TRAINABLE_PARAM_NAMES):
                param.requires_grad_(True)
        enabled = {
            name for name, param in self.model.named_parameters() if param.requires_grad
        }
        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}")
        param_counts = count_trainable_params(self.model)
        print(
            f"Trainable parameters (VGAP.MODE {self.model.vgap_mode!r}): "
            f"{sum(param_counts.values()):,} {param_counts}"
        )

        self.model.to(self.device)
        param_groups = build_param_groups(
            self.model,
            cfg.OPTIM.LR,
            cfg.TRAINER.BIOMEDCCPL.VGAP.READOUT_LR_MULT,
        )
        self.optim = build_optimizer(self.model, cfg.OPTIM, param_groups=param_groups)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("causal_model", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.BIOMEDCCPL.PREC == "amp" else None

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        model = self.model
        optim = self.optim
        scaler = self.scaler
        prec = self.cfg.TRAINER.BIOMEDCCPL.PREC

        if prec == "amp":
            with autocast():
                logits, total_loss, loss_ce, loss_ortho, loss_nem, loss_sar = model(
                    image, label
                )
            optim.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            logits, total_loss, loss_ce, loss_ortho, loss_nem, loss_sar = model(
                image, label
            )
            self.model_backward_and_update(total_loss)

        loss_summary = {
            "total_loss": total_loss.item(),
            "loss_ce": loss_ce.item(),
            "loss_ortho": loss_ortho.item(),
            "loss_nem": loss_nem.item(),
            "loss_sar": loss_sar.item(),
            "acc": compute_accuracy(logits, label)[0].item(),
        }
        loss_summary.update({name: value.item() for name, value in model.logs.items()})

        # Step the epoch-level learning-rate scheduler after the final batch.
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def after_train(self):
        super().after_train()
        if self.model.vgap_mode == "image":
            self.save_readout_attention()

    @torch.no_grad()
    def save_readout_attention(self, n_images=20, path=None):
        """Save the readout attention on fixed test images (ADR-0001 6.9)."""
        dataset = self.test_loader.dataset
        indices = fixed_image_indices(len(dataset), n_images)
        items = [dataset[i] for i in indices]
        image = torch.stack([item["img"] for item in items]).to(self.device)

        self.model.eval()
        maps = self.model.readout_attention(image)
        logits = self.model(image)
        record = {
            "indices": indices,
            "impaths": [item["impath"] for item in items],
            "labels": torch.tensor([item["label"] for item in items]),
            "preds": logits.argmax(dim=-1).cpu(),
            "layers": list(self.model.image_layers),
            "protonum": self.model.protonum,
            "beta": torch.tanh(self.model.image_readout.g).item(),
            **{name: value.cpu() for name, value in maps.items()},
        }
        path = path or osp.join(self.output_dir, "readout_attention.pt")
        torch.save(record, path)
        print(f"Saved readout attention for {len(indices)} test images to {path}")
        return path

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar"
        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            print(f"Loading weights to {name} from '{model_path}' (epoch = {epoch})")

            result = self._models[name].load_state_dict(state_dict, strict=False)
            warn_on_key_mismatch(result, model_path)
