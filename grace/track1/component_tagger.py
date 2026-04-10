"""Track 1 component tagger - BIO token classification over Premise/Claim/MajorClaim.

Uses a HuggingFace ``AutoModelForTokenClassification`` with 7 output classes
(O + B/I x 3 entity types) and a configurable loss function:

- **CE** (default): class-weighted cross-entropy
- **Focal**: focal loss (Lin et al. 2017) that down-weights easy O-tokens
  and focuses on hard MajorClaim/Attack tokens

Optional architectural upgrades:
- **CRF**: linear-chain CRF layer enforcing valid BIO transitions
- **BiLSTM**: bidirectional LSTM between encoder and classifier/CRF

Inference uses sliding-window decoding with stride=128 and mean-logit
pooling across overlap regions so long abstracts (up to ~1400 tokens) are
not silently truncated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.nn import CrossEntropyLoss
from torch.nn import functional as torch_f
from transformers import AutoModel, AutoModelForTokenClassification, AutoTokenizer

from grace.io.offsets import SpanAligner
from grace.io.schema import GraceCase

if TYPE_CHECKING:
    from collections.abc import Sequence


_LABELS: tuple[str, ...] = (
    "O",
    "B-Premise",
    "I-Premise",
    "B-Claim",
    "I-Claim",
    "B-MajorClaim",
    "I-MajorClaim",
)
_LABEL2ID = {lbl: i for i, lbl in enumerate(_LABELS)}


# ── Focal Loss ──────────────────────────────────────────────────────


class FocalLoss(torch.nn.Module):
    """Focal loss for class-imbalanced token classification (Lin et al. 2017).

    Down-weights well-classified easy examples (O tokens) and focuses the
    model on hard, misclassified rare entities (MajorClaim, Attack).

    With gamma=0 this degenerates to weighted cross-entropy.
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        gamma: float = 2.0,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = torch_f.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


# ── Config ──────────────────────────────────────────────────────────


@dataclass(slots=True)
class ComponentTaggerConfig:
    """Configuration for :class:`ComponentTagger`."""

    backbone: str = "xlm-roberta-base"
    max_length: int = 512
    stride: int = 128
    class_weights: dict[str, float] | None = None
    # Loss type: "ce" (cross-entropy) or "focal"
    loss_type: str = "ce"
    focal_gamma: float = 2.0
    # Optional CRF layer for valid BIO sequence enforcement
    use_crf: bool = False
    # Optional BiLSTM between encoder and classifier/CRF
    use_bilstm: bool = False
    bilstm_hidden: int = 256


# ── Tagger ──────────────────────────────────────────────────────────


class ComponentTagger:
    """Fine-tunes a HuggingFace token classification model for Track 1 BIO tagging.

    Supports optional CRF and BiLSTM layers on top of the encoder.
    When CRF or BiLSTM is enabled, we use AutoModel (raw encoder) instead
    of AutoModelForTokenClassification, and build our own classification head.
    """

    num_labels = len(_LABELS)

    def __init__(self, cfg: ComponentTaggerConfig) -> None:
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.backbone, use_fast=True)
        self.aligner = SpanAligner(hf_tokenizer=self.tokenizer)

        if cfg.use_crf or cfg.use_bilstm:
            # Use raw encoder + custom head when CRF/BiLSTM is enabled
            self.encoder = AutoModel.from_pretrained(cfg.backbone)
            hidden_size = self.encoder.config.hidden_size

            if cfg.use_bilstm:
                self.bilstm = torch.nn.LSTM(
                    hidden_size,
                    cfg.bilstm_hidden,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )
                classifier_input = cfg.bilstm_hidden * 2
            else:
                self.bilstm = None
                classifier_input = hidden_size

            self.classifier = torch.nn.Linear(classifier_input, self.num_labels)
            self.dropout = torch.nn.Dropout(0.1)

            if cfg.use_crf:
                from torchcrf import CRF

                self.crf = CRF(self.num_labels, batch_first=True)
            else:
                self.crf = None

            # Unified model reference for optimizer param collection
            self.model = self.encoder
            self._custom_head = True
        else:
            # Standard HF token classification model
            self.model = AutoModelForTokenClassification.from_pretrained(
                cfg.backbone,
                num_labels=self.num_labels,
                id2label=dict(enumerate(_LABELS)),
                label2id=_LABEL2ID,
            )
            self.encoder = None
            self.bilstm = None
            self.crf = None
            self.classifier = None
            self.dropout = None
            self._custom_head = False

    def parameters(self) -> list[torch.nn.Parameter]:
        """All trainable parameters (encoder + custom head components)."""
        params = list(self.model.parameters())
        if self.bilstm is not None:
            params.extend(self.bilstm.parameters())
        if self.classifier is not None:
            params.extend(self.classifier.parameters())
        if self.crf is not None:
            params.extend(self.crf.parameters())
        return params

    def to(self, device: torch.device) -> ComponentTagger:
        """Move all components to device."""
        self.model.to(device)
        if self.bilstm is not None:
            self.bilstm.to(device)
        if self.classifier is not None:
            self.classifier.to(device)
        if self.crf is not None:
            self.crf.to(device)
        return self

    def _get_emissions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get logits/emissions from the model (works for both modes)."""
        if self._custom_head:
            hidden = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state

            if self.bilstm is not None:
                hidden, _ = self.bilstm(hidden)

            hidden = self.dropout(hidden)
            return self.classifier(hidden)
        else:
            return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    def _class_weights_tensor(self, device: torch.device) -> torch.Tensor:
        if not self.cfg.class_weights:
            return torch.ones(self.num_labels, device=device)
        weights = torch.ones(self.num_labels, device=device)
        for i, lbl in enumerate(_LABELS):
            if lbl == "O":
                weights[i] = self.cfg.class_weights.get("O", 1.0)
            else:
                etype = lbl.split("-", 1)[1]
                weights[i] = self.cfg.class_weights.get(etype, 1.0)
        return weights

    def _build_loss_fn(self, device: torch.device) -> CrossEntropyLoss | FocalLoss:
        """Build the token-level loss function based on config."""
        weights = self._class_weights_tensor(device)
        if self.cfg.loss_type == "focal":
            return FocalLoss(
                weight=weights,
                gamma=self.cfg.focal_gamma,
                ignore_index=-100,
            )
        return CrossEntropyLoss(weight=weights, ignore_index=-100)

    def train_step(
        self,
        cases: Sequence[GraceCase],
        device: torch.device,
    ) -> float:
        """Run one forward+backward pass over a batch of cases. Returns mean loss."""
        self._set_train(True)
        loss_fn = self._build_loss_fn(device)
        total_loss = 0.0

        for case in cases:
            enc = self.aligner.encode_with_labels(
                case.raw_text,
                case.entities,
                max_length=self.cfg.max_length,
            )
            input_ids = torch.tensor([enc["input_ids"]], device=device)
            attn = torch.tensor([enc["attention_mask"]], device=device)
            labels = torch.tensor([enc["labels"]], device=device)

            emissions = self._get_emissions(input_ids, attn)

            if self.crf is not None:
                # CRF loss: negative log-likelihood
                mask = labels != -100
                crf_labels = labels.clone()
                crf_labels[~mask] = 0  # CRF doesn't understand -100
                crf_loss = -self.crf(emissions, crf_labels, mask=mask, reduction="mean")

                # Optionally combine with token-level loss for dual signal
                token_loss = loss_fn(
                    emissions.view(-1, self.num_labels),
                    labels.view(-1),
                )
                loss = 0.5 * crf_loss + 0.5 * token_loss
            else:
                loss = loss_fn(
                    emissions.view(-1, self.num_labels),
                    labels.view(-1),
                )

            loss.backward()
            total_loss += float(loss.item())

        return total_loss / max(len(cases), 1)

    def _set_train(self, mode: bool) -> None:
        """Set training mode on all components."""
        self.model.train(mode)
        if self.bilstm is not None:
            self.bilstm.train(mode)
        if self.classifier is not None:
            self.classifier.train(mode)
        if self.crf is not None:
            self.crf.train(mode)

    @torch.no_grad()
    def predict(self, cases: Sequence[GraceCase]) -> tuple[GraceCase, ...]:
        """Sliding-window inference with mean-logit pooling across overlaps."""
        self._set_train(False)
        out: list[GraceCase] = []
        device = next(self.model.parameters()).device

        for case in cases:
            enc = self.tokenizer(
                case.raw_text,
                return_offsets_mapping=True,
                return_overflowing_tokens=True,
                stride=self.cfg.stride,
                truncation=True,
                max_length=self.cfg.max_length,
                padding=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            emissions = self._get_emissions(input_ids, attention_mask)

            # For CRF with single window, use Viterbi decode directly
            if self.crf is not None and emissions.shape[0] == 1:
                mask = attention_mask.bool()
                label_ids_nested = self.crf.decode(emissions, mask=mask)
                # Map back through offset_mapping
                label_ids_flat = label_ids_nested[0]
                offset_map = enc["offset_mapping"][0]

                ordered_keys: list[tuple[int, int]] = []
                ordered_labels: list[int] = []
                for pos, lbl_id in enumerate(label_ids_flat):
                    s = int(offset_map[pos][0])
                    t = int(offset_map[pos][1])
                    if s == 0 and t == 0:
                        continue
                    if int(attention_mask[0][pos]) == 0:
                        continue
                    ordered_keys.append((s, t))
                    ordered_labels.append(lbl_id)

                if not ordered_keys:
                    out.append(
                        GraceCase(
                            id=case.id,
                            raw_text=case.raw_text,
                            track=1,
                            entities=(),
                            relations=(),
                        )
                    )
                    continue

                entities = self.aligner.decode_bio_to_entities(
                    case.raw_text,
                    ordered_labels,
                    ordered_keys,
                )
                out.append(
                    GraceCase(
                        id=case.id,
                        raw_text=case.raw_text,
                        track=1,
                        entities=entities,
                        relations=(),
                    )
                )
                continue

            # Multi-window: aggregate logits per (char_start, char_end)
            logits = emissions
            summed: dict[tuple[int, int], torch.Tensor] = {}
            counts: dict[tuple[int, int], int] = {}
            num_windows, seq_len, _ = logits.shape
            for w in range(num_windows):
                for pos in range(seq_len):
                    s = int(enc["offset_mapping"][w][pos][0])
                    t = int(enc["offset_mapping"][w][pos][1])
                    if s == 0 and t == 0:
                        continue
                    if int(enc["attention_mask"][w][pos]) == 0:
                        continue
                    key = (s, t)
                    tok_logits = logits[w][pos].detach().cpu()
                    if key in summed:
                        summed[key] = summed[key] + tok_logits
                        counts[key] += 1
                    else:
                        summed[key] = tok_logits
                        counts[key] = 1

            if not summed:
                out.append(
                    GraceCase(
                        id=case.id,
                        raw_text=case.raw_text,
                        track=1,
                        entities=(),
                        relations=(),
                    )
                )
                continue

            sorted_keys = sorted(summed.keys())
            averaged = torch.stack([summed[k] / counts[k] for k in sorted_keys])

            # For CRF with multi-window, decode on averaged logits
            if self.crf is not None:
                avg_device = averaged.unsqueeze(0).to(device)
                mask = torch.ones(1, len(sorted_keys), dtype=torch.bool, device=device)
                label_ids = self.crf.decode(avg_device, mask=mask)[0]
            else:
                label_ids = averaged.argmax(dim=-1).tolist()

            offset_mapping = list(sorted_keys)

            entities = self.aligner.decode_bio_to_entities(
                case.raw_text,
                label_ids,
                offset_mapping,
            )
            out.append(
                GraceCase(
                    id=case.id,
                    raw_text=case.raw_text,
                    track=1,
                    entities=entities,
                    relations=(),
                )
            )
        return tuple(out)

    def save(self, save_dir: str) -> None:
        """Save the model + tokenizer + custom head to ``save_dir``."""
        import json
        from pathlib import Path

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        if self._custom_head:
            torch.save(self.classifier.state_dict(), save_path / "classifier.pt")
            if self.bilstm is not None:
                torch.save(self.bilstm.state_dict(), save_path / "bilstm.pt")
            if self.crf is not None:
                torch.save(self.crf.state_dict(), save_path / "crf.pt")

        # Save config for reconstruction
        cfg_dict = {
            "loss_type": self.cfg.loss_type,
            "focal_gamma": self.cfg.focal_gamma,
            "use_crf": self.cfg.use_crf,
            "use_bilstm": self.cfg.use_bilstm,
            "bilstm_hidden": self.cfg.bilstm_hidden,
        }
        (save_path / "tagger_config.json").write_text(json.dumps(cfg_dict), encoding="utf-8")

    @classmethod
    def load(cls, save_dir: str, cfg: ComponentTaggerConfig) -> ComponentTagger:
        """Load a previously-saved model into a fresh tagger instance."""
        from pathlib import Path

        save_path = Path(save_dir)
        tagger = cls(cfg)

        if cfg.use_crf or cfg.use_bilstm:
            tagger.encoder = AutoModel.from_pretrained(save_dir)
            tagger.model = tagger.encoder
            if tagger.classifier is not None and (save_path / "classifier.pt").exists():
                tagger.classifier.load_state_dict(
                    torch.load(save_path / "classifier.pt", weights_only=True)
                )
            if tagger.bilstm is not None and (save_path / "bilstm.pt").exists():
                tagger.bilstm.load_state_dict(
                    torch.load(save_path / "bilstm.pt", weights_only=True)
                )
            if tagger.crf is not None and (save_path / "crf.pt").exists():
                tagger.crf.load_state_dict(torch.load(save_path / "crf.pt", weights_only=True))
        else:
            tagger.model = AutoModelForTokenClassification.from_pretrained(save_dir)

        tagger.tokenizer = AutoTokenizer.from_pretrained(save_dir, use_fast=True)
        tagger.aligner = SpanAligner(hf_tokenizer=tagger.tokenizer)
        return tagger
