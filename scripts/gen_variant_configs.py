#!/usr/bin/env python3
"""Generate architecture variant configs from a base backbone config.

Given a backbone config (e.g., rigoberta2.yaml), generates focal, crf,
focal_crf, focal_crf_bilstm variants by merging architecture settings.

Usage:
    python scripts/gen_variant_configs.py configs/track1/rigoberta2.yaml
"""

import sys
from pathlib import Path

import yaml

VARIANTS = {
    "focal": {
        "subtask1_overrides": {
            "loss": "focal",
            "focal_gamma": 2.5,
            "class_weights": {"O": 0.25, "Premise": 1.0, "Claim": 2.0, "MajorClaim": 8.0},
        },
        "notes_suffix": "+ focal loss (gamma=2.5, O=0.25)",
    },
    "crf": {
        "subtask1_overrides": {
            "use_crf": True,
        },
        "notes_suffix": "+ CRF",
    },
    "focal_crf": {
        "subtask1_overrides": {
            "loss": "focal",
            "focal_gamma": 2.5,
            "use_crf": True,
            "class_weights": {"O": 0.25, "Premise": 1.0, "Claim": 2.0, "MajorClaim": 8.0},
        },
        "notes_suffix": "+ focal + CRF",
    },
    "focal_crf_bilstm": {
        "subtask1_overrides": {
            "loss": "focal",
            "focal_gamma": 2.5,
            "use_crf": True,
            "use_bilstm": True,
            "bilstm_hidden": 256,
            "class_weights": {"O": 0.25, "Premise": 1.0, "Claim": 2.0, "MajorClaim": 8.0},
        },
        "notes_suffix": "+ focal + CRF + BiLSTM",
    },
}

NLI_VARIANTS = {
    "nli_small": {
        "subtask2_overrides": {
            "method": "nli",
            "nli_backbone": "cross-encoder/nli-deberta-v3-small",
            "rel_epochs": 5,
            "partial_attack_threshold": 0.3,
        },
        "notes_suffix": "+ NLI-DeBERTa-v3-small",
    },
    "nli_base": {
        "subtask2_overrides": {
            "method": "nli",
            "nli_backbone": "cross-encoder/nli-deberta-v3-base",
            "rel_epochs": 5,
            "partial_attack_threshold": 0.3,
        },
        "notes_suffix": "+ NLI-DeBERTa-v3-base",
    },
    "nli_xlmr": {
        "subtask2_overrides": {
            "method": "nli",
            "nli_backbone": "symanto/xlm-roberta-base-snli-mnli-anli-xnli",
            "rel_epochs": 5,
            "partial_attack_threshold": 0.3,
        },
        "notes_suffix": "+ NLI-XLM-R multilingual",
    },
}


def main() -> None:
    base_path = Path(sys.argv[1])
    base = yaml.safe_load(base_path.read_text())
    base_stem = base_path.stem  # e.g., "rigoberta2"

    all_variants = {**VARIANTS, **NLI_VARIANTS}

    for variant_name, variant_cfg in all_variants.items():
        out = yaml.safe_load(yaml.dump(base))  # deep copy
        out["tag"] = f"{base_stem}-{variant_name}"

        if "subtask1_overrides" in variant_cfg:
            out["task"]["subtask1"].update(variant_cfg["subtask1_overrides"])

        if "subtask2_overrides" in variant_cfg:
            if "subtask2" not in out["task"]:
                out["task"]["subtask2"] = {}
            out["task"]["subtask2"].update(variant_cfg["subtask2_overrides"])

        out["notes"] = f"{base.get('notes', '')} {variant_cfg['notes_suffix']}"

        out_path = base_path.parent / f"{base_stem}_{variant_name}.yaml"
        out_path.write_text(yaml.dump(out, default_flow_style=False, allow_unicode=True))
        print(f"  Created: {out_path}")


if __name__ == "__main__":
    main()
