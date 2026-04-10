#!/usr/bin/env python3
"""Generate synthetic training data using Claude API.

Targets the two bottleneck classes:
- MajorClaim entities (64 examples, 2.8% of Track 1 entities)
- Attack relations (36 examples, 2.5% of Track 1 relations)

Uses seed examples from the training data to prompt Claude for diverse
new Spanish RCT abstracts with correct char-offset annotations.

Usage:
    python scripts/generate_synthetic.py --target majorclaim --count 100
    python scripts/generate_synthetic.py --target attack --count 50
    python scripts/generate_synthetic.py --target both --count 100
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import anthropic

from grace.io.loaders import load_track1

_TOPICS = [
    "diabetes mellitus tipo 2",
    "hipertensión arterial",
    "insuficiencia cardíaca",
    "cáncer de mama",
    "enfermedad pulmonar obstructiva crónica",
    "artritis reumatoide",
    "esclerosis múltiple",
    "enfermedad de Alzheimer",
    "leucemia linfoblástica aguda",
    "infarto agudo de miocardio",
    "fibrilación auricular",
    "asma bronquial",
    "enfermedad renal crónica",
    "hepatitis C",
    "VIH/SIDA",
    "depresión mayor",
    "esquizofrenia",
    "melanoma cutáneo",
    "cáncer colorrectal",
    "lupus eritematoso sistémico",
    "epilepsia",
    "trastorno bipolar",
    "osteoporosis",
    "enfermedad de Parkinson",
    "sepsis",
    "neumonía adquirida en la comunidad",
    "tromboembolismo pulmonar",
    "pancreatitis aguda",
    "cirrosis hepática",
    "enfermedad inflamatoria intestinal",
]

_PROMPT_TEMPLATE = (
    """Eres un experto en ensayos clínicos randomizados (RCTs). """
    """Genera un NUEVO resumen de RCT en español sobre {topic}.

El resumen debe contener componentes argumentativos anotados con offsets de caracteres EXACTOS.

{requirements}

Formato de salida (JSON estricto):
{{
  "raw_text": "texto completo del resumen...",
  "entities": [
    {{"id": "T1", "text": "texto exacto", "start": N, "end": N, "type": "MajorClaim"}},
    {{"id": "T2", "text": "otro texto", "start": N, "end": N, "type": "Premise"}},
    ...
  ],
  "relations": [
    {{"id": "R1", "arg1_id": "T2", "arg2_id": "T1", "relation_type": "Support"}},
    ...
  ]
}}

REGLAS CRÍTICAS:
1. raw_text[start:end] DEBE ser EXACTAMENTE igual a "text" para cada entidad
2. Los offsets son en caracteres (no bytes ni tokens)
3. start es inclusivo, end es exclusivo
4. El resumen debe tener 150-300 palabras
5. Usa terminología médica real en español
6. Las relaciones deben ser entre entidades que existen en la lista

Ejemplo de referencia (NO copiar, solo para formato):
{seed_example}"""
)

_MAJORCLAIM_REQUIREMENTS = """Requisitos obligatorios:
- Al menos 1 MajorClaim (conclusión principal del estudio)
- 3-6 Premises (evidencia: datos numéricos, p-values, intervalos de confianza)
- 1-3 Claims (afirmaciones intermedias)
- Relaciones Support y Attack entre componentes
- La MajorClaim debe estar en la última oración del resumen"""

_ATTACK_REQUIREMENTS = """Requisitos obligatorios:
- Al menos 1 relación Attack (una premisa contradice una afirmación)
- 3-6 Premises con datos numéricos
- 2-3 Claims
- Incluir resultados contradictorios (ej: un outcome positivo pero otro negativo)
- La relación Attack debe reflejar una contradicción real en los datos"""


def _extract_seed_examples(train_path: Path, target: str, count: int = 3) -> list[dict]:
    """Extract seed examples containing the target class."""
    cases = list(load_track1(train_path))
    seeds = []

    for case in cases:
        if target == "majorclaim":
            has_target = any(e.type == "MajorClaim" for e in case.entities)
        elif target == "attack":
            has_target = any(r.relation_type == "Attack" for r in case.relations)
        else:
            has_target = True

        if has_target:
            seeds.append(
                {
                    "raw_text": case.raw_text[:500],  # truncate for prompt
                    "entities": [
                        {
                            "id": e.id,
                            "text": e.text[:80],
                            "start": e.start,
                            "end": e.end,
                            "type": e.type,
                        }
                        for e in case.entities[:5]
                    ],
                    "relations": [
                        {
                            "id": r.id,
                            "arg1_id": r.arg1_id,
                            "arg2_id": r.arg2_id,
                            "relation_type": r.relation_type,
                        }
                        for r in case.relations[:5]
                    ],
                }
            )

    random.shuffle(seeds)
    return seeds[:count]


def _validate_offsets(case: dict) -> tuple[bool, list[str]]:
    """Validate that all entity offsets are correct."""
    errors = []
    raw = case.get("raw_text", "")

    for ent in case.get("entities", []):
        start = ent.get("start", 0)
        end = ent.get("end", 0)
        text = ent.get("text", "")

        if start < 0 or end > len(raw) or start >= end:
            errors.append(f"{ent['id']}: invalid offsets [{start}:{end}] (text len={len(raw)})")
            continue

        actual = raw[start:end]
        if actual != text:
            errors.append(f"{ent['id']}: offset mismatch: '{actual}' != '{text}'")

    # Validate relation references
    ent_ids = {e["id"] for e in case.get("entities", [])}
    for rel in case.get("relations", []):
        if rel["arg1_id"] not in ent_ids:
            errors.append(f"{rel['id']}: arg1_id '{rel['arg1_id']}' not in entities")
        if rel["arg2_id"] not in ent_ids:
            errors.append(f"{rel['id']}: arg2_id '{rel['arg2_id']}' not in entities")

    return len(errors) == 0, errors


def generate_batch(
    client: anthropic.Anthropic,
    target: str,
    count: int,
    train_path: Path,
    model: str = "claude-sonnet-4-20250514",
) -> list[dict]:
    """Generate a batch of synthetic cases."""
    seeds = _extract_seed_examples(train_path, target)
    if not seeds:
        print(f"WARNING: No seed examples found for target={target}")
        return []

    requirements = _MAJORCLAIM_REQUIREMENTS if target == "majorclaim" else _ATTACK_REQUIREMENTS
    results = []
    failures = 0

    for i in range(count):
        topic = random.choice(_TOPICS)  # noqa: S311
        seed = random.choice(seeds)  # noqa: S311

        prompt = _PROMPT_TEMPLATE.format(
            topic=topic,
            requirements=requirements,
            seed_example=json.dumps(seed, ensure_ascii=False, indent=2)[:1000],
        )

        try:
            response = client.messages.create(
                model=model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text

            # Extract JSON from response
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                case = json.loads(content[json_start:json_end])
            else:
                failures += 1
                continue

            # Validate
            valid, errors = _validate_offsets(case)
            if valid:
                case["id"] = f"synthetic_{target}_{i}"
                results.append(case)
                print(
                    f"  [{i+1}/{count}] OK: {len(case.get('entities', []))} entities, "
                    f"{len(case.get('relations', []))} relations"
                )
            else:
                failures += 1
                print(f"  [{i+1}/{count}] INVALID: {'; '.join(errors[:3])}")

        except Exception as e:
            failures += 1
            print(f"  [{i+1}/{count}] ERROR: {e}")

        # Rate limiting
        if i < count - 1:
            time.sleep(0.5)

    print(f"\nGenerated {len(results)}/{count} valid cases ({failures} failures)")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["majorclaim", "attack", "both"], default="both")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("downloaded_data/synthetic"))
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("downloaded_data/public_data/public_data/track_1_train.json"),
    )
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    args = parser.parse_args()

    client = anthropic.Anthropic()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.target in ("majorclaim", "both"):
        print(f"\n=== Generating {args.count} MajorClaim abstracts ===")
        mc_cases = generate_batch(client, "majorclaim", args.count, args.train, args.model)
        out_path = args.output / "synthetic_majorclaim.json"
        out_path.write_text(json.dumps(mc_cases, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved to {out_path}")

    if args.target in ("attack", "both"):
        print(f"\n=== Generating {args.count} Attack-rich abstracts ===")
        atk_cases = generate_batch(client, "attack", args.count, args.train, args.model)
        out_path = args.output / "synthetic_attack.json"
        out_path.write_text(json.dumps(atk_cases, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
