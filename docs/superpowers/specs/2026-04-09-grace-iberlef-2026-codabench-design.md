# GRACE @ IberLEF 2026 — System Design

**Date:** 2026-04-09
**Task:** GRACE — Granular Recognition of Argumentative Clinical Evidence
**Codabench:** https://www.codabench.org/competitions/13280/
**Official site:** https://grace2026.hitz.eus (redirects to Codabench)
**Host:** HiTZ (Basque Center for Language Technology, UPV/EHU)
**Organizers:** Iker de la Iglesia, Aitziber Atutxa, Ander Barrena, Koldo Gojenola, Raquel Martínez, Soto Montalvo, Miguel Ángel Rodríguez, Sofía Zakhir Puig, Vanesa Gómez Martínez
**Workspace:** `C:\Users\Prekzursil\Downloads\bionlp\`
**Status:** Draft v1 — awaiting user review gate

---

## 0. Executive summary

We will build a single repository-level system under `grace/` that competes in **both tracks** of GRACE @ IberLEF 2026, targets a full research contribution (leaderboard + IberLEF system paper + public code release under Apache-2.0), and fits the user's compute:

- **Local:** RTX 4050 Laptop 6 GB VRAM + 16 GB shared memory. **Only suitable for BETO, XLM-R-base, and smoke tests.**
- **Remote:** Kaggle Notebooks P100 16 GB. **XLM-R-large training is constrained to here.** Budget against Kaggle's 30 GPU-hour/week cap.
- **APIs:** Claude + GPT-4 via SDK for Track 2 LLM reasoning. Hard budget cap of **$250 total** enforced via `experiments/budget.yaml`.
- **Windows caveat:** `bitsandbytes` on native Windows 11 is fragile; if QLoRA becomes necessary the user installs WSL2 on Day 1. Otherwise stick to native torch/transformers.
- **Codabench quota:** 3-5 submissions/day typical — budget at most 2 dev-phase and 3 test-phase uploads.

The system is hybrid by design:

- **Track 1 (Clinical Trial Evidence & Argumentation)** — multilingual encoder fine-tuning (XLM-R-large / mDeBERTa-v3 / BETO) with class-weighted loss and cross-lingual rare-class augmentation.
- **Track 2 (Clinical Case Reasoning on MIR)** — LLM chain-of-thought reasoning conditioned on the provided `correct_choice_id`, distilled into a compact open-weights NLI student for reproducibility and rules-compliance.

The novelty for the system paper is in (1) filtered cross-lingual projection of rare classes in Track 1, and (2) LLM-reasoning distillation conditioned on the correct answer in Track 2. An offset-alignment library is reusable tooling contributed alongside the paper.

**Timeline:** 24 days from 2026-04-09 to the 2026-05-03 submission deadline, with the test set released on 2026-04-22. Paper deadline 2026-05-24. Workshop 2026-09-22.

---

## 1. Architecture and directory layout

### 1.1 System architecture

```
                ┌─────────────────────────────────────────┐
                │    downloaded_data/  (organizer files)  │
                │    public_data/track_{1,2}_{train,dev}  │
                │    track{1,2}_scoring_program/          │
                └────────────────┬────────────────────────┘
                                 │
                                 ▼
                ┌─────────────────────────────────────────┐
                │   grace/io/  (loader + offset library)  │
                │   • GraceCase / GraceEntity / GraceRel  │
                │       frozen dataclasses                │
                │   • load_track1 / load_track2           │
                │   • OfficialTokenizer (\w+|[^\w\s])     │
                │   • CharSpan ↔ TokenSpan round-trip     │
                │   • fingerprint hashing for dedup       │
                └────────────────┬────────────────────────┘
                                 │
             ┌───────────────────┴────────────────────┐
             ▼                                        ▼
   ┌─────────────────────┐              ┌──────────────────────────────┐
   │  grace/track1/      │              │  grace/track2/               │
   │  (encoder pipeline) │              │  (LLM-distilled pipeline)    │
   │                     │              │                              │
   │  component_tagger   │              │  sentence_classifier         │
   │    XLM-R / mDeBERTa │              │    BETO + LLM few-shot head  │
   │    BIO × 3 types    │              │                              │
   │                     │              │  premise_extractor           │
   │  relation_classifier│              │    LLM-then-align OR BIO head│
   │    pairwise head    │              │    on relevant sentences     │
   │    4-class softmax  │              │                              │
   │                     │              │  relation_reasoner           │
   │  data_augmenter     │              │    LLM CoT over (premise ×   │
   │    cross-lingual    │              │    option) conditioned on    │
   │    minority-class   │              │    correct_choice_id         │
   │    backfill         │              │                              │
   │                     │              │  distilled_student           │
   │                     │              │    Spanish NLI head on       │
   │                     │              │    LLM-generated silver data │
   └──────────┬──────────┘              └──────────────┬───────────────┘
              │                                        │
              └───────────────────┬────────────────────┘
                                  ▼
                ┌─────────────────────────────────────────┐
                │   grace/eval/                           │
                │   • official scorer wrapper (import     │
                │     track{1,2}_scoring_program.py)      │
                │   • dev-set leaderboard tracker         │
                │   • per-class error analysis            │
                │   • offset-mismatch diagnostics         │
                └────────────────┬────────────────────────┘
                                 ▼
                ┌─────────────────────────────────────────┐
                │   grace/submit/                         │
                │   • build JSON in exact submission      │
                │     format (dict annotations)           │
                │   • round-trip validate vs official     │
                │     scorer before packaging             │
                │   • package zip for Codabench upload    │
                └─────────────────────────────────────────┘
```

### 1.2 Repository layout

```
bionlp/
├── .venv/                         (existing — pinned Python 3.12 + tooluniverse)
├── .env                           (existing — add OPENAI_API_KEY, ANTHROPIC_API_KEY)
├── downloaded_data/               (existing — organizer data, READ-ONLY)
│
├── grace/                         ← source package
│   ├── __init__.py
│   ├── io/
│   │   ├── schema.py              frozen dataclasses
│   │   ├── loaders.py             load_track1 / load_track2
│   │   ├── tokenizer.py           OfficialTokenizer (scorer-parity)
│   │   └── offsets.py             SpanAligner
│   ├── track1/
│   │   ├── augment.py             cross-lingual rare-class backfill
│   │   ├── component_tagger.py    XLM-R BIO model
│   │   ├── relation_classifier.py pairwise head
│   │   └── predict.py             end-to-end inference
│   ├── track2/
│   │   ├── sentence_clf.py
│   │   ├── premise_extractor.py
│   │   ├── llm_reasoner.py        Claude/GPT wrapper w/ structured output
│   │   ├── distilled_student.py
│   │   └── predict.py
│   ├── eval/
│   │   ├── scorer.py              imports official track{1,2}_scoring_program
│   │   ├── tracker.py             ledger.jsonl writer
│   │   └── diagnose.py            offset mismatch, class confusion
│   └── submit/
│       ├── formatter.py
│       ├── validator.py
│       └── package.py
│
├── scripts/
│   ├── audit_data.py
│   ├── train_track1.py
│   ├── train_track2.py
│   ├── run_llm_baseline.py
│   ├── verify_rules.py
│   └── submit.py
│
├── configs/
│   ├── track1/
│   │   ├── xlmr_large_base.yaml
│   │   ├── mdeberta_v3_base.yaml
│   │   └── xlmr_large_augmented.yaml
│   └── track2/
│       ├── llm_claude_cot.yaml
│       ├── llm_gpt4_cot.yaml
│       └── distilled_student.yaml
│
├── experiments/                   (gitignored outputs)
│   ├── runs/<timestamp>-<tag>/
│   │   ├── config.yaml
│   │   ├── metrics.json
│   │   ├── predictions_dev.json
│   │   └── diagnostics.json
│   ├── ledger.jsonl
│   ├── llm_cache/<hash>.json
│   └── audits/<timestamp>.md
│
├── notebooks/                     (optional, post-deadline only)
│   └── (reserved for post-competition exploration; cut from the 24-day
│        sprint scope after the plan-review-gate — audits happen via
│        scripts/audit_data.py and grace/eval/diagnose.py instead)
│
├── docs/
│   ├── plans/                     (existing)
│   ├── superpowers/specs/
│   │   ├── 2026-04-09-grace-iberlef-2026-codabench-design.md
│   │   └── 2026-04-09-grace-rules-snapshot.md
│   └── paper/
│       ├── draft.md
│       └── references.bib
│
├── tests/
│   ├── test_offsets.py
│   ├── test_loaders.py
│   ├── test_submit_format.py
│   └── test_track1_augment.py
│
├── README.md                      (existing — add GRACE section)
└── pyproject.toml                 (new)
```

### 1.3 Dependencies

```toml
[project]
name = "grace-iberlef-2026"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "transformers>=4.45",
  "torch>=2.3",
  "datasets>=2.20",
  "accelerate>=0.33",
  "peft>=0.12",
  "bitsandbytes>=0.43",
  "scikit-learn>=1.5",
  "pandas>=2.2",
  "pyyaml>=6.0",
  "python-dotenv>=1.0",
  "anthropic>=0.35",
  "openai>=1.40",
  "tenacity>=8.5",
  "rich>=13.7",
  "typer>=0.12",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.3",
  "pytest-cov>=5.0",
  "black>=24.8",
  "ruff>=0.6",
  "mypy>=1.11",
  "jupyter>=1.0",
]
```

### 1.4 Rationale for structural decisions

1. **Package named `grace`.** Short, matches the task name, single importable namespace for notebooks and scripts.
2. **`grace/io/` shared across tracks.** Both tracks reuse the same dataclasses, tokenizer, and offset library; there is exactly one place where offset alignment is defined, eliminating duplication.
3. **`grace/eval/scorer.py` imports the organizer's scoring programs verbatim** rather than reimplementing them. Guarantees that our dev-set numbers match Codabench exactly. Hash-fingerprinting of the scorer files detects any organizer updates.
4. **`experiments/runs/` + `experiments/ledger.jsonl`.** Every training run is exactly one directory plus one JSONL line. The paper's results table is a `jq` query against `ledger.jsonl`.
5. **`configs/` as YAML.** No hardcoded hyperparameters in Python; experiments swap by config file, not code.
6. **`tests/` directory is non-negotiable.** Per project rules (CLAUDE.md) TDD is mandatory; offset round-trip tests are the highest-leverage tests in the project.
7. **`.env` gets API keys appended** alongside existing NCBI/SS/OpenFDA keys. Dotenv infra already exists.

---

## 2. Data ingestion and offset alignment

Every offset bug in this layer becomes a silent submission failure — a model with 95% correct reasoning that is off by one character scores 0% strict F1. This section defines the contract.

### 2.1 Dataclasses (`grace/io/schema.py`)

Frozen and immutable. Tuples instead of lists; `slots=True` for memory and correctness.

```python
from dataclasses import dataclass
from typing import Literal

EntityType = Literal["Premise", "Claim", "MajorClaim"]
RelationType = Literal["Support", "Attack", "Partial-Attack"]
SentRelevancy = Literal["relevant", "not-relevant"]

@dataclass(frozen=True, slots=True)
class GraceEntity:
    id: str
    text: str
    start: int                  # char offset, INCLUSIVE
    end: int                    # char offset, EXCLUSIVE
    type: EntityType

@dataclass(frozen=True, slots=True)
class GraceRelation:
    id: str
    arg1_id: str
    arg2_id: str
    relation_type: RelationType

@dataclass(frozen=True, slots=True)
class GraceSentence:
    sentence: str
    start: int
    end: int

@dataclass(frozen=True, slots=True)
class GraceChoice:
    id: str
    text: str
    start: int
    end: int

@dataclass(frozen=True, slots=True)
class GraceCase:
    id: str
    raw_text: str
    track: Literal[1, 2]
    metadata_context: str | None = None
    context_sentences: tuple[GraceSentence, ...] = ()
    choices: tuple[GraceChoice, ...] = ()
    correct_choice_id: str | None = None
    sentence_relevancy: tuple[SentRelevancy, ...] = ()
    entities: tuple[GraceEntity, ...] = ()
    relations: tuple[GraceRelation, ...] = ()
```

### 2.2 Loaders (`grace/io/loaders.py`)

Signatures:

```python
def load_track1(path: Path) -> tuple[GraceCase, ...]
def load_track2(path: Path) -> tuple[GraceCase, ...]
def save_predictions(cases: Sequence[GraceCase], path: Path, track: int) -> None
```

**Format fact established by data audit (2026-04-09):**
The actual train/dev JSON files use the scoring-program layout — `annotations` is a DICT with keys `{entities, relations}` (Track 1) and `{sentence_relevancy, entities, relations}` (Track 2). The `instance_examples/` sample files use an out-of-date flat-list layout; loaders MUST target the scoring-program layout, not the example files.

**Contract enforced on load:**
- Every entity: `0 <= start < end <= len(raw_text)`
- Every entity: `raw_text[start:end] == entity.text` (substring exact match)
- Every relation: `arg1_id` and `arg2_id` exist in `entities`
- Track 2: `len(sentence_relevancy) == len(context_sentences)`
- Track 2: `correct_choice_id in {c.id for c in choices}`
- Any violation raises a typed exception with the case ID and the exact field.

### 2.3 Official tokenizer (`grace/io/tokenizer.py`)

Verbatim port of the scorer's regex: `r"\w+|[^\w\s]"` with `re.UNICODE`. Our tokenizer must produce identical outputs to ensure relaxed-match IoU calculations match the scorer.

```python
import re
from typing import NamedTuple

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

class Token(NamedTuple):
    text: str
    start: int
    end: int

class OfficialTokenizer:
    def tokenize(self, text: str) -> tuple[Token, ...]:
        return tuple(Token(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(text))

    def token_indices_in_span(self, tokens, start, end) -> frozenset[int]:
        return frozenset(i for i, t in enumerate(tokens) if t.start >= start and t.end <= end)

    def token_iou(self, a: frozenset[int], b: frozenset[int]) -> float:
        if not a and not b:
            return 1.0
        union = len(a | b)
        return len(a & b) / union if union else 0.0
```

### 2.4 SpanAligner (`grace/io/offsets.py`) — the key utility

Bridges HuggingFace tokenizer offsets with the official char-offset contract. Operations:

- `encode_with_labels(text, gold_entities, max_length)` — for training: convert gold char spans into BIO labels on subwords. Handles overlap, truncation, and entity splits across chunks for long abstracts.
- `decode_bio_to_entities(text, token_labels, offset_mapping)` — for inference: convert predicted BIO tags back into char-span entities. Predicted end offsets snap to whitespace boundaries; if a raw subword span ends mid-word, extend to end-of-word.
- `snap_to_token_boundary(text, char_start, char_end)` — given arbitrary offsets, snap to nearest valid official-tokenizer boundaries. Used when LLM returns slightly-off offsets.
- `validate_round_trip(text, entities)` — invariant: for every entity, `text[e.start:e.end] == e.text`. Raises AlignmentError on first violation.

**Sliding-window strategy for long Track 1 abstracts** (max 4.4 KB ≈ 1100-1400 tokens):
- `stride=128` overlap
- per-window BIO predictions
- merge via majority vote on overlap regions
- re-check offset consistency per merged entity

### 2.5 Test invariants (`tests/test_offsets.py`)

```python
class TestOffsetInvariants:
    def test_every_gold_entity_substring_matches(self): ...
    def test_load_save_roundtrip_is_identity(self): ...
    def test_official_tokenizer_matches_scoring_program(self): ...
    def test_huggingface_tokenizer_roundtrip(self): ...
    def test_snap_to_boundary_is_idempotent(self): ...
    def test_predictions_pass_official_scorer(self): ...  # MOST VALUABLE
    def test_variable_choice_count_track2(self): ...
```

`test_predictions_pass_official_scorer` is the single highest-value test: it loads train.json, copies `annotations → predictions`, runs the official scorer, and verifies every metric is 1.0. Catches format regressions in under 10 seconds.

### 2.6 Data audit script (`scripts/audit_data.py`)

Reproducible CLI producing Markdown reports in `experiments/audits/<timestamp>.md`. Baseline numbers established by initial audit (2026-04-09):

| File | Cases | Format validation |
|---|---|---|
| `track_1_train.json` | **350** | `annotations` as DICT with `{entities, relations}` |
| `track_1_dev.json` | **50** | same |
| `track_2_train.json` | **128** | `annotations` as DICT with `{sentence_relevancy, entities, relations}` |
| `track_2_dev.json` | **24** | same |

**Track 1 train entity distribution:** Premise 1536 (68%), Claim 666 (29%), MajorClaim 64 (3%).
**Track 1 train relation distribution:** Support 1213 (86%), Partial-Attack 169 (12%), Attack 36 (2.5%).
**Track 2 train:** 128 cases; ~9 entities/case, ~10 relations/case; balanced Premise/Claim (49/51%), balanced Support/Attack (49/51%), sentence relevance 53/47% (relevant/not-relevant). Variable choice count: 86 cases with 5 options, 42 with 4.

---

## 3. Track 1 encoder pipeline

### 3.1 Backbone selection

| Model | Spanish coverage | Max seq | Size | Laptop 4050 fit | Rationale |
|---|---|---|---|---|---|
| **XLM-RoBERTa-large** | ✅ multilingual | 512 | 560 M | ⚠️ Kaggle P100 only | **Primary — best expected headline number** |
| **BETO (bert-base-spanish-wwm-cased)** | ✅ Spanish | 512 | 110 M | ✅ | **Secondary — local debug + fast iteration + paper's second system** |
| XLM-RoBERTa-base | ✅ | 512 | 278 M | ✅ | Smoke-test baseline only |

**Training plan (revised after plan-review-gate):** **Two backbones only** — XLM-R-large as the primary submission system, BETO as the local-debug rig and the paper's second-best-system comparison point. **mDeBERTa-v3 cut** — 3-backbone × 5-ablation × 3-seed sweep does not fit in the Kaggle 30-GPU-hour/week cap. Ensemble = XLM-R-large (3 seeds) → logit averaging.

**Not considered:** SciBERT/BioBERT/ClinicalBERT (English-only), LlamaCare/BioMistral (overparameterized for token classification). mDeBERTa-v3 is a **post-deadline stretch goal** if the workshop paper needs a third system for reviewer questions.

### 3.2 Subtask 1 — Component Detection

**Architecture:** Backbone + linear head → 7 BIO labels: `O, B-Premise, I-Premise, B-Claim, I-Claim, B-MajorClaim, I-MajorClaim`.

**Training:**
- Sliding window encoding, `max_length=512, stride=128`
- Class-weighted cross-entropy, weights inversely proportional to label frequency, clamped `[1.0, 20.0]`; MajorClaim ~10×, Premise 1×
- Optional focal loss ablation, `γ=2.0`
- AdamW, `lr=2e-5`, `weight_decay=0.01`, 10% warmup, linear decay
- 10 epochs, early stopping on dev Macro F1 (official metric)
- Batch 16 on Kaggle P100, gradient accumulation to effective 32 if needed
- fp16/bf16 mixed precision
- 3 seeds (41, 42, 43), report mean±std

**Inference:**
- Sliding-window encoding per abstract
- Merge overlapping predictions by max-probability voting
- Greedy BIO decoding (no CRF — data is too small for meaningful CRF gain; ablation only)
- `SpanAligner.decode_bio_to_entities` snaps to whitespace boundaries

**Expected dev Macro F1:**
- XLM-R-large baseline: 0.60-0.68
- + cross-lingual augmentation: 0.63-0.72

### 3.3 Subtask 2 — Relation Classification

Pairwise classifier over gold entities during training, predicted entities during inference.

**Architecture:** Backbone + 4-class softmax head (`Support, Attack, Partial-Attack, no-relation`). Separate encoder from Subtask 1 to avoid interference.

**Input encoding — compare in ablation:**

- **Scheme A — marker tokens**: `[CLS] <context> [SEP] <e1> span1 </e1> ... <e2> span2 </e2> [SEP]`. Pool `[CLS]` + positions of `<e1>`/`<e2>`, concat, linear head.
- **Scheme B — typed markers with pooling**: `[CLS] <premise> span1 </premise> [SEP] <claim> span2 </claim> [SEP]`. Mean-pool inside each marked region, concat, linear head.

**Pair generation:**
- Training: all gold `(entity_i, entity_j)` pairs in the same abstract; label from `gold_relations` or `no-relation` if absent
- Inference: all predicted pairs in the same abstract, skipping identical pairs
- Negative sampling: subsample `no-relation` to 3× the rate of true relations

**Loss:** class-weighted cross-entropy with `Attack ~20×` and `Partial-Attack ~7×` (ratios from audit).

**Expected Macro F1:** 0.35-0.50 strict — Attack has 36 training examples, making this a research-hard problem.

### 3.4 Cross-lingual data augmentation for rare classes (novel contribution)

**Gap:** MajorClaim 64 examples, Attack 36 examples in GRACE train. Too few for robust learning.

**Pipeline:**

1. **Source corpora (public, permissive licenses):**
   - English AbstRCT (original)
   - HiTZ AbstRCT-French and AbstRCT-Italian (projected)
   - Public HiTZ AbstRCT-ES is NOT usable directly — it only has 2 entity types, while GRACE has 3.

2. **Translate rare-class instances → Spanish** via DeepL (paper's preferred MT) or Claude. Project annotation offsets using `awesome-align` / `SimAlign`.

3. **Post-process projections** with `SpanAligner.snap_to_token_boundary` to match GRACE tokenizer.

4. **Filter** projected spans by:
   - Spanish XNLI entailment: projected premise should entail/contradict projected claim consistent with gold Support/Attack
   - Length ratio sanity: `|es_span| / |en_span| ∈ [0.6, 2.5]`
   - Alignment confidence: IoU ≥ 0.7

5. **Silver training data** added to GRACE train. Expected yield: +100-200 MajorClaim, +50-120 Attack, +150-250 Partial-Attack.

6. **Ablation set (for paper):**
   - **A0** — GRACE only (baseline)
   - **A1** — GRACE + silver unfiltered
   - **A2** — GRACE + silver entailment-filtered (primary contribution)
   - **A3** — A2 + class-weighted loss
   - **A4** — A3 + 3-seed averaging
   - Isolating ablations: "augment only MajorClaim" / "augment only Attack"

**Claim:** Entailment-filtered cross-lingual projection improves rare-class recall on Spanish clinical argument mining.

### 3.5 Training CLIs

```bash
python scripts/train_track1.py --config configs/track1/xlmr_large_augmented.yaml
python scripts/train_track1.py --config ... --seeds 41,42,43 --out experiments/runs/
python scripts/train_track1.py --eval-only --checkpoint <path>
```

Each run produces: `config.yaml`, `metrics.json`, `predictions_dev.json`, `diagnostics.json`, plus a one-line `ledger.jsonl` append.

---

## 4. Track 2 LLM-distilled pipeline

### 4.1 Task framing

Three facts from the data audit reshape the problem:

1. **`metadata.choices` gives 4-5 pre-segmented option spans with exact char offsets.** Claim output spans are copy-from-metadata — Subtask 2's Claim F1 is ~1.0 for free if we are disciplined. Subtask 2's official score is driven almost entirely by Premise extraction quality.
2. **`metadata.correct_choice_id` is provided at input time** (confirmed in both example files and the README). Subtask 3 is not "which option is correct" — it is "given the correct answer, how does each premise relate to each option?" That is a grounded NLI-style task suited to LLM chain-of-thought.
3. **128 training cases** is too small to fine-tune a 7B LLM from scratch but more than enough for few-shot prompting (6-8 exemplars) and for distilling a small student NLI model (~5,700 premise×option pairs).

### 4.2 Subtask 1 — Evidence Sentence Detection

Three-way ensemble:

**Model A — BETO sentence classifier** (cheap, reproducible, primary)
- `dccuchile/bert-base-spanish-wwm-cased`
- Input: `[CLS] <context> [SEP] <sentence> [SEP] <correct_option_text> [SEP]`
- Binary head, class-weighted BCE
- 10 epochs, early stop on dev F1(relevant)

**Model C — Few-shot LLM prompt** (quality ceiling)
- 6-8 exemplars stratified by label and medical specialty
- Structured JSON output
- Claude 4.6 Sonnet primary, GPT-4o secondary
- Cost budget ~$20-30 for dev+test

**Cut after plan-review-gate:** Model B (XLM-R + XNLI transfer). A 2-way A+C ensemble is simpler, more defensible in the paper, and saves 1-2 days of Phase 4. The Model B variant becomes a post-deadline stretch.

**Ensemble:** 2-way majority vote (A, C). Ties → `relevant` (false negatives on Subtask 1 propagate to Subtask 2).

**Expected F1(relevant):** 0.72-0.85 (solo BETO ~0.72, with LLM ~0.80, ensemble ~0.82-0.85).

### 4.3 Subtask 2 — Premise Span Extraction

Two-stage: LLM proposes, extractive head aligns.

**Stage 1 — LLM proposes:**
For each sentence classified as `relevant`, prompt the LLM to return JSON of **verbatim substrings** from the sentence that serve as clinical evidence, conditioned on the case, correct option, and sentence.

**Stage 2 — Snap to offsets:**
For each proposed phrase:
1. Exact substring match in `raw_text`
2. Fuzzy match with edit distance ≤ 2 on failure
3. Drop-unmatched on failure (log for error analysis)
4. `SpanAligner.snap_to_token_boundary` locks final `(start, end)`

**Stage 3 — Scope A eval:** predictions outside gold-relevant sentences count as FPs, so Subtask 1 quality caps Subtask 2 score.

**Fallback:** Fine-tune BETO-base span extractor on GRACE train + LLM silver labels. This is the distillation half.

**Expected Scope A Strict Micro F1:** 0.40-0.55 (relaxed IoU ≥ 0.5 should be 0.55-0.70).

### 4.4 Subtask 3 — Relation Classification (research heart)

**Framing:** For each extracted premise × each provided option, predict `{Support, Attack, none}`. Scorer treats "none" as "not emitted".

**Prompt structure:** Spanish medical reasoning prompt including case, premise (with char span), all N options, `correct_choice_id`, and asking for step-by-step reasoning per option followed by structured JSON labels.

**Key design:** `correct_choice_id` is passed as an input signal, not a prediction target. Training distribution assumes the system knows the correct answer; the task is to explain it.

**Few-shot exemplars (4):**
1. Clean Support (direct biomarker → correct diagnosis)
2. Clean Attack (finding contradicting a plausible wrong option)
3. Partial/ambiguous (multi-relation premise)
4. Negation (absence of symptom as evidence)

**Scale (revised after plan-review-gate, with honest token accounting):**
- Train: 128 cases × ~9 premises × ~5 options ≈ 5,760 pair calls
- Dev: 24 cases × ~45 ≈ 1,080 pair calls
- Per-call input tokens: ~3,500-5,000 (exemplars + case + premise + all options)
- Per-call output tokens: ~400-800 (CoT reasoning)
- At Claude Sonnet pricing ($3/MTok input, $15/MTok output): ~$0.015-0.027 per pair × 6,840 pairs = **$100-180 for Subtask 3 alone**
- Plus Subtask 1 (Model C sentence classifier LLM calls): ~$20-30
- Plus Subtask 2 (premise extractor LLM calls): ~$20-30
- Plus prompt-engineering iteration (500-2000 test calls): ~$10-30
- **Total LLM budget: $150-250 hard cap** (enforced via `experiments/budget.yaml` read by `LLMReasoner` in every run)

### 4.5 Distillation to open-weights student

**Why distill:**
- Insurance against IberLEF rules that may forbid closed APIs on the test set (verified in §5)
- 100× cheaper inference, fully reproducible
- Adds a concrete paper contribution

**Student:**
- Backbone: `xlm-roberta-base` or `dccuchile/bert-base-spanish-wwm-cased`
- Input: `[CLS] <case_context> [SEP] <premise_text> [SEP] <option_text> [SEP] <correct_option_text> [SEP]`
- 3-class head: Support / Attack / none

**Training data:**
- Gold: 1,259 relations + sampled no-relation pairs → ~2,500 labeled pairs
- Silver: LLM predictions on training set where LLM ≈ gold (confident distillation)
- Augmentation: option-order swaps, language-variant swaps, back-translation via EN

**Training:**
- AdamW, `lr=3e-5`, batch 32
- Class-weighted cross-entropy (near-balanced Support/Attack)
- Dev Macro F1 Strict (official metric)
- Ablation: gold only / gold+silver / gold+silver+augmentation

**Expected Strict Macro F1 for Subtask 3:**
- BETO gold only: 0.55-0.65
- BETO + silver: 0.60-0.70
- LLM few-shot: 0.65-0.75
- Ensemble (LLM + student): 0.68-0.78

### 4.6 Overall expected Track 2 score

| Subtask | Baseline | Ensemble | Stretch |
|---|---|---|---|
| S1 — F1(relevant) | 0.75 | 0.82 | 0.87 |
| S2 — Scope A Strict Micro F1 | 0.42 | 0.50 | 0.58 |
| S3 — Strict Macro F1 | 0.58 | 0.72 | 0.78 |
| **Overall** | **0.58** | **0.68** | **0.74** |

### 4.7 Infrastructure

`grace/track2/llm_reasoner.py`:
- Prompt templates in `configs/track2/prompts/`
- Response caching to `experiments/llm_cache/<hash>.json` (idempotent by prompt hash)
- Tenacity exponential-backoff retries
- Per-call cost tracking → `experiments/runs/<tag>/llm_cost.json`
- Provider abstraction: Claude / GPT-4 / Qwen2.5-72B behind one interface
- **Hard budget cap** that aborts execution when exceeded

Training/inference compute split:
- LLM calls: API
- Distilled student training: local 4050 or Kaggle P100
- Test-set inference: run both pipelines, submit best or ensemble

### 4.8 Open risks for Track 2

1. **LLM offset hallucination** → snap-to-boundary + fuzzy match + drop-unmatched + BETO fallback head
2. **Distribution shift** at test time → multi-specialty exemplars + ensemble robustness
3. **API cost overrun** → hard budget cap + caching
4. **Rules forbid closed APIs** → distilled student as fallback path (baked in)

---

## 5. Evaluation, rules, and project hygiene

### 5.1 Official scorer wrapper (`grace/eval/scorer.py`)

Import organizer's scoring programs directly:

```python
import importlib.util
from pathlib import Path

SCORER_ROOT = Path("downloaded_data")

def _load_scorer_module(track: int):
    path = SCORER_ROOT / f"track{track}_scoring_program" / f"track{track}_scoring_program" / f"track{track}_scoring_program.py"
    spec = importlib.util.spec_from_file_location(f"grace_scorer_t{track}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def score_track1(predictions_path: Path, gold_path: Path) -> dict:
    return _load_scorer_module(1).evaluate(predictions_path, gold_path)

def score_track2(predictions_path: Path, gold_path: Path) -> dict:
    return _load_scorer_module(2).evaluate(predictions_path, gold_path)
```

**SHA256 fingerprint** of each scorer file logged alongside every run's metrics. Organizer updates detected immediately.

### 5.2 Experiment ledger

Every run appends one line to `experiments/ledger.jsonl`:

```json
{
  "timestamp": "2026-04-12T14:23:01",
  "tag": "xlmr-large-aug-a3-seed42",
  "git_sha": "abc123...",
  "track": 1,
  "subtask": "both",
  "backbone": "xlm-roberta-large",
  "config_path": "configs/track1/xlmr_large_augmented.yaml",
  "scorer_sha256": "fedc...",
  "dev_metrics": {
    "subtask1_official": 0.6712,
    "subtask2_official": 0.4421,
    "overall": 0.5567
  },
  "train_cost_minutes": 38,
  "llm_cost_usd": 0.00,
  "notes": "A3 ablation with augmented rare-class data + class weights"
}
```

Paper results table derivable via `jq` one-liner. Ledger is append-only, always committed.

### 5.3 Reproducibility discipline

- Seeds set in `random`, `numpy`, `torch`, `torch.cuda` (all four)
- `torch.use_deterministic_algorithms(True)` where feasible
- `transformers` version pinned in `pyproject.toml`
- Model revisions pinned via `revision="..."` in `from_pretrained`
- SHA256 of each input JSON logged per run
- `.env` never committed (verified in `.gitignore` before first commit)

### 5.4 Error analysis (`grace/eval/diagnose.py`)

Every run produces `diagnostics.json`:
- Per-class confusion matrix
- Offset error histogram (P50, P90, P99, max)
- Worst-N cases: top 10 dev cases by F1 drop from corpus average
- Length-vs-score correlation
- Per-medical-specialty scores (Track 2)
- Scorer bug workaround checks (Track 1 `relaxed_mixed`)

### 5.5 Submission packaging (`grace/submit/`)

Three-step pipeline before any Codabench upload:

1. **formatter.py** — `GraceCase` tuples → scorer-expected JSON (annotations-as-dict)
2. **validator.py** — run official scorer on submission with `gold_path=dev.json`; abort if score ≠ logged dev score ± 0.001
3. **package.py** — zip with naming convention, print upload path

**User confirmation required before any upload** per `explicit_permission` rules.

### 5.6 Rules verification gate

Before committing to the closed-API pipeline, verify IberLEF 2026 GRACE rules explicitly allow it. Produce `docs/superpowers/specs/2026-04-09-grace-rules-snapshot.md` with verbatim rule text + URL + fetch date.

**Verification order:**
1. Fetch Codabench Terms tab (SPA — may require fallback)
2. Search corpora-list archive and HiTZ announcement for rule snippets
3. Email organizers only as last resort with user approval

**Rules-contingent strategy:**
- **Closed APIs forbidden on test set:** Final submission = open-weights distilled student. Expected Track 2 score drops from 0.68-0.74 → ~0.60-0.68. Paper contribution intact.
- **Closed APIs allowed:** Final = LLM + distilled student ensemble. Paper gains additional ablation.
- **Pipeline design works without modification either way** — the student is baked in from day one.

### 5.7 Python project hygiene (trimmed after plan-review-gate)

Per user's global rules in `~/.claude/rules/python/`:
- PEP 8 + type annotations on all function signatures
- Formatting: `black`, `ruff` (with I rule set for import sorting)
- Testing: `pytest` with `pytest-cov`
- **Coverage thresholds** — per-module to avoid ML-code coverage theater:
  - `grace/io/*`: 95%+ lines (correctness-critical)
  - `grace/eval/*`: 95%+ lines (correctness-critical)
  - `grace/submit/*`: 90%+ lines
  - `grace/track1/*`, `grace/track2/*`: 70%+ lines (training loops are hard to cover without a GPU)
  - Overall: 80%+ lines
  - Enforced via `.coverage-thresholds.json` read by a custom pytest plugin in `tests/conftest.py`
- **Pre-commit hooks** — minimal to reduce sprint friction: `black → ruff` only. No mypy, no pytest in pre-commit, no bandit. `mypy` runs in CI / manually before PRs but not on every commit.
- **Security:** `bandit -r grace/` before every merge to main (not every commit)
- **CLI framework:** plain `argparse` (not `typer`) — one less dep, less ceremony, fine for a solo project
- Library code uses `logging`, not `print()`; `print` OK in scripts/CLI output

### 5.8 Git workflow

- Branch: `grace-iberlef-2026`
- Atomic commits with conventional prefixes (`feat:`, `fix:`, `test:`, `chore:`)
- Pre-commit hook active
- No force-pushes without user confirmation
- PRs at milestones: baseline ready → augmentation ready → final ready
- Per-PR `/self-reflect` feeds paper methods section

### 5.9 Dev-loop CLI

```bash
python scripts/audit_data.py downloaded_data/public_data/public_data/
python scripts/train_track1.py --config configs/track1/xlmr_large_base.yaml
python scripts/train_track2.py --config configs/track2/llm_claude_cot.yaml
python scripts/run_llm_baseline.py --track 2 --split dev --model claude-opus-4-6
python scripts/submit.py --track 1 --run experiments/runs/<tag>/ --validate-only
python scripts/submit.py --track 1 --run experiments/runs/<tag>/ --upload
```

---

## 6. Timeline, milestones, risks, non-goals

### 6.1 24-day plan

Day 0 is 2026-04-09 (today). Day 13 is 2026-04-22 (test set release). Day 24 is 2026-05-03 (submission deadline). Day 45 is 2026-05-24 (paper deadline).

| Day | Date | Phase | Deliverable | Checkpoint? |
|---|---|---|---|---|
| 0 | 04-09 | Design | Design doc + plan committed + plan-review-gate passed | ✅ User approval gate |
| 1 | 04-10 | Foundation | `grace/` scaffold (minimal pre-commit, argparse CLIs), `pyproject.toml`, `.env`, WSL2 setup if bitsandbytes needed | — |
| 1 | 04-10 | Foundation | Loaders + dataclasses + OfficialTokenizer + tokenizer-parity test | — |
| 2 | 04-11 | Foundation | `SpanAligner` (snap, validate, BIO encode/decode, **sliding window** for long abstracts) + scorer self-consistency test | — |
| 2 | 04-11 | Foundation | Manual data-reading pass: 15 Track 1 cases + 10 Track 2 cases, written observations | — |
| 3 | 04-12 | Rules + Eval | Rules-verification snapshot; scorer wrapper + ledger + diagnose.py + formatter + validator + package.py + user-facing submit.py CLI | ⚠️ Blocking for Track 2 closed-API decision |
| 4 | 04-13 | Baseline T1 | XLM-R-base local smoke training (overfit-check) + Kaggle notebook setup + checkpoint/resume logic | — |
| 5 | 04-14 | Baseline T1 | XLM-R-large BIO tagger + relation classifier 3 seeds on Kaggle; first ledger numbers | — |
| 6 | 04-15 | Baseline T2 | BETO sentence classifier + LLM reasoner infra (Anthropic + OpenAI only; no OpenRouter) + cache + budget cap | — |
| 7 | 04-16 | Baseline T2 | Prompt templates + exemplar library + premise extractor + relation reasoner | ✅ User checkpoint: review prompts + exemplars |
| 8 | 04-17 | Augmentation | Cross-lingual download + translate + project + **hand-labeled projection eval** (30 examples) | — |
| 9 | 04-18 | Augmentation | Entailment filter + Track 1 A0-A2 ablations on XLM-R-large | — |
| 10 | 04-19 | Augmentation | A3 class-weighted + A4 seed-averaged ablations + per-rare-class delta table | — |
| 11 | 04-20 | LLM + distill | Full LLM reasoner train+dev pass (builds cache); HP sweep on best Track 1 backbone | — |
| 12 | 04-21 | LLM + distill | Distilled student training (gold / gold+silver / +augmentation) + **conditioning ablation** (with vs. without correct_choice_id) | ✅ User checkpoint: dev scores review |
| **13** | **04-22** | **TEST RELEASE + Integration** | **Official test set released.** Download, smoke test, verify format. Run ensembles on dev for both tracks. Submit dev-phase if open. | ✅ User confirmation for any upload |
| 14-18 | 04-23…04-27 | Iteration | Error-analysis-driven fixes. **Hard rule: no new modules after Day 12.** Stop if Day 20 dev score within 3 points of Day 18. | — |
| 19 | 04-28 | Paper draft | Outline + Intro + Related Work sections; results tables auto-generated from ledger | ✅ User review: outline |
| 20 | 04-29 | Paper draft | Task Description + System + Experiments + Methods sections | — |
| 21 | 04-30 | Paper draft | Results + Error Analysis + Discussion sections | — |
| 22 | 05-01 | Final run | Regenerate test predictions with finalized ensemble; diagnostics. **Code freeze.** | — |
| 23 | 05-02 | Final run | Final submission packages; full scorer-parity check on dev | ✅ **User confirmation required** before upload |
| **24** | **05-03** | **SUBMIT** | **Upload Track 1 + Track 2 final submissions to Codabench** (≤3 of the 5 daily quota) | ✅ User confirms |
| 25-44 | 05-04…05-23 | Paper finish | Limitations, Conclusion, camera-ready polish, references.bib, discrepancy report to organizers | — |
| **45** | **05-24** | **Paper deadline** | **IberLEF system paper submitted to CEUR-WS** | ✅ User reviews before submit |
| 45+ | 05-25+ | Code release | `grace-iberlef-2026` repo public on GitHub; HF models if competitive; offset-alignment utility documented as reusable contribution | — |

**Stopping criteria (enforced during Days 14-18 iteration):**

1. **Convergence rule:** If Day 20 dev score is within 3 F1 points of Day 18 dev score, freeze — no more iteration.
2. **Augmentation rule:** If A2 (entailment-filtered silver) does not beat A0 (baseline) by ≥ 2 macro-F1 points, cut it from the final submission but keep the ablation for the paper.
3. **Module freeze:** No new Python modules after Day 12. Days 14-18 are hyperparameter tuning + error-analysis fixes only.
4. **Budget cap:** LLM spend hard-capped at $250 via `experiments/budget.yaml`. `LLMReasoner` aborts with `BudgetExceededError` before the call that would exceed the cap.
5. **Submission cap awareness:** Codabench typically allows 3-5 submissions/day. Budget at most 2 for the dev phase and 3 for the test phase across both tracks.

### 6.2 Human checkpoints

1. **Right after this spec** — approve full design, start execution
2. **After plan review gate** — metaswarm plan-review (required)
3. **Before execution method choice** — orchestrated / subagent-driven / parallel session
4. **Day 2** — rules-verification snapshot review
5. **Day 5** — Track 2 prompts + exemplars review
6. **Day 11** — dev-score review, pivot-or-iterate decision
7. **Day 12** — dev-phase Codabench upload confirmation
8. **Day 19** — paper outline review
9. **Day 23-24** — final submission upload confirmation
10. **Day 44-45** — final paper review before CEUR-WS submission

### 6.3 Risk register

| ID | Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|---|
| R1 | Offset misalignment → strict F1 = 0 | Critical | Medium | §2 invariants; `SpanAligner` unit tests; scorer self-consistency on every submission |
| R2 | Rules forbid closed LLMs on test set | High | Medium | Distilled student from day 1; rules verified by day 2 |
| R3 | Rare classes drag macro F1 | High | High | §3.4 augmentation ablation; class-weighted loss; per-class diagnostics |
| R4 | LLM API cost overrun | Medium | Low-Med | Hard budget cap; aggressive caching |
| R5 | Track 2 test-set specialty shift | Medium | Medium | Multi-specialty exemplars; ensemble robustness |
| R6 | Codabench SPA blocks rule verification | Low | High | Manual browser / email organizers as fallback |
| R7 | Kaggle/Colab session limits | Medium | Medium | Checkpoint every N steps; resume-from-checkpoint |
| R8 | Data version drift mid-competition | Medium | Low | Data fingerprints in every run |
| R9 | Official scorer bugs affecting official score | Medium | Low | Compare vs. our calculation; flag discrepancies publicly |
| R10 | Paper deadline slip | Low | Low | Paper drafting starts day 19, not day 25 |
| R11 | User time commitment changes | Medium | Low | Design is modular; scope-down to single track without restart |
| R12 | LLM hallucinated spans | Medium | Medium | Snap-to-boundary + fuzzy match + drop-unmatched + BETO fallback |

### 6.4 Non-goals

- Training a pretrained LLM from scratch
- Manual annotation of extra data
- Multi-modal extensions (images in MIR exams)
- Building a GUI
- Multilingual test-time submission — Spanish only
- Leaderboard-gaming tricks
- Reimplementing the scoring program — we import it
- Shipping a pip package before the deadline
- Beating SciBERT+GRU+CRF on Track 1 by a large margin — realistic ceiling constrained by rare-class data

### 6.5 Success criteria

**Hard success:**
- Both tracks submitted on time, valid format
- Both tracks in top half (≥ median)
- System paper submitted
- Code released publicly

**Stretch success:**
- Track 2 in top 3
- Track 1 in top half with clear rare-class contribution
- Paper accepted to IberLEF proceedings
- Methods replicated by independent user from README in < 2 hours

**Research success (independent of leaderboard):**
- Cross-lingual augmentation ablation yields reviewable rare-class recall claim
- Distilled student achieves ≥ 80% of LLM ensemble's Track 2 score at < 1% inference cost
- Offset-alignment library reusable for future Spanish clinical NLP shared tasks

### 6.6 Post-competition

- Paper deadline: 2026-05-24
- Code release: within 1 week of submission deadline
- HuggingFace model upload if competitive
- Dataset contribution: filtered cross-lingual augmentations (if licensing allows)

---

## 7. Sources

- Codabench competition page: https://www.codabench.org/competitions/13280/
- Official task site (redirects to Codabench): https://grace2026.hitz.eus
- Corpora-list announcement: http://www.mail-archive.com/corpora@list.elra.info/msg05615.html
- IberLEF 2026 tasks index: https://sites.google.com/view/iberlef-2026/tasks
- HiTZ multilingual-abstrct repo: https://github.com/hitz-zentroa/multilingual-abstrct
- AbstRCT-ES HuggingFace dataset: https://huggingface.co/datasets/HiTZ/AbstRCT-ES
- Yeginbergen & Agerri (2024), "Cross-lingual Argument Mining in the Medical Domain", arXiv:2301.10527
- Chizhikova et al. (2024), "CasiMedicos-Arg: A Medical Question Answering Dataset Annotated with Explanatory Argumentative Structures", arXiv:2410.05235
- casimedicos-arg HuggingFace dataset: https://huggingface.co/datasets/HiTZ/casimedicos-arg
- casimedicos-exp HuggingFace dataset: https://huggingface.co/datasets/HiTZ/casimedicos-exp
- Antidote CasiMedicos corpus repo: https://github.com/ixa-ehu/antidote-casimedicos
- AbstRCT corpus original: referenced in Yeginbergen & Agerri 2024
- Official Track 1 scoring program: `downloaded_data/track1_scoring_program/track1_scoring_program/track1_scoring_program.py`
- Official Track 2 scoring program: `downloaded_data/track2_scoring_program/track2_scoring_program/track2_scoring_program.py`
- Track 1 README: `downloaded_data/track1_scoring_program/track1_scoring_program/README-track1.md`
- Track 2 README: `downloaded_data/track2_scoring_program/track2_scoring_program/README.md`
- Data audit (2026-04-09): this document's §2.6

---

## Appendix A — Data audit snapshot (2026-04-09)

Produced by initial audit run on `downloaded_data/public_data/public_data/`:

**Format:** Track 1 and Track 2 both use `annotations` as a DICT (matching the scorer), not a flat list (matching the outdated instance example). Loaders must target the DICT format.

**Track 1 Train (350 cases):**
- Entity types: Premise 1536 (68%), Claim 666 (29%), MajorClaim 64 (3%)
- Relation types: Support 1213 (86%), Partial-Attack 169 (12%), Attack 36 (2.5%)
- Avg entities/case 6.47, avg relations/case 4.05
- Text length: avg 2209, median 2108, max 4370 chars

**Track 1 Dev (50 cases):**
- Entity types: Premise 218 (67%), Claim 99 (30%), MajorClaim 9 (3%)
- Relation types: Support 186 (85%), Partial-Attack 25 (11%), Attack 8 (4%)
- Text length: avg 2338, median 2233, max 3738 chars

**Track 2 Train (128 cases):**
- Entity types: Premise 572 (49%), Claim 586 (51%)
- Relation types: Support 613 (49%), Attack 646 (51%)
- Avg entities/case 9.05, avg relations/case 9.84
- Sentence relevance: 263 relevant (53%), 237 not-relevant (47%)
- Sentences/case avg 3.91
- Choices/case: 86 with 5 options, 42 with 4 options
- Text length: avg 632, median 594, max 1755 chars

**Track 2 Dev (24 cases):**
- Entity types: Premise 121 (52%), Claim 110 (48%)
- Relation types: Support 133 (49%), Attack 138 (51%)
- Sentence relevance: 61 relevant (60%), 41 not-relevant (40%)
- Choices/case: 18 with 5, 6 with 4
- Text length: avg 677, median 653, max 1300 chars

---

## Appendix B — Discrepancies flagged during design

1. **Instance-example format mismatch:** `instance_examples/track1_example.json` uses a flat `annotations` list plus top-level `relations`, but the actual training JSONs and the scoring program use a DICT-valued `annotations` with `entities` and `relations` keys. Loaders follow the scoring-program format. Non-blocking for us; worth reporting to organizers.

2. **Track 1 scorer `relaxed_mixed` bug:** In `track1_scoring_program.py` lines ~397-398 and ~521-522, the "relaxed_mixed" block uses strict-count variables (`fp_s`, `fn_s`) instead of relaxed variables (`fp_r`, `fn_r`). Affects only the complementary "relaxed_mixed" numbers, not the official score. Worth reporting to organizers during development window.

3. **Track 1 README copy-paste:** Opens with "Given a clinical case question with multiple-choice answers…" which is the Track 2 wording. Cosmetic.

4. **Metadata YAML typo:** Track 1 and Track 2 `metadata.yaml` in the scoring programs says "GRACE 2016" instead of "GRACE 2026". Cosmetic.

---

## Appendix C — Rules verification protocol (day 2 deliverable)

Procedure to produce `docs/superpowers/specs/2026-04-09-grace-rules-snapshot.md`:

1. Attempt `WebFetch` on `https://www.codabench.org/competitions/13280/` with prompt targeting "Terms", "Rules", "Participation", "External resources", "Pretrained models".
2. If SPA blocks rendering, attempt fetch of `https://www.codabench.org/api/competitions/13280/` or similar JSON endpoints.
3. Search corpora-list archive and HiTZ channel for explicit rule text.
4. If still blocked, ask the user to open the Codabench Terms tab in a browser and paste the verbatim rules text.
5. Final fallback: email organizers (with user approval), citing participation question.

Output: verbatim rule text, URL, fetch date, classification of allowed/disallowed for (a) closed LLMs, (b) ensembles, (c) extra corpora, (d) test-set submission count, (e) system paper requirements.

---

*End of design document.*
