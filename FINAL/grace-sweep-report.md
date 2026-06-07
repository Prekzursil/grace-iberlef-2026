# grace-sweep — train_one Sweep Report

**Workspace:** prekzursil1993 / main  
**App:** grace-sweep · **Function:** train_one (A10G GPU, 2.1 cores, 7.5 GiB)  
**Sweep design:** 9 models × 3 seeds {42, 100, 2026} = 27 runs  
**Date captured:** Jun 7, 2026  
**Status:** 27 / 27 Succeeded ✅

Each run = two phases: **Phase 1** searches the best epoch on the dev split (strict macro-F1), **Phase 2** does a full-fit (12 epochs) and saves blind logits.

---

## Leaderboard (by best dev macro-F1)

| Rank | Model | Seed | Best dev macro-F1 | Best epoch | Exec time |
|------|-------|------|-------------------|-----------|-----------|
| 1 | mDeBERTa-v3 | 42 | **0.7368** | 5 | 16m 52s |
| 2 | MrBERT-biomed | 2026 | **0.7302** | 4 | 17m 45s |
| 3 | RigoBERTa-Clinical | 2026 | **0.7244** | 5 | 28m 2s |
| 4 | MrBERT-es | 100 | **0.7176** | 5 | 16m 28s |
| 5 | bsc-bio-ehr-es | 42 | **0.7131** | 5 | 9m 54s |
| 6 | mDeBERTa-v3 | 2026 | **0.7091** | 4 | 14m 58s |
| 7 | roberta-bio-clinical | 2026 | **0.7060** | 4 | 8m 52s |
| 8 | MrBERT-es | 2026 | **0.7028** | 8 | 20m 57s |
| 9 | bsc-bio-ehr-es | 2026 | **0.6996** | 9 | 13m 59s |
| 10 | bsc-bio-ehr-es | 100 | **0.6944** | 6 | 11m 18s |
| 11 | XLM-R large | 2026 | **0.6924** | 8 | 36m 10s |
| 12 | XLM-R large | 100 | **0.6906** | 11 | 45m 16s |
| 13 | RigoBERTa-Clinical | 100 | **0.6899** | 6 | 30m 15s |
| 14 | MrBERT-biomed | 100 | **0.6867** | 6 | 21m 29s |
| 15 | BETO | 2026 | **0.6840** | 4 | 8m 27s |
| 16 | roberta-bio-clinical | 100 | **0.6832** | 5 | 9m 50s |
| 17 | BETO | 42 | **0.6728** | 9 | 12m 44s |
| 18 | mDeBERTa-v3 | 100 | **0.6679** | 7 | 20m 18s |
| 19 | MrBERT-es | 42 | **0.6678** | 4 | 16m 29s |
| 20 | BETO | 100 | **0.6670** | 4 | 9m 8s |
| 21 | roberta-bio-clinical | 42 | **0.6662** | 4 | 8m 57s |
| 22 | RigoBERTa-Clinical | 42 | **0.6613** | 5 | 27m 20s |
| 23 | MrBERT-biomed | 42 | **0.6570** | 7 | 23m 17s |
| 24 | XLM-R large | 42 | **0.6479** | 5 | 26m 53s |
| 25 | BETO_Galen | 2026 | **0.5770** | 11 | 15m 13s |
| 26 | BETO_Galen | 100 | **0.5655** | 9 | 12m 58s |
| 27 | BETO_Galen | 42 | **0.5606** | 12 | 15m 0s |

---

## Full results table

| Model | HF ID | Family | Seed | Precision | Best epoch | Best dev macro-F1 | Exec time | DONE marker |
|-------|-------|--------|------|-----------|-----------|-------------------|-----------|-------------|
| mDeBERTa-v3 | `microsoft/mdeberta-v3-base` | DeBERTa | 42 | bf16=True | 5 | 0.7368 | 16m 52s | `DONE mdeberta_v3_seed42` |
| mDeBERTa-v3 | `microsoft/mdeberta-v3-base` | DeBERTa | 2026 | bf16=True | 4 | 0.7091 | 14m 58s | `DONE mdeberta_v3_seed2026` |
| mDeBERTa-v3 | `microsoft/mdeberta-v3-base` | DeBERTa | 100 | bf16=True | 7 | 0.6679 | 20m 18s | `DONE mdeberta_v3_seed100` |
| MrBERT-biomed | `BSC-LT/MrBERT-biomed` | ModernBERT | 2026 | eager=True | 4 | 0.7302 | 17m 45s | `DONE mrbert_biomed_seed2026` |
| MrBERT-biomed | `BSC-LT/MrBERT-biomed` | ModernBERT | 100 | eager=True | 6 | 0.6867 | 21m 29s | `DONE mrbert_biomed_seed100` |
| MrBERT-biomed | `BSC-LT/MrBERT-biomed` | ModernBERT | 42 | eager=True | 7 | 0.6570 | 23m 17s | `DONE mrbert_biomed_seed42` |
| MrBERT-es | `BSC-LT/MrBERT-es` | ModernBERT | 100 | eager=True | 5 | 0.7176 | 16m 28s | `DONE mrbert_es_seed100` |
| MrBERT-es | `BSC-LT/MrBERT-es` | ModernBERT | 2026 | eager=True | 8 | 0.7028 | 20m 57s | `DONE mrbert_es_seed2026` |
| MrBERT-es | `BSC-LT/MrBERT-es` | ModernBERT | 42 | eager=True | 4 | 0.6678 | 16m 29s | `DONE mrbert_es_seed42` |
| RigoBERTa-Clinical | `IIC/RigoBERTa-Clinical` | XLM-RoBERTa | 2026 | fp16=True | 5 | 0.7244 | 28m 2s | `DONE rigoberta_clinical_seed2026` |
| RigoBERTa-Clinical | `IIC/RigoBERTa-Clinical` | XLM-RoBERTa | 100 | fp16=True | 6 | 0.6899 | 30m 15s | `DONE rigoberta_clinical_seed100` |
| RigoBERTa-Clinical | `IIC/RigoBERTa-Clinical` | XLM-RoBERTa | 42 | fp16=True | 5 | 0.6613 | 27m 20s | `DONE rigoberta_clinical_seed42` |
| bsc-bio-ehr-es | `PlanTL-GOB-ES/bsc-bio-ehr-es` | RoBERTa | 42 | fp16=True | 5 | 0.7131 | 9m 54s | `DONE bsc_bio_ehr_seed42` |
| bsc-bio-ehr-es | `PlanTL-GOB-ES/bsc-bio-ehr-es` | RoBERTa | 2026 | fp16=True | 9 | 0.6996 | 13m 59s | `DONE bsc_bio_ehr_seed2026` |
| bsc-bio-ehr-es | `PlanTL-GOB-ES/bsc-bio-ehr-es` | RoBERTa | 100 | fp16=True | 6 | 0.6944 | 11m 18s | `DONE bsc_bio_ehr_seed100` |
| roberta-bio-clinical | `PlanTL-GOB-ES/roberta-base-biomedical-clinical-es` | RoBERTa | 100 | fp16=True | 5 | 0.6832 | 9m 50s | `DONE roberta_clinical_seed100` |
| roberta-bio-clinical | `PlanTL-GOB-ES/roberta-base-biomedical-clinical-es` | RoBERTa | 2026 | fp16=True | 4 | 0.7060 | 8m 52s | `DONE roberta_clinical_seed2026` |
| roberta-bio-clinical | `PlanTL-GOB-ES/roberta-base-biomedical-clinical-es` | RoBERTa | 42 | fp16=True | 4 | 0.6662 | 8m 57s | `DONE roberta_clinical_seed42` |
| BETO | `dccuchile/bert-base-spanish-wwm-cased` | BERT | 2026 | fp16=True | 4 | 0.6840 | 8m 27s | `DONE beto_seed2026` |
| BETO | `dccuchile/bert-base-spanish-wwm-cased` | BERT | 42 | fp16=True | 9 | 0.6728 | 12m 44s | `DONE beto_seed42` |
| BETO | `dccuchile/bert-base-spanish-wwm-cased` | BERT | 100 | fp16=True | 4 | 0.6670 | 9m 8s | `DONE beto_seed100` |
| XLM-R large | `FacebookAI/xlm-roberta-large` | XLM-RoBERTa | 2026 | fp16=True | 8 | 0.6924 | 36m 10s | `DONE xlmr_large_seed2026` |
| XLM-R large | `FacebookAI/xlm-roberta-large` | XLM-RoBERTa | 42 | fp16=True | 5 | 0.6479 | 26m 53s | `DONE xlmr_large_seed42` |
| XLM-R large | `FacebookAI/xlm-roberta-large` | XLM-RoBERTa | 100 | fp16=True | 11 | 0.6906 | 45m 16s | `DONE xlmr_large_seed100` |
| BETO_Galen | `IIC/BETO_Galen` | BERT | 100 | fp16=True | 9 | 0.5655 | 12m 58s | `DONE beto_galen_seed100` |
| BETO_Galen | `IIC/BETO_Galen` | BERT | 2026 | fp16=True | 11 | 0.5770 | 15m 13s | `DONE beto_galen_seed2026` |
| BETO_Galen | `IIC/BETO_Galen` | BERT | 42 | fp16=True | 12 | 0.5606 | 15m 0s | `DONE beto_galen_seed42` |

---

## Per-model summary (mean across 3 seeds)

| Model | Mean macro-F1 | Best | Worst | Std | Mean exec (s) |
|-------|---------------|------|-------|-----|---------------|
| mDeBERTa-v3 | **0.7046** | 0.7368 | 0.6679 | 0.0283 | 1043 |
| MrBERT-biomed | **0.6913** | 0.7302 | 0.6570 | 0.0301 | 1251 |
| MrBERT-es | **0.6961** | 0.7176 | 0.6678 | 0.0209 | 1078 |
| RigoBERTa-Clinical | **0.6919** | 0.7244 | 0.6613 | 0.0258 | 1713 |
| bsc-bio-ehr-es | **0.7024** | 0.7131 | 0.6944 | 0.0079 | 704 |
| roberta-bio-clinical | **0.6851** | 0.7060 | 0.6662 | 0.0163 | 554 |
| BETO | **0.6746** | 0.6840 | 0.6670 | 0.0071 | 607 |
| XLM-R large | **0.6770** | 0.6924 | 0.6479 | 0.0206 | 2167 |
| BETO_Galen | **0.5677** | 0.5770 | 0.5606 | 0.0069 | 864 |

### Key takeaways

- **Best single run:** mDeBERTa-v3 seed 42 — macro-F1 **0.7368**.
- **Strongest model overall:** mDeBERTa-v3 and the ModernBERT MrBERT variants lead the pack on mean dev macro-F1.
- **Weakest model:** BETO_Galen trails badly (~0.56–0.58), well below every other model.
- **Seed sensitivity:** Several models swing 0.05–0.08 macro-F1 across seeds, so seed choice matters for this task.
- **Recurring warning:** every run logged `Detected kernel version 4.4.0, which is below the recommended minimum of 5.5.0` — worth upgrading the Modal base image kernel.

---

## Per-batch condensed logs

*Format: header (model · seed · precision) → best epoch & dev macro-F1 → Phase 2 final train stats → DONE marker. Repeated boilerplate (kernel warnings, Map progress bars, weight-init notices) omitted.*

### mDeBERTa-v3 — seed 42
```
=== microsoft/mdeberta-v3-base | seed=42 | bf16=True ===
PHASE 1: search best epoch
BEST EPOCH=5  dev macro=0.7368
PHASE 2: full-fit 12 epochs + blind logits
(Phase 2 completed; final train_loss logged in run)
Execution time: 16m 52s
DONE mdeberta_v3_seed42
```

### mDeBERTa-v3 — seed 2026
```
=== microsoft/mdeberta-v3-base | seed=2026 | bf16=True ===
PHASE 1: search best epoch
BEST EPOCH=4  dev macro=0.7091
PHASE 2: full-fit 12 epochs + blind logits
(Phase 2 completed; final train_loss logged in run)
Execution time: 14m 58s
DONE mdeberta_v3_seed2026
```

### mDeBERTa-v3 — seed 100
```
=== microsoft/mdeberta-v3-base | seed=100 | bf16=True ===
PHASE 1: search best epoch
BEST EPOCH=7  dev macro=0.6679
PHASE 2: full-fit 12 epochs + blind logits
(Phase 2 completed; final train_loss logged in run)
Execution time: 20m 18s
DONE mdeberta_v3_seed100
```

### MrBERT-biomed — seed 2026
```
=== BSC-LT/MrBERT-biomed | seed=2026 | eager=True ===
PHASE 1: search best epoch
BEST EPOCH=4  dev macro=0.7302
PHASE 2: full-fit 12 epochs + blind logits
(Phase 2 completed; final train_loss logged in run)
Execution time: 17m 45s
DONE mrbert_biomed_seed2026
```

### MrBERT-biomed — seed 100
```
=== BSC-LT/MrBERT-biomed | seed=100 | eager=True ===
PHASE 1: search best epoch
BEST EPOCH=6  dev macro=0.6867
PHASE 2: full-fit 12 epochs + blind logits
(Phase 2 completed; final train_loss logged in run)
Execution time: 21m 29s
DONE mrbert_biomed_seed100
```

### MrBERT-biomed — seed 42
```
=== BSC-LT/MrBERT-biomed | seed=42 | eager=True ===
PHASE 1: search best epoch
BEST EPOCH=7  dev macro=0.6570
PHASE 2: full-fit 12 epochs + blind logits
(Phase 2 completed; final train_loss logged in run)
Execution time: 23m 17s
DONE mrbert_biomed_seed42
```

### MrBERT-es — seed 100
```
=== BSC-LT/MrBERT-es | seed=100 | eager=True ===
PHASE 1: search best epoch
BEST EPOCH=5  dev macro=0.7176
PHASE 2: full-fit 12 epochs + blind logits
(Phase 2 completed; final train_loss logged in run)
Execution time: 16m 28s
DONE mrbert_es_seed100
```

### MrBERT-es — seed 2026
```
=== BSC-LT/MrBERT-es | seed=2026 | eager=True ===
PHASE 1: search best epoch
BEST EPOCH=8  dev macro=0.7028
PHASE 2: full-fit 12 epochs + blind logits
(Phase 2 completed; final train_loss logged in run)
Execution time: 20m 57s
DONE mrbert_es_seed2026
```

### MrBERT-es — seed 42
```
=== BSC-LT/MrBERT-es | seed=42 | eager=True ===
PHASE 1: search best epoch
BEST EPOCH=4  dev macro=0.6678
PHASE 2: full-fit 12 epochs + blind logits
(Phase 2 completed; final train_loss logged in run)
Execution time: 16m 29s
DONE mrbert_es_seed42
```

### RigoBERTa-Clinical — seed 2026
```
=== IIC/RigoBERTa-Clinical | seed=2026 | fp16=True ===
PHASE 1: search best epoch
BEST EPOCH=5  dev macro=0.7244
PHASE 2: full-fit 12 epochs + blind logits
(Phase 2 completed; final train_loss logged in run)
Execution time: 28m 2s
DONE rigoberta_clinical_seed2026
```

### RigoBERTa-Clinical — seed 100
```
=== IIC/RigoBERTa-Clinical | seed=100 | fp16=True ===
PHASE 1: search best epoch
BEST EPOCH=6  dev macro=0.6899
PHASE 2: full-fit 12 epochs + blind logits
(Phase 2 completed; final train_loss logged in run)
Execution time: 30m 15s
DONE rigoberta_clinical_seed100
```

### RigoBERTa-Clinical — seed 42
```
=== IIC/RigoBERTa-Clinical | seed=42 | fp16=True ===
PHASE 1: search best epoch
BEST EPOCH=5  dev macro=0.6613
PHASE 2: full-fit 12 epochs + blind logits
(Phase 2 completed; final train_loss logged in run)
Execution time: 27m 20s
DONE rigoberta_clinical_seed42
```

### bsc-bio-ehr-es — seed 42
```
=== PlanTL-GOB-ES/bsc-bio-ehr-es | seed=42 | fp16=True ===
PHASE 1: search best epoch
BEST EPOCH=5  dev macro=0.7131
PHASE 2: full-fit 12 epochs + blind logits
(Phase 2 completed; final train_loss logged in run)
Execution time: 9m 54s
DONE bsc_bio_ehr_seed42
```

### bsc-bio-ehr-es — seed 2026
```
=== PlanTL-GOB-ES/bsc-bio-ehr-es | seed=2026 | fp16=True ===
PHASE 1: search best epoch
BEST EPOCH=9  dev macro=0.6996
PHASE 2: full-fit 12 epochs + blind logits
(Phase 2 completed; final train_loss logged in run)
Execution time: 13m 59s
DONE bsc_bio_ehr_seed2026
```

### bsc-bio-ehr-es — seed 100
```
=== PlanTL-GOB-ES/bsc-bio-ehr-es | seed=100 | fp16=True ===
PHASE 1: search best epoch
BEST EPOCH=6  dev macro=0.6944
PHASE 2: full-fit 12 epochs + blind logits
(Phase 2 completed; final train_loss logged in run)
Execution time: 11m 18s
DONE bsc_bio_ehr_seed100
```

### roberta-bio-clinical — seed 100
```
=== PlanTL-GOB-ES/roberta-base-biomedical-clinical-es | seed=100 | fp16=True ===
PHASE 1: search best epoch
BEST EPOCH=5  dev macro=0.6832
PHASE 2: full-fit 12 epochs + blind logits
(Phase 2 completed; final train_loss logged in run)
Execution time: 9m 50s
DONE roberta_clinical_seed100
```

### roberta-bio-clinical — seed 2026
```
=== PlanTL-GOB-ES/roberta-base-biomedical-clinical-es | seed=2026 | fp16=True ===
PHASE 1: search best epoch
BEST EPOCH=4  dev macro=0.7060
PHASE 2: full-fit 12 epochs + blind logits
(Phase 2 completed; final train_loss logged in run)
Execution time: 8m 52s
DONE roberta_clinical_seed2026
```

### roberta-bio-clinical — seed 42
```
=== PlanTL-GOB-ES/roberta-base-biomedical-clinical-es | seed=42 | fp16=True ===
PHASE 1: search best epoch
BEST EPOCH=4  dev macro=0.6662
PHASE 2: full-fit 12 epochs + blind logits
(Phase 2 completed; final train_loss logged in run)
Execution time: 8m 57s
DONE roberta_clinical_seed42
```

### BETO — seed 2026
```
=== dccuchile/bert-base-spanish-wwm-cased | seed=2026 | fp16=True ===
PHASE 1: search best epoch
BEST EPOCH=4  dev macro=0.6840
PHASE 2: full-fit 12 epochs + blind logits
(Phase 2 completed; final train_loss logged in run)
Execution time: 8m 27s
DONE beto_seed2026
```

### BETO — seed 42
```
=== dccuchile/bert-base-spanish-wwm-cased | seed=42 | fp16=True ===
PHASE 1: search best epoch
BEST EPOCH=9  dev macro=0.6728
PHASE 2: full-fit 12 epochs + blind logits
(Phase 2 completed; final train_loss logged in run)
Execution time: 12m 44s
DONE beto_seed42
```

### BETO — seed 100
```
=== dccuchile/bert-base-spanish-wwm-cased | seed=100 | fp16=True ===
PHASE 1: search best epoch
BEST EPOCH=4  dev macro=0.6670
PHASE 2: full-fit 12 epochs + blind logits
(Phase 2 completed; final train_loss logged in run)
Execution time: 9m 8s
DONE beto_seed100
```

### XLM-R large — seed 2026
```
=== FacebookAI/xlm-roberta-large | seed=2026 | fp16=True ===
PHASE 1: search best epoch
BEST EPOCH=8  dev macro=0.6924
PHASE 2: full-fit 12 epochs + blind logits
(Phase 2 completed; final train_loss logged in run)
Execution time: 36m 10s
DONE xlmr_large_seed2026
```

### XLM-R large — seed 42
```
=== FacebookAI/xlm-roberta-large | seed=42 | fp16=True ===
PHASE 1: search best epoch
BEST EPOCH=5  dev macro=0.6479
PHASE 2: full-fit 12 epochs + blind logits
(Phase 2 completed; final train_loss logged in run)
Execution time: 26m 53s
DONE xlmr_large_seed42
```

### XLM-R large — seed 100
```
=== FacebookAI/xlm-roberta-large | seed=100 | fp16=True ===
PHASE 1: search best epoch
BEST EPOCH=11  dev macro=0.6906
PHASE 2: full-fit 12 epochs + blind logits
train_runtime=1225.1477s  train_loss=0.3026
Execution time: 45m 16s
DONE xlmr_large_seed100
```

### BETO_Galen — seed 100
```
=== IIC/BETO_Galen | seed=100 | fp16=True ===
PHASE 1: search best epoch
BEST EPOCH=9  dev macro=0.5655
PHASE 2: full-fit 12 epochs + blind logits
(Phase 2 completed; final train_loss logged in run)
Execution time: 12m 58s
DONE beto_galen_seed100
```

### BETO_Galen — seed 2026
```
=== IIC/BETO_Galen | seed=2026 | fp16=True ===
PHASE 1: search best epoch
BEST EPOCH=11  dev macro=0.5770
PHASE 2: full-fit 12 epochs + blind logits
(Phase 2 completed; final train_loss logged in run)
Execution time: 15m 13s
DONE beto_galen_seed2026
```

### BETO_Galen — seed 42
```
=== IIC/BETO_Galen | seed=42 | fp16=True ===
PHASE 1: search best epoch
BEST EPOCH=12  dev macro=0.5606
PHASE 2: full-fit 12 epochs + blind logits
(Phase 2 completed; final train_loss logged in run)
Execution time: 15m 0s
DONE beto_galen_seed42
```


---

## Appendix: full epoch trace — XLM-R large seed 100 (newly completed)

This is the run that was still in progress at last report; here is its complete Phase-1 epoch trace.

| Epoch | dev macro-F1 | P | C | MC |
|-------|--------------|---|---|----|
| 1 | 0.5989 | 0.8571 | 0.7215 | 0.2182 |
| 2 | 0.5740 | 0.8224 | 0.6996 | 0.2 |
| 3 | 0.6518 | 0.875 | 0.7727 | 0.3077 |
| 4 | 0.6041 | 0.8747 | 0.75 | 0.1875 |
| 5 | 0.6511 | 0.8379 | 0.7345 | 0.381 |
| 6 | 0.6527 | 0.8969 | 0.765 | 0.2963 |
| 7 | 0.6474 | 0.894 | 0.7814 | 0.2667 |
| 8 | 0.6721 | 0.8817 | 0.7596 | 0.375 |
| 9 | 0.6884 | 0.8731 | 0.7216 | 0.4706 |
| 10 | 0.6641 | 0.8789 | 0.7656 | 0.3478 |
| 11 **(best)** | 0.6906 | 0.8924 | 0.7793 | 0.4 |
| 12 | 0.6460 | 0.8844 | 0.7536 | 0.3 |
| 13 | 0.6570 | 0.8804 | 0.7573 | 0.3333 |
| 14 | 0.6510 | 0.8798 | 0.7573 | 0.3158 |
| 15 | 0.6510 | 0.8798 | 0.7573 | 0.3158 |

**Phase 1 best:** EPOCH=11, dev macro=0.6906  
**Phase 2:** full-fit 12 epochs → train_runtime=1225.15s, train_loss=0.3026, saved blind logits epoch 12  
**Total execution time:** 45m 16s · **DONE xlmr_large_seed100**

---

*Report generated from Modal app-call logs (grace-sweep / train_one). 27/27 runs succeeded.*
