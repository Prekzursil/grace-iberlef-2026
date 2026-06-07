# GRACE @ IberLEF 2026 — Track 1 Report Deliverables

Presentation + report assets for the 27-run benchmark and ensemble.

## Deliverables

| File | What |
|------|------|
| `GRACE_Track1_IberLEF2026.pptx` | 28-slide presentation (dark dashboard theme) |
| `GRACE_Track1_Report.docx` | 10-page technical report |
| `GRACE_Track1_Report.pdf` | PDF export of the report |
| `figures/*.png` | 13 dashboard-style figures (200 DPI) |
| `slides_png/*.PNG` | Each slide exported to PNG (for quick preview / sharing) |
| `report_data.json` | **Single source of truth** — every number in every artifact comes from here |
| `../report_out/GRACE_track1_submission.json` | Blind-test submission (all-9 ensemble) |

## Headline numbers

- Best single model: **mDeBERTa-v3-base 0.705 ± 0.028** dev strict macro-F1
- **All-9 ensemble: 0.743** (official scorer) — bias-free, beats top-3 (0.742) and best single (+3.9 pts)
- Blind submission: 2,474 abstracts → 11,752 components, 0 offset mismatches

## Regenerate everything

Run from `FINAL/`:

```bash
# 1. ensemble numbers + blind submission (Modal) — already done; outputs in ../report_out/
python -m modal run modal_ensemble.py
python download_report.py            # pull grace-report volume -> ../report_out/

# 2. consolidate all numbers into one source of truth
python report/collect_data.py        # -> report/report_data.json

# 3. figures, then documents
python report/make_figures.py        # -> report/figures/*.png
python report/build_docx.py          # -> report/GRACE_Track1_Report.docx
python report/build_pptx.py          # -> report/GRACE_Track1_IberLEF2026.pptx
```

### Render to PNG/PDF for QA (Windows, no LibreOffice needed)

```powershell
# PPTX -> per-slide PNGs (PowerPoint COM, SaveAs format 18 = PNG)
$ppt = New-Object -ComObject PowerPoint.Application
$p = $ppt.Presentations.Open("$PWD\GRACE_Track1_IberLEF2026.pptx", $true, $false, $false)
$p.SaveAs("$PWD\slides_png", 18); $p.Close(); $ppt.Quit()

# DOCX -> PDF (Word COM, format 17 = PDF)
$w = New-Object -ComObject Word.Application
$d = $w.Documents.Open("$PWD\GRACE_Track1_Report.docx", $false, $true)
$d.ExportAsFixedFormat("$PWD\GRACE_Track1_Report.pdf", 17); $d.Close($false); $w.Quit()
```

## Design notes

- One dark theme (`style.py`) across all figures so charts read as Modal-style dashboard tiles
  and blend into the dark slides; on the white DOCX pages they embed as dashboard panels.
- Honest methodology throughout: report mean ± std (not best seed); submit the zero-peek all-9
  ensemble (not the dev-selected top-3); MajorClaim's low F1 framed as data scarcity, not model failure.
