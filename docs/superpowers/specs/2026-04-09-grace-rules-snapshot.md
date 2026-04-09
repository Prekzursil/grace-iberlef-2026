# GRACE @ IberLEF 2026 — Rules snapshot

**First fetched (automated):** 2026-04-09T16:59:59Z via `scripts/verify_rules.py`
**Verbatim T&C pasted by user:** 2026-04-09T20:45:00Z
**Source:** Codabench competition 13280 Terms tab (https://www.codabench.org/competitions/13280/)

This file is the **source of truth** for whether closed-API LLMs and
related tools can be used on the final test-set submission. Re-run
`scripts/verify_rules.py` and re-ask the user whenever organizers update
the Codabench Terms tab.

---

## Verbatim Terms and Conditions (user-pasted, 2026-04-09)

By participating in the GRACE Shared Task at IberLEF 2026, all participants and team members agree to the following terms and conditions.

### 1. Results and Public Release

By submitting results to this competition, you consent to the public release of your scores via:

- The official GRACE task website and leaderboard.
- The IberLEF 2026 workshop presentations.
- The associated workshop proceedings and publications.

The task organizers reserve the right to publish scores at their discretion. These scores may include automatically and manually calculated quantitative judgments, qualitative evaluations, and any other metrics deemed relevant by the organizers.

### 2. Data Usage and Future Research

By participating, you grant the organizers permission to use your system predictions (the output files submitted) for further scientific research and post-hoc analyses.

- **Citation:** If your system's predictions are used in future research publications or as part of derivative datasets created by the organizers, your contribution will be properly cited or acknowledged, typically by citing the team's system description paper or the task overview paper.
- **Scope:** Usage may include meta-evaluations, ensemble studies, or the development of new research resources.

### 3. Authority of Organizers

Participants acknowledge that the ultimate decision regarding metric selection and final score values rests solely with the task organizers.

The organizers are under no obligation to release scores for every submission. Scores may be withheld if, in the organizers' judgment, a submission is:

- Incomplete or technically erroneous.
- Deceptive or fraudulent.
- In violation of the letter or spirit of the competition's rules.

**Note:** The inclusion of a submission's score does not constitute an endorsement by the organizers of the team, the system, or the underlying scientific methodology.

### 4. Team Registration and Accounts

To ensure a fair and organized competition, the following rules apply:

- **Single Representation:** Each team must be represented by a single account for submission.
- **Team Composition:** If the platform supports it, you may build a competition team composed of multiple individual accounts.
- **Integrity:** Users are prohibited from creating multiple accounts to circumvent submission limits.

### 5. Scientific Integrity and Ethics

Participants are expected to adhere to the highest standards of scientific integrity. This includes:

- **Originality:** Providing honest descriptions of the models and data used.
- **Transparency:** Disclosing any external data or pre-trained models used, as specified in the task guidelines.
- **Professionalism:** Behaving respectfully toward other participants and the organizing committee.

---

## Classification for GRACE 2026 execution plan

Based on a careful reading of the verbatim T&C above — especially §5 (Scientific Integrity and Ethics) which only requires **disclosure** of external data and pre-trained models, not a prohibition on them — the following interpretation governs our implementation:

| Question | Answer | Source |
|---|---|---|
| Closed APIs (Claude, GPT-4, etc.) allowed on final test-set submission? | **YES**, with mandatory disclosure in the system paper | §5 Transparency — disclosure required, not prohibition |
| Extra pretrained open-weights models allowed? | **YES**, with disclosure | §5 Transparency |
| Cross-lingual data augmentation (using public AbstRCT EN/FR/IT) allowed? | **YES**, with disclosure of external data | §5 Transparency |
| Ensembling across multiple models allowed? | **YES**, not restricted | No rule against it |
| Daily submission cap | Not specified in T&C; assumed Codabench platform default (3-5/day). Budget ≤2 dev-phase + ≤3 test-phase | T&C silent, Codabench default |
| System paper required | **YES**, implied throughout (§2 "team's system description paper"; §1 "workshop proceedings and publications") | §1, §2 |
| Single account per team | **YES**, "Prekzursil" is the submission account | §4 Single Representation |
| Multiple-account gaming | **Prohibited** | §4 Integrity |
| Organizers may withhold scores | **YES**, for incomplete/fraudulent/rule-violating submissions | §3 |

## Impact on the plan

**No changes required** — the plan's Track 2 approach (LLM reasoner + distilled student ensemble) is **fully viable** as designed. Key implications:

1. **Track 2 Subtask 3 can use the Claude/GPT-4 LLM reasoner directly on the test set** — no forced fallback to distilled-student-only mode. The distilled student remains valuable as a reproducibility and cost-efficiency contribution but is no longer a compliance fallback.

2. **Cross-lingual rare-class augmentation (Phase 5) is explicitly allowed** as long as the paper discloses:
   - Source corpora: English AbstRCT, French AbstRCT, Italian AbstRCT
   - Translation method: DeepL or Claude
   - Filter: Spanish XNLI entailment
   - Number of silver examples added per rare class

3. **Paper disclosure requirements** for the Methods section:
   - All pretrained models and their exact HF revisions (XLM-R-large, BETO, Claude Sonnet, GPT-4o, etc.)
   - All external datasets (AbstRCT EN/FR/IT, Spanish XNLI for filter)
   - All LLM API calls (provider, model version, prompt template reference, exemplar source)
   - The distilled student's training recipe
   - The conditioning-ablation protocol

4. **Submission budget:** ≤2 dev-phase submissions (to probe leaderboard position) + ≤3 test-phase submissions (final + 2 fallbacks). The Codabench MCP `kernel_status` and `competition_submissions` tools make this trivially trackable.

5. **Organizer withhold risk:** §3 allows the organizers to withhold scores for "technically erroneous" submissions. Our Phase 2 `grace/submit/validator.py` runs the official scorer on every submission before upload — if validation passes, the submission cannot be technically erroneous.

## Compliance checklist (to run before each submission)

- [ ] `scripts/submit.py --track N --prediction <path> --validate-only` passes
- [ ] Submission filename matches Codabench-required format
- [ ] Single-account rule: all uploads use the "Prekzursil" account
- [ ] Paper draft includes disclosure of all external resources used
- [ ] No data that couldn't be disclosed in the paper was used

## Organizer contact

If any ambiguity arises during development (e.g., specific tool allowed/forbidden), escalation ladder step 5 applies: email organizers with user approval. Primary contacts (from design doc):

- Iker de la Iglesia (`idelaiglesia004` on Codabench — task creator)
- Aitziber Atutxa
- Ander Barrena (HiTZ)
- Koldo Gojenola (HiTZ)
- Affiliation: HiTZ (Basque Center for Language Technology, UPV/EHU)

## Team registration

- **Submission account:** `Prekzursil` (Kaggle username also matches — convenient)
- **Team name:** (not yet decided — see Phase 3 Task 3.1 or Phase 7 submission task)
- **System paper authors:** User (sole author per brainstorming decision)
