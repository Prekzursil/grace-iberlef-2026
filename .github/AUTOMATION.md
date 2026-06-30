# Green-by-default automation (3 tiers)

This repo stays green-by-default through a layered, cost-aware automation stack.
Each tier escalates only when the cheaper tier below it cannot resolve the issue.

## Tier 1 - Dependabot auto-merge (free, native)

- Workflow: `.github/workflows/dependabot-auto-merge.yml`
- On a Dependabot PR it enables GitHub native auto-merge (`gh pr merge --auto --squash`).
- The repo setting "Allow auto-merge" is enabled, so GitHub completes the merge
  ONLY after every required check is GREEN. It never force-merges a red PR.

## Tier 2 - GitHub Models autofix (free, in-Actions) - PRIMARY auto-fix

- Workflow: `.github/workflows/autofix-models.yml`
- Triggers ONLY on a failed default-branch `quality` run (`workflow_run`
  conclusion == failure) or a newly created CodeQL alert.
- Uses the free GitHub Models inference action (`openai/gpt-4o-mini`) with the
  built-in `GITHUB_TOKEN` to draft a unified-diff fix, applies it, and opens a
  PR labelled `autofix`.
- NEVER auto-merges. The normal `quality` + `codeql` checks and a human review
  gate every autofix PR. Bounded: one PR per trigger, and it exits quietly when
  no safe patch can be produced.

## Tier 3 - Copilot coding agent (student subscription) - JUDICIOUS escalation

GitHub Copilot premium requests are FINITE, so this tier is manual and
deliberate, never wired to fire on every failure.

Escalate to Copilot ONLY when the free Tier-2 Models autofix cannot resolve a
stuck failing check:

1. Confirm the `autofix-models` run produced `NO_FIX` (or its PR still fails CI
   after review).
2. Open (or reuse) a GitHub Issue describing the stuck failing check, with the
   failing job log link and the root-cause summary.
3. Assign that issue to `@copilot` (Copilot coding agent) to draft a fix PR.
4. Review and merge the resulting PR through the normal `quality` + `codeql`
   gates - never bypass checks, never `--no-verify`, never force-merge.

Do NOT automate Tier 3. Use it sparingly to conserve premium requests.
