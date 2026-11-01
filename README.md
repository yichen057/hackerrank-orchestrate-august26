# Message Notification Router

[![tests](https://github.com/yichen057/hackerrank-orchestrate-august26/actions/workflows/tests.yml/badge.svg)](https://github.com/yichen057/hackerrank-orchestrate-august26/actions/workflows/tests.yml)

Routes every message in a 110-row WhatsApp-style dataset to `notify`, `digest`, or
`mute`, across text, images, and voice notes. Built in 24 hours for HackerRank
Orchestrate, August 2026.

**The design question is where the model's judgment is allowed to apply.** A
deterministic pipeline prepares the evidence and enforces the output contract; the
model only chooses among options the code has already narrowed. Its entire output is
constrained by a per-request JSON Schema whose evidence field is an enum of exactly
the message IDs retrieval selected, so a cited ID that was never offered is refused
before it reaches application code.

## At a glance

| | |
|---|---|
| Action accuracy | `27/30` (90%) under leave-one-out, identical in both recorded runs |
| Safety errors | `0` false mutes and `0` missed mutes in that evaluation, across 10 mute rows |
| Fallback rows shipped | `0` of `110` |
| Value of the model | removing it drops action accuracy to `11/30`, a gap of `+53.33` points |
| Runtime dependencies | 4: `openai`, `pydantic`, `Pillow`, `av` |
| Tests | 65, no API calls: `pip install -r requirements-dev.txt && python -m pytest tests/` |

Message-type accuracy and evidence match move by a row between runs. [Evaluation](#evaluation)
reports both runs side by side rather than the better one, and
[Known limitations](#known-limitations) is deliberately not a short list.

## Where to look

| Path | What it holds |
|---|---|
| [`docs/architecture.md`](./docs/architecture.md) | the full design, stage by stage |
| [`problem_statement.md`](./problem_statement.md) | the official task, fields, and allowed values |
| `code/pipeline.py` | the single pass: prepare, decide, enforce |
| `code/schemas.py` | the output contract and the candidate-set re-check |
| `code/retrieval.py` | scoring a user's history down to the 8 candidates sent to the model |
| `code/evaluation/main.py` | leave-one-out, with four guards against label leakage |
| `tests/` | the invariants in the table above, executable |

## Public repository and licensing notice

This is an independent participant solution built from HackerRank's public
[`hackerrank-orchestrate-august26`](https://github.com/interviewstreet/hackerrank-orchestrate-august26)
starter repository. The challenge statement, starter files, dataset, and media
remain upstream materials; this repository does not claim ownership of them or
grant additional rights to them. See [`NOTICE.md`](./NOTICE.md) for the exact
attribution and scope.

The MIT license in [`LICENSE`](./LICENSE) applies only to the original solution
changes authored for this submission after upstream commit
`b2c4b904580c04cc3a66a82670b07838daf34a10`. It does not apply to the upstream
HackerRank materials or to third-party trademarks and media.

Generated submission archives such as `code.zip`, local caches, environment
files, handoff notes, interview notes, and duplicate workspaces are intentionally
excluded from the public source tree. The runnable source and historical
evaluation artifacts remain available directly in the repository.

## Verification labels

- **VERIFIED** means the claim was observed in a command run on 2026-08-01.
- **IMPLEMENTED — NOT EXERCISED** means the code path exists, but the final run
  did not activate it.
- **UNVERIFIED** means there is no direct test evidence for the claim.
- **INTERPRETATION** marks a conclusion that cannot be proved by a local test.

Claims without one of these labels are setup instructions, file descriptions,
or requirements quoted from the official repository.

## Setup

The verified environment used Python 3.11.14 and these pinned dependencies:

```text
openai==2.46.0
pydantic==2.13.4
Pillow==12.3.0
av==18.0.0
```

Create the environment with Python 3.11. Use whichever Python launcher exists
on the machine, then install the pinned packages:

```text
python3.11 -m venv .venv
# If python3.11 is not available under that name:
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Run only one of the two environment-creation commands. On Windows, the
equivalent launcher is `py -3.11`, and the environment's Python executable is
`.venv\Scripts\python.exe`. **NOT EXERCISED:** the Windows commands were not
tested during this submission.

Supply configuration through environment variables. For a POSIX shell:

```text
export GEMINI_API_KEY="your-key"
export GEMINI_MODEL="gemini-3.6-flash"
export GEMINI_MIN_CALL_INTERVAL_SECONDS="2"
```

Never put the key in this repository. The code reads it from the environment.

### Environment variables

| Variable | Required | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Gemini credential read from the process environment. |
| `GEMINI_MODEL` | Yes | Model used for media extraction and routing. The verified model is `gemini-3.6-flash`. |
| `GEMINI_MIN_CALL_INTERVAL_SECONDS` | No | Minimum interval between logical calls. Default: `2`. |
| `GEMINI_BASE_URL` | No | Endpoint override. The implemented default is Gemini's OpenAI-compatible endpoint. |

## One-command run

Portable POSIX-shell command, run from the repository root after exporting the
variables above:

```text
.venv/bin/python code/main.py
```

On Windows PowerShell, after setting the three variables with `$env:NAME=...`,
run `.venv\Scripts\python.exe code\main.py`. **NOT EXERCISED:** this Windows
command is documented for portability but was not tested during the
competition.

**VERIFIED MACOS COMMAND:** The exact command actually used for the final
110-row run was:

```text
zsh -ic 'GEMINI_MODEL=gemini-3.6-flash GEMINI_MIN_CALL_INTERVAL_SECONDS=2 .venv/bin/python code/main.py'
```

The `zsh -ic` wrapper starts a fresh interactive shell, so it reads
`GEMINI_API_KEY` from the macOS shell profile rather than using a stale
inherited value.

The default output is `submission/output.csv`. To choose another path:

```text
.venv/bin/python code/main.py --output results/output.csv
```

To rerun only cached fallback decisions while keeping validated model
decisions:

```text
.venv/bin/python code/main.py --refresh-fallbacks
```

To measure the local degradation path without making API calls, use the
explicit offline flag and a non-submission output path:

```text
.venv/bin/python code/main.py --force-offline --output code/evaluation/offline_baseline_110.csv
```

**VERIFIED:** `--force-offline` made zero logical API calls and zero HTTP
requests. It uses `code/evaluation/.cache/offline_baseline/`, not either the
submission cache or leave-one-out cache, and it refuses
`submission/output.csv`. Without this opt-in flag, normal behavior is
unchanged.

## Architecture

1. Local Python code reads the official CSV files and checks the decision
   cache.
2. Local code detects real media formats, prepares supported media, retrieves
   up to eight historical candidates, and computes explicit user, group,
   business, and quiet-hours signals.
3. Gemini describes or transcribes uncached media, then returns one structured
   routing decision per message.
4. Local code validates the decision, caches it, and writes the exact CSV
   contract.

**OBSERVED DURING THE FINAL RUN; RAW CONSOLE OUTPUT NOT RETAINED:** Prompt version
`routing-v4-confidence-evidence-guidance` invalidated all 110 previous decision
cache records. The final run made 110 new model decisions and used zero
fallback decisions.

**OBSERVED DURING THE FINAL RUN; RAW CONSOLE OUTPUT NOT RETAINED:** The final run made 19 media API calls: 11 image descriptions and
8 audio transcriptions. It recorded four media-cache hits for repeated media
references later in that run. The media cache that existed before the run did
not match the current media fingerprints, so no pre-run media result was
reused.

## Safety and resilience

- Incoming text, media-derived text, history, links, and file names are placed
  inside an explicit untrusted-data boundary. **IMPLEMENTED — NOT SEPARATELY
  ADVERSARIALLY TESTED.**
- The model cannot generate `message_id`; local code copies it from the input.
  **VERIFIED:** the final output ID set exactly matches all 110 input IDs.
- Action, message type, confidence, reason length, evidence count, and evidence
  membership are validated locally. **VERIFIED:** all 110 final rows passed the
  output checks.
- Evidence IDs are restricted to the runtime candidate set in the structured
  schema. **IMPLEMENTED — NOT SEPARATELY ADVERSARIALLY TESTED.**
- HTTP 429, transient 5xx, connection errors, and timeouts have up to six
  attempts with waits of 15, 30, 60, 90, and 120 seconds. A 429 also increases
  the pacing interval. **IMPLEMENTED — NOT EXERCISED:** the final run had zero
  transport retries and zero backoff seconds.
- One invalid structured response receives one correction attempt.
  **IMPLEMENTED — NOT EXERCISED:** the final run had zero validation retries.
- Exhausted routing or validation attempts produce the local fallback:
  `digest`, `unknown`, confidence `0.20`, and no evidence. **IMPLEMENTED — NOT
  EXERCISED:** the final run used zero fallbacks.
- A media failure does not stop routing. **VERIFIED:** `img_012` returned an
  empty image description, was marked unreadable, and all 110 routing rows were
  still produced.
- API credentials are not included in prompts, cache records, or output rows.
  **IMPLEMENTED — NOT FORMALLY SECRET-SCANNED.**

## Output format

`submission/output.csv` has this exact header:

```text
message_id,action,message_type,reason,confidence,evidence_message_ids
```

Use `none` when no historical evidence is cited. Otherwise, one or two IDs are
separated by semicolons.

**VERIFIED:** The current file has 110 rows, exactly one row for every input
ID, allowed enums only, two-decimal confidence values inside `0..1`, non-empty
reasons shorter than 200 characters, at most two unique evidence IDs per row,
and only IDs present in `dataset/message_history.csv`. The blank
`dataset/output.csv` template was not changed.

## Evaluation

Run the 30-row leave-one-out evaluation with:

```text
.venv/bin/python code/evaluation/main.py
```

For each sample, the evaluator removes that sample from the prompt and uses
only the other 29 examples. Labels from the held-out row are used only after
prediction as scoring references. Evaluation decisions use the separate cache
namespace `code/evaluation/.cache/leave_one_out/` and do not enter the
submission cache.

**VERIFIED FOR THE SUBMITTED V4 CONFIGURATION:** Two leave-one-out runs used
the same submitted prompt wording. The earlier run occurred before the cache
version label was advanced; the current run records the final
`routing-v4-confidence-evidence-guidance` label.

| Metric | Earlier identical-wording run | Current submitted-v4 artifact |
|---|---:|---:|
| Action accuracy | `27/30` | `27/30` |
| Message-type accuracy | `26/30` | `25/30` |
| Both correct | `24/30` | `23/30` |
| Evidence exact set match | `12/30` | `11/30` |
| Evidence partial overlap | `2/30` | `3/30` |
| Evidence no overlap | `16/30` | `16/30` |
| Confidence minimum | `0.82` | `0.84` |
| Confidence mean | `0.8557` | `0.8583` |
| Confidence median | `0.85` | `0.85` |
| Confidence maximum | `0.89` | `0.88` |

The model is not deterministic, so rerunning the pipeline will not reproduce
this CSV byte for byte. With unchanged prompt wording, the 30-sample results
varied by one row for message type, both-correct, evidence exact match, and
evidence partial overlap. The confidence minimum also changed from `0.82` to
`0.84`.

Evaluation artifacts are explicit:

- `code/evaluation/leave_one_out_v4_submitted.json`: current submitted-v4 run.
- `code/evaluation/leave_one_out_v4_prior_observation.json`: earlier run with
  identical wording before the cache-version label was advanced.
- `code/evaluation/leave_one_out_v5_diagnostic.json`: later v5 diagnostic; not
  the submitted configuration.
- `code/evaluation/leave_one_out_v3_original.json`: original v3 leave-one-out
  source used by the retrieval diagnostic.
- `code/evaluation/retrieval_diagnostic_v3_original.json`: original v3
  retrieval-only diagnostic.

`code/evaluation/fallback_model_smoke.py` and
`code/evaluation/retrieval_diagnostic.py` are historical result generators,
not clean-unpack evaluation entry points. The first reads comparison values
from the excluded production cache. The second reads media descriptions from
the excluded leave-one-out cache. Their named JSON result artifacts are
shipped; those two generators are not expected to rerun without their original
caches. The supported clean-unpack evaluation entry point is
`code/evaluation/main.py`.

### Forced-offline baseline

Run the zero-API baseline with:

```text
.venv/bin/python code/main.py --force-offline --output code/evaluation/offline_baseline_110.csv
.venv/bin/python code/evaluation/offline_baseline.py
```

**VERIFIED:** The 110-row run produced `digest=110`, `unknown=110`, confidence
`0.20` on all 110 rows, evidence `none` on all 110 rows, and 110 fallback
records. It made zero logical API calls and zero HTTP requests. The separate
offline cache contains 110 decisions and 19 media-unreadable records; neither
production cache namespace changed.

**VERIFIED:** Against the 30 solved examples, the deterministic fallback
scored action `11/30` (`36.67%`), message type `1/30` (`3.33%`), both correct
`1/30` (`3.33%`), and evidence exact match `2/30` (`6.67%`). These results are
stored in `code/evaluation/offline_fallback_baseline.json`.

| Metric | Offline fallback | Submitted-model run | Earlier identical-wording run | Model contribution |
|---|---:|---:|---:|---:|
| Action | `11/30` | `27/30` | `27/30` | `+53.33` percentage points in both runs |
| Message type | `1/30` | `25/30` | `26/30` | `+80.00` to `+83.33` percentage points |
| Both correct | `1/30` | `23/30` | `24/30` | `+73.33` to `+76.67` percentage points |
| Evidence exact | `2/30` | `11/30` | `12/30` | `+30.00` to `+33.33` percentage points |

### Manual Gemini 3.5 contingency check

The contingency was tested manually with exactly three calls by setting
`GEMINI_MODEL=gemini-3.5-flash`. The program still never changes models
automatically.

- **VERIFIED FOR ONE TEXT MESSAGE:** Structured routing succeeded for
  `msg_023`. Gemini 3.5 returned `notify/payment/0.86` with evidence
  `message_0243`; Gemini 3.6 had returned
  `notify/business_update/0.89` with the same evidence. The schema worked,
  but the decisions were not identical.
- **VERIFIED FOR ONE IMAGE:** Description succeeded for `img_002`. Both models
  extracted PVR INOX, *The Odyssey*, the `₹199` offer, and the relevant dates.
  Their wording was not identical.
- **VERIFIED FOR ONE VOICE NOTE:** Transcription succeeded for `vn_007`. Both
  models extracted the HDFC/Swiggy offer, percentage discounts, lifetime-free
  claim, immediate keypress request, and robocall or phishing concern. Their
  wording was not identical.

**LIMITED TEST EVIDENCE:** One successful call on each path proves that these
three tested requests worked with Gemini 3.5 and the current endpoint. It does
not establish general accuracy, reliability, or identical behavior across the
full dataset. Exact results and call counters are in
`code/evaluation/gemini_3_5_smoke.json`.

## Current submission verification

**VERIFIED:** The current `submission/output.csv` has:

- actions: `notify=36`, `digest=22`, `mute=52`
- message types: `personal=14`, `urgent=18`, `event=9`, `payment=2`,
  `business_update=12`, `promotion=16`, `greeting=4`, `forward=4`, `spam=1`,
  `scam=30`, `unknown=0`
- confidence: minimum `0.80`, maximum `0.92`, mean `0.8646`, median `0.87`
- evidence `none`: `14` rows
- fallback rows: `0`
- reason length: `74..114` characters

## Known limitations

- **UNVERIFIED / UNTUNED:** Retrieval weights are heuristic and untuned. Their
  individual contributions have not been optimized against a validation set.
- **VERIFIED ABSENCE; UNTESTED ACCURACY:** The `payment` type has no solved
  example, so payment classification accuracy is untested. The final CSV does
  contain two model predictions of `payment`.
- **VERIFIED ONLY ON THREE INPUTS:** `gemini-3.5-flash` is a manual
  contingency only. Structured routing worked on one text message, image
  description worked on one image, and audio transcription worked on one
  voice note. This limited smoke test does not establish full-dataset support,
  and the program never switches models automatically.
- **NOT VERIFIED FOR THE CURRENT OUTPUT:** An earlier review described roughly
  four percent of reasons as matching a provided example verbatim. Direct
  checking of the current regenerated CSV found `3/110`, or `2.73%`.
- **VERIFIED METRIC; INTERPRETATION:** Evidence exact-set match was `12/30` in
  the earlier identical-wording run and `11/30` in the current submitted-v4
  artifact. This submission treats those as conservative lower-bound
  observations because the official scoring description evaluates evidence
  relevance rather than requiring identical IDs; whether they are truly lower
  bounds cannot be verified without the hidden evaluator.
- **USER-VERIFIED ACROSS RUNS:** Media extraction is not fully deterministic.
  `img_012` succeeded in an earlier run, then returned an empty description in
  a later run. The later run marked it unreadable and routed its message using
  the remaining text, metadata, and context.
- **POST-COMPETITION FIX, OFFLINE REGRESSION-TESTED:** Provider failures use
  the closed vocabulary in `code/errors.py`, classified from SDK types and
  HTTP status codes, never exception text. Media cache reads/writes normalize
  legacy error fields; unreadable records carry no content. Transport logs
  and correction prompts no longer echo exception text or Pydantic input
  values. Tests inject fake secrets and local paths into failures and check
  results, persisted cache records, correction prompts, and transport output.
  These tests do not establish that arbitrary successful model content or all
  third-party library logging is free of sensitive data. No live provider
  requests were used to validate this fix; the frozen competition output is
  unchanged.
- **INTERPRETATION:** Solved examples are sent as compressed few-shot
  demonstrations containing their labels. No sample label is mapped to a
  target message, and no target label lookup or hardcoded target label exists.
  This submission interprets the organizers' “format and style” wording as
  permitting labelled demonstrations; that interpretation has not been
  confirmed by the organizers.

## Submission artifacts

The official submission requires:

1. The runnable code package, including this README and `requirements.txt`.
2. The generated predictions CSV at `submission/output.csv`.
3. The shared transcript at
   `$HOME/hackerrank_orchestrate_august26/log.txt`.

Nothing in this repository submits files automatically.

The competition-time `code.zip` archive is intentionally not tracked on the
public branch because it duplicates the source tree and bundled upstream data.
The immutable `submission-final` tag remains the reference for the original
competition submission state.
