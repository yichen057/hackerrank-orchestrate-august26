# Packaging manifest

This manifest describes a directory-based package. Build an empty staging
directory by copying only the files listed under “Must be included,” then
archive that staging directory. Do not archive the repository root and do not
use `git archive`, because source files may be untracked.

## Must be included in the code zip

### Root instructions and configuration

- `.gitignore`
- `AGENTS.md`
- `CLAUDE.md`
- `README.md`
- `problem_statement.md`
- `requirements.txt`

### Architecture and packaging documentation

- `docs/architecture.md`
- `docs/packaging_manifest.md`

### Runnable solution

- `code/cache.py`
- `code/config.py`
- `code/context.py`
- `code/data_io.py`
- `code/fallback.py`
- `code/main.py`
- `code/media.py`
- `code/pipeline.py`
- `code/prompts.py`
- `code/provider.py`
- `code/retrieval.py`
- `code/schemas.py`

### Evaluation code and named evidence artifacts

- `code/evaluation/main.py`
- `code/evaluation/offline_baseline.py`
- `code/evaluation/fallback_model_smoke.py` — historical generator; requires
  excluded production-cache comparison records and is not expected to rerun
  from a clean unpack.
- `code/evaluation/retrieval_diagnostic.py` — historical generator; requires
  excluded leave-one-out media-cache records and is not expected to rerun from
  a clean unpack.
- `code/evaluation/offline_baseline_110.csv`
- `code/evaluation/offline_fallback_baseline.json`
- `code/evaluation/gemini_3_5_smoke.json`
- `code/evaluation/leave_one_out_v4_submitted.json`
- `code/evaluation/leave_one_out_v4_prior_observation.json`
- `code/evaluation/leave_one_out_v5_diagnostic.json`
- `code/evaluation/leave_one_out_v3_original.json`
- `code/evaluation/retrieval_diagnostic_v3_original.json`

The v5 file is clearly marked as a diagnostic that is not the submitted
configuration. The current submitted-v4 evidence is
`leave_one_out_v4_submitted.json`.

The supported clean-unpack evaluation command uses `code/evaluation/main.py`.
The two historical generators above are included for auditability, while their
named JSON result artifacts preserve the observed results without shipping
their caches.

### Required dataset CSV files

- `dataset/business_accounts.csv`
- `dataset/daily_notification_summary.csv`
- `dataset/group_members.csv`
- `dataset/groups.csv`
- `dataset/images.csv`
- `dataset/message_events.csv`
- `dataset/message_history.csv`
- `dataset/messages.csv`
- `dataset/output.csv`
- `dataset/sample_messages.csv`
- `dataset/user_business_history.csv`
- `dataset/users.csv`
- `dataset/voice_notes.csv`

### Required audio files

- `dataset/media/audio/vn_001.mp3`
- `dataset/media/audio/vn_002.mp3`
- `dataset/media/audio/vn_003.mp3`
- `dataset/media/audio/vn_004.mp3`
- `dataset/media/audio/vn_005.mp3`
- `dataset/media/audio/vn_006.mp3`
- `dataset/media/audio/vn_007.mp3`
- `dataset/media/audio/vn_008.mp3`
- `dataset/media/audio/vn_009.mp3`
- `dataset/media/audio/vn_012.mp3`
- `dataset/media/audio/vn_013.mp3`
- `dataset/media/audio/vn_014.mp3`
- `dataset/media/audio/vn_015.mp3`

### Required image files

- `dataset/media/images/img_001.jpg`
- `dataset/media/images/img_002.jpg`
- `dataset/media/images/img_003.jpg`
- `dataset/media/images/img_004.jpg`
- `dataset/media/images/img_005.jpg`
- `dataset/media/images/img_006.jpg`
- `dataset/media/images/img_007.jpg`
- `dataset/media/images/img_008.jpg`
- `dataset/media/images/img_010.jpg`
- `dataset/media/images/img_011.jpg`
- `dataset/media/images/img_012.jpg`
- `dataset/media/images/img_013.jpg`
- `dataset/media/images/img_014.jpg`
- `dataset/media/images/img_016.jpg`
- `dataset/media/images/img_020.jpg`
- `dataset/media/images/img_022.jpg`
- `dataset/media/images/img_023.jpg`
- `dataset/media/images/img_024.jpg`
- `dataset/media/images/img_025.jpg`
- `dataset/media/images/img_026.jpg`

## Separate submission artifacts — not part of the code zip

- `submission/output.csv` — upload as the predictions CSV.
- `$HOME/hackerrank_orchestrate_august26/log.txt` — upload as the chat
  transcript.

## Must be excluded from the code zip

- `.git/`
- `.venv/`
- `.env`
- `.env.*`
- `.DS_Store` at any depth
- `__pycache__/` at any depth
- `*.pyc`, `*.pyo`, and `*.pyd` at any depth
- `code/.cache/`
- `code/evaluation/.cache/`

These paths contain local environments, secrets, operating-system metadata,
compiled bytecode, or generated caches. They are not required to install or
run the solution from scratch.
