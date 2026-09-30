# Former standalone repository audit

Audit date: 2026-09-30

The maintained Python package sources and curated example notebooks are now in
this monorepo. The former standalone working trees have not been archived;
archival remains an explicit follow-up action after review of this record.

## FluoPulse

Audited working tree: the former standalone `fluopulse-analysis` checkout.

Findings:

- The working tree is on `main` and is dirty.
- Nine tracked notebooks (`01`, `02`, `03`, and `06` through `11`) have
  substantial uncommitted changes.
- `Untitled.ipynb` is untracked.
- `.virtual_documents/` is untracked and appears to be generated editor state.
- The shared-event/session/synchronization source migration is represented in
  the monorepo package, while the monorepo additionally contains the maintained
  FluoPulse GLM implementation.
- All 11 named example notebooks were copied into
  `packages/fluopulse-analysis/examples/notebooks` with outputs and execution
  counts removed.
- The one-cell `Untitled.ipynb` only installs Pynapple. It was intentionally not
  promoted into the maintained examples; the original remains untouched.

Validation performed:

- every migrated notebook is valid nbformat 4 JSON;
- all code cells parse as Python;
- outputs and execution counts are cleared;
- no source cell contains a personal `C:\Users\...` path or standalone-output
  checkout path;
- package-level imports used by the examples resolve against the monorepo
  FluoPulse API.

The notebooks were not executed end-to-end because they require local raw
recordings and user-edited session configuration. Package tests cover the
called reusable APIs; end-to-end execution remains a data-dependent release
check.

## iFLiP3

Audited working tree: the former standalone `iflip3-analysis` checkout.

Findings:

- The working tree is on `main` and is dirty from the shared-core migration.
- The migrated package source matches the monorepo for the checked
  acquisition/session modules.
- The two named example notebooks were copied into
  `packages/iflip3-analysis/examples/notebooks` and cleared of execution state.
- The substantive 39-cell `Untitled.ipynb` was preserved as
  `03_decay_component_exploration_draft.ipynb`, given an explanatory title,
  and changed to use an editable generic data path.

All three notebooks pass the same source-only validation, and their package
imports resolve. The third remains explicitly labeled as an exploratory draft,
not a supported batch workflow. End-to-end execution requires representative
iFLiP3 source data and remains a data-dependent release check.

No standalone files were deleted, moved, or overwritten during this audit or
migration.
