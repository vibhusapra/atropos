<!--
╭───────────────────────────────────────────────────────────╮
│  ✨  RL-ENVIRONMENTS PULL REQUEST TEMPLATE  ✨             │
│  Fill out each field → delete guidance placeholders.      │
│  Incomplete items slow down review & scoring.             │
╰───────────────────────────────────────────────────────────╯
-->

## 🔖 Environment Snapshot
| Field | Your Entry |
|-------|------------|
| **Environment Name** | <!-- e.g. "SudokuVerifier-v0" --> |
| **Short Description** | <!-- One-sentence purpose/goal. --> |
| **Category** | <!-- Select: Verifiable-Reasoning / RLAIF / RLHF / Other  --> |
| **Dataset Needed?** | <!-- No / Yes (link & license) --> |
| **External Deps** | <!-- Extra pip packages, system libs, etc. --> |
| **Environmental Variables** | <!-- variable name(s) --> |
| **Expected Episode Length** | <!-- e.g. 128 timesteps --> |
| **Compute Footprint Estimate** | <!-- "<1 GB RAM, <1 min CPU verification" or similar --> |

---

## 🧪 Zero-Training Test Results
<details>

**W&B Link:**

**Examples of the Environment scoring a good example and a bad example:**

</details>


## ✅ Developer & Reviewer Checklist
- [ ] Code follows project style (black, isort, flake8 pass with pre-commit).
- [ ] I have performed a self-review of my own code
- [ ] Docstrings added for all new public classes / functions.
- [ ] If .env vars required, did you add it to the .env.example in repo root?
- [ ] Automatic rollout script (`scripts/run_smoke_test.py`) runs without training and reproduces the metrics above.
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation to include my environment
- [ ] My changes generate no new warnings
- [ ] New and existing unit tests pass locally with my changes

---
