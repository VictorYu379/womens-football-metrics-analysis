# Who Gets the Credit?

**Cross-Gender Model Transfer and Player Valuation in Women's Soccer**

Research code accompanying the [paper abstract](Who_Gets_the_Credit_Abstract_LaTeX_Style.pdf). The study asks whether models trained on men's soccer assign credit to women players in the same way as models trained on women's data. It moves beyond event-level predictive accuracy to examine **player-ranking stability**, with particular attention to defenders, hybrid-role players, and non-shooting actions.

## 1. Overview

**Background.** Player valuation models inform scouting and recruitment, but similar event-level predictive accuracy does not guarantee consistent player rankings across training populations.

**Research question.** Do models trained on men's and women's soccer data assign similar values and rankings to the same women players? We compare xG, xA, xT, VAEP, and Atomic-VAEP.

**Main findings.** The abstract reports strong ranking agreement for xG, xA, xT, and VAEP (Spearman 0.954–0.977), but lower agreement for Atomic-VAEP (0.652), especially among defenders. Interceptions and receptions account for much of this instability, suggesting caution when transferring models to evaluate defensive and transitional contributions.

## 2. Guide for reviewers

**To inspect the results, open the notebooks below. No training or data download is needed to read their saved outputs.** Independently reproducing the results requires running the documented workflow.

| Workflow | Notebook | Results to inspect |
| --- | --- | --- |
| xA | [xA experiment](enriched_xa_features_competition_events.en.ipynb) | Sections 6–12: player rankings, Spearman correlation, pass profiles, endpoint grids, and position buckets |
| xG | [xG experiment](enriched_xg_features_competition_events.en.ipynb) | Sections 5–7 and 9–12: player rankings, StatsBomb xG reference, full/downsampled comparison, spatial and positional diagnostics |
| xT | [xT experiment](enriched_xt_features_competition_events.en.ipynb) | Sections 11–15: learned threat maps, player rankings, Spearman correlation, and visual comparisons |
| VAEP | [VAEP experiment](enriched_vaep_features_competition_events.en.ipynb) | Sections 7–11: action values, player rankings, position buckets, and action-type distributions |
| Atomic-VAEP | [Atomic-VAEP experiment](enriched_atomic_vaep_features_competition_events.en.ipynb) | Sections 7–12: atomic action values, player rankings, position/action-type distributions, and event spatial maps |

The two figures in the supplied abstract correspond to the following Atomic-VAEP analysis sections in this experiment baseline:

| Abstract figure | Corresponding notebook analysis |
| --- | --- |
| Figure 1: action-type differences in Atomic-VAEP values | [Atomic-VAEP](enriched_atomic_vaep_features_competition_events.en.ipynb), Section 11: action-type delta distributions |
| Figure 2: position-specific Atomic-VAEP ranking shifts | [Atomic-VAEP](enriched_atomic_vaep_features_competition_events.en.ipynb), Sections 10 and 10.1: position summaries and within-position rank movement |

## 3. Repository layout

```text
README.md                         Project overview and reviewer guide
Who_Gets_the_Credit_Abstract_LaTeX_Style.pdf   Supplied abstract and figures
*.en.ipynb                        Experiment notebooks
xa/                               xA data preparation, models, and tuning
xg/                               xG data preparation, models, and tuning
vaep/                             VAEP data preparation and probability models
requirement.txt                   Runtime dependencies
```

The notebooks remain in the project root so their relative paths continue to work. Atomic-VAEP preparation and model fitting are implemented directly in its notebook.

## 4. Set up an environment for reproduction

**Experiment baseline:** `824d701`.

Run all commands from the **project root**. A working Python 3 installation is required; the local `.venv` is not distributed with the repository.

Create a virtual environment:

```shell
python -m venv .venv
```

Install dependencies with that environment's interpreter.

**Windows PowerShell:**

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirement.txt
```

**macOS / Linux:**

```shell
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirement.txt
```

In the remaining commands, `python` means the virtual-environment interpreter. Activate the environment, or replace `python` with `.\.venv\Scripts\python.exe` on Windows or `.venv/bin/python` on macOS/Linux. In JupyterLab, select the corresponding environment as the notebook kernel.

`requirement.txt` specifies lower bounds, not a fully pinned environment. For a reproduction run, record the Python version and installed package versions rather than assuming every dependency combination is identical to the environment that produced the saved outputs.

## 5. Data source and preparation

We use StatsBomb Open Data: 771 women's matches from 2023/24 and 1,517 men's matches from 2015/16, as described in the abstract.

Before modeling:

1. **Load events:** `xa/xa_data.py` and `xg/xg_data.py` load completed-pass and shot data, using local caches or fetching missing data from StatsBomb.
2. **Build action representations:** the xT notebook and `vaep/vaep_data.py` prepare left-to-right SPADL actions. Atomic-VAEP converts cached SPADL actions into Atomic-SPADL.
3. **Prepare inputs:** construct metric-specific features and labels, reserve women's test matches at game level, and downsample men's training data where applicable.

Data and intermediate tables are cached in `data/` and `vaep_data/`. On a cold-cache run, run standard VAEP before Atomic-VAEP to create the SPADL and player-position caches it requires.

## 6. Execute the experiments

Before running xA, follow the [xA training guide](xa/README.md) and use `--output-dir xa/xa_all_models` if its tuning results are missing. xG and VAEP tuning results are included; see the [xG](xg/README.md) and [VAEP](vaep/README.md) guides for details.

Start JupyterLab from the project root:

```shell
python -m jupyter lab
```

Use a **fresh kernel for each notebook**, and run its cells from top to bottom. This also avoids mixing same-named helper modules from different model directories in a shared kernel.

Recommended order:

1. xA: `enriched_xa_features_competition_events.en.ipynb`
2. xG: `enriched_xg_features_competition_events.en.ipynb`
3. xT: `enriched_xt_features_competition_events.en.ipynb`
4. VAEP: `enriched_vaep_features_competition_events.en.ipynb`
5. Atomic-VAEP: `enriched_atomic_vaep_features_competition_events.en.ipynb`

Expected outputs include split and training summaries, player-level statistics, and the tables and plots listed in the reviewer guide. Tuning scripts save their artifacts separately.

Cold-cache preparation and full hyperparameter searches can be expensive; runtime and memory requirements depend on the environment.

## 7. Methodological notes for review

- **Within-workflow comparisons:** models are evaluated on the same women's test events within each main notebook. Different metrics use different event populations and player filters; the five notebooks should not be assumed to have identical player cohorts or test-match sets.
- **Splits:** the xA, xG, and VAEP data modules reserve women's test matches before tuning/training. Their configured seed is `42`; xA uses a `0.20` women's test share, while xG and VAEP use `0.30`. xT and Atomic-VAEP define their splits in their own notebooks; Atomic-VAEP also uses seed `42` and a `0.30` test share. Consult the actual split summaries when checking comparability.
- **Training-data size:** xA, xG, VAEP, and Atomic-VAEP include a downsampled-men vs. women comparison. xG additionally fits a full-men model. The xT notebook trains on its men's action pool without the same row-level size-matching step.
- **Hyperparameters:** Within each experiment, models trained on men's and women's data use the same hyperparameter configuration. Refer to the experiment code for the specific settings.
- **Sequence integrity:** standard VAEP prepares features/labels from complete match sequences before downsampling. Atomic-VAEP computes features/labels on each selected game's complete sequence, then retains sampled rows during per-game preparation. Both use a feature-window size of 3 actions and a label horizon of 10, in their respective standard or atomic action representations.

For a new reproduction, retain the repository revision, environment details, data/cache provenance, tuning JSON files, and executed notebooks. Viewing saved outputs and independently reproducing them are distinct review activities.

## 8. Interpretation and limitations

- Spearman correlation measures agreement between model-derived player rankings, not ground-truth player ability, and is distinct from event-level predictive performance.
- Data come from one provider, and the men's and women's samples differ in season and competition coverage, so observed transfer differences should not be interpreted as isolating gender alone.
- Position-only xT agreement does not establish the absence of other tactical or contextual differences.
- The abstract identifies synchronized men's and women's data from the same season as an important next step.
