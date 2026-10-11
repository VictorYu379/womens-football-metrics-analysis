# Who Gets the Credit?

**Cross-Gender Model Transfer and Player Valuation in Women's Soccer**

**English** | [简体中文](README.zh-CN.md)

Research code accompanying the [paper abstract](Who_Gets_the_Credit_Abstract_LaTeX_Style.pdf). The study asks whether models trained on men's soccer assign credit to women players in the same way as models trained on women's data. It moves beyond event-level predictive accuracy to examine **player-ranking stability**, with particular attention to defenders, hybrid-role players, and non-shooting actions.

## 1. Paper overview and result provenance

### Research question and study design

Player valuation models inform scouting and talent identification. Similar predictive performance does not necessarily imply similar judgments about which players contribute most. The paper compares five valuation frameworks: **xG, xA, xT, VAEP, and Atomic-VAEP**, examining both overall rankings and differences by position and action type.

The supplied abstract describes the following design:

- **Data:** StatsBomb Open Data, covering **771 women's matches from 2023/24** and **1,517 men's matches from 2015/16**. These counts are reported by the abstract, not recomputed from a new download.
- **Evaluation:** within each metric, separately trained men's and women's models score the same held-out women's events, with match-level train/test splits. Player-level rankings are then compared.
- **Sample-size control:** men's training events are downsampled for xG, xA, VAEP, and Atomic-VAEP to match the women's event count. xT does not use the same row-level downsampling step. Event populations and sample sizes differ across metrics.
- **Action-level analysis:** interceptions, ball receptions, and progressive passes receive particular attention because non-shooting contributions can be important for defenders and players with hybrid roles.

### Main findings

The abstract emphasizes high ranking agreement for most metrics and a much larger disagreement under Atomic-VAEP. The following abstract-reported correlations match the saved overall Spearman outputs in the selected experiment baseline, `824d701`, to the displayed precision. No experiments were rerun to compile this summary.

| Framework | Player-ranking Spearman correlation |
| --- | --- |
| xG | 0.977 |
| xA | 0.971 |
| xT | 0.975 |
| VAEP | 0.954 |
| Atomic-VAEP | **0.652** |

Additional findings **reported by the abstract** are:

- Atomic-VAEP ranking correlations are **0.365 for centre-backs**, **0.413 for fullbacks/wingbacks**, and **0.877 for forwards**, suggesting stronger instability for defensive roles in that experiment snapshot.
- Interceptions and receptions account for much of the defender instability, while progressive passes matter less than expected. The interpretation focuses on defensive transitions, not attacking progression alone.
- Women-specific training does not always improve predictive performance; the abstract describes nearly identical xG Brier scores for men's-trained and women's-trained models. It reports statistically significant correlations at `p < 0.001`, while emphasizing the size and location of ranking shifts rather than significance alone.

These findings motivate caution when using transferred valuation models for recruitment, especially for defenders and players whose contributions depend on transitional actions. They should not be read as estimates of an isolated gender effect.

### Experiment baseline

**The experiment baseline for these bilingual materials is commit `824d701`.** At this baseline, Atomic-VAEP prepares its features and fits its models directly in the notebook. Both training populations use `HistGradientBoostingClassifier` with `max_iter=300`, `learning_rate=0.05`, and `random_state=42`; no separate Atomic-VAEP tuning JSON is required.

The root Atomic-VAEP notebook and its English copy contain the saved overall correlation `0.652` and position-level values matching the abstract after rounding. No local archive is needed to inspect these results. Saved outputs support inspection of this experiment snapshot; independently reproducing them still requires running the documented workflow in a recorded environment.

## 2. Guide for reviewers

**To inspect the results, start with the English notebooks below. No training or data download is needed to read their saved outputs.** The original notebooks are retained, and each `.en.ipynb` copy preserves its code, execution counts, metadata, and saved outputs. Only Markdown explanations are translated, with an explicit xG clarification noted below. Saved outputs document an existing run; opening them is not a new reproducibility run.

| Workflow | English notebook | Results to inspect |
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

For Chinese explanations and links to the original main notebooks, use [README.zh-CN.md](README.zh-CN.md).

## 3. Repository layout

```text
README.md / README.zh-CN.md        English / Chinese reviewer guides
Who_Gets_the_Credit_Abstract_LaTeX_Style.pdf   Supplied abstract and figures
*.ipynb / *.en.ipynb               Original notebooks / English copies
xa/                               xA data preparation, models, and tuning
xg/                               xG data preparation, models, and tuning
vaep/                             VAEP data preparation and probability models
requirement.txt                   Runtime dependencies
```

The notebooks remain in the project root so their relative paths continue to work. Both languages share the same Python modules, tuning artifacts, and data caches; there is no separate English implementation. Atomic-VAEP preparation and model fitting are implemented directly in its notebook at this baseline.

## 4. Set up an environment for reproduction

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

## 5. Data and tuning prerequisites

### Data and caches

- Competition/season lists and split rules are defined in the notebooks and their data modules. Inspect these definitions when checking the study population.
- When required caches are missing, the xA, xG, xT, and standard VAEP workflows obtain StatsBomb open data and rebuild intermediate tables. Allow network access for a cold-cache run. Atomic-VAEP itself does not download raw events.
- `data/`, `open-data/`, and `vaep_data/` are ignored by Git. Do not assume a fresh checkout contains the original machine's data or prepared caches.
- `vaep_data/` stores reusable intermediate data. For a cold-cache run, run standard VAEP before Atomic-VAEP to generate `women_spadl_ltr.parquet`, `men2015_spadl_ltr.parquet`, and the required `women_pos.parquet` under this directory.
- Atomic-VAEP reads `women_atomic_spadl_ltr.parquet` and `men2015_atomic_spadl_ltr.parquet` when present, or converts the existing SPADL caches into these Atomic-SPADL caches. If neither action-cache representation is available, or the women's position cache is missing, it raises `FileNotFoundError` rather than rebuilding from raw StatsBomb events.

### Required tuning JSON files

The xA, xG, and standard VAEP notebooks read selected hyperparameters and **fit fresh models inside the notebook**. They do not load the fitted `.joblib` models produced during tuning. xT learns its threat maps directly. Atomic-VAEP fits its probability models using the fixed settings defined in its own notebook. Neither xT nor Atomic-VAEP requires a tuning JSON at this baseline.

| Workflow | Exact path read by the notebook | Availability in this code snapshot |
| --- | --- | --- |
| xA | `xa/xa_all_models/xa_original_training_results.json` | Not tracked at this path; generate it before rerunning xA |
| xG | `xg/xgboost_tuning_results/xgboost_xg_tuning_results.json` | Included |
| VAEP | `vaep/hist_gradient_boosting_tuning_results/vaep_training_results.json` | Included |

**Skip tuning when the required JSON already exists.** If a file is missing, run the corresponding command:

```shell
python -u xa/train_original_xa_models.py --model-family all --output-dir xa/xa_all_models
```

```shell
python -u xg/train_xgboost_xg_models.py
```

```shell
python -u vaep/train_vaep_probability_models.py --search-n-jobs 1
```

For xA, `--model-family xgboost` is sufficient for the notebook; `all` also runs the logistic-regression comparison. An xA result saved under another directory is not automatically read by the notebook.

Preserve existing artifacts before intentionally regenerating them. The xA and VAEP runners require `--overwrite` to replace existing artifacts; the xG runner should likewise only be rerun when replacing its results is intended. Do not use `--quick` smoke-test results as replacements for full-search results in a formal reproduction.

Module details: [xA](xa/README.md), [xG](xg/README.md), and [VAEP](vaep/README.md). Atomic-VAEP's cache handling, feature construction, and model settings are documented in its notebook.

## 6. Execute the experiments

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

The Chinese originals execute the same code, so running both languages is unnecessary. Expected notebook outputs include split/training summaries, player-level statistics, and the tables and plots listed in the reviewer guide. Tuning scripts separately write their search artifacts; generated data and caches are not part of the language copies.

Cold-cache preparation and full hyperparameter searches can be expensive. This repository does not provide a measured end-to-end runtime or minimum-RAM guarantee. The VAEP tuning command above uses one search worker to limit memory duplication; this does not establish a fixed RAM requirement. Atomic-VAEP fits its configured classifiers directly in the notebook.

## 7. Methodological notes for review

- **Within-workflow comparisons:** models are evaluated on the same women's test events within each main notebook. Different metrics use different event populations and player filters; the five notebooks should not be assumed to have identical player cohorts or test-match sets.
- **Splits:** the xA, xG, and VAEP data modules reserve women's test matches before tuning/training. Their configured seed is `42`; xA uses a `0.20` women's test share, while xG and VAEP use `0.30`. xT and Atomic-VAEP define their splits in their own notebooks; Atomic-VAEP also uses seed `42` and a `0.30` test share. Consult the actual split summaries when checking comparability.
- **Training-data size:** xA, xG, VAEP, and Atomic-VAEP include a downsampled-men vs. women comparison. xG additionally fits a full-men model. The xT notebook trains on its men's action pool without the same row-level size-matching step.
- **Hyperparameters:** xA, xG, and standard VAEP reuse men's-selected hyperparameters for the women's fresh models as well; VAEP selects parameters separately for `scores` and `concedes`. Atomic-VAEP instead uses the same fixed HGB settings for both targets and both training populations (`max_iter=300`, `learning_rate=0.05`, `random_state=42`). In xG, the original Chinese Section 5 still mentions women-specific parameters, but the code applies `BEST_MEN_XGBOOST_PARAMS` to all three models. The English notebook explicitly clarifies this; no experiment code or saved result is changed by translation.
- **Sequence integrity:** standard VAEP prepares features/labels from complete match sequences before downsampling. Atomic-VAEP computes features/labels on each selected game's complete sequence, then retains sampled rows during per-game preparation. Both use a feature-window size of 3 actions and a label horizon of 10, in their respective standard or atomic action representations.
- **Interpretation:** Spearman correlation measures agreement between model-derived player rankings, not ground-truth player ability, and is distinct from event-level predictive performance. Data come from one provider, and the men's and women's samples differ in season and competition coverage, so observed transfer differences should not be interpreted as isolating gender alone. Position-only xT agreement does not establish the absence of other tactical or contextual differences. The abstract identifies synchronized men's and women's data from the same season as an important next step.

For a new reproduction, retain the repository revision, environment details, data/cache provenance, tuning JSON files, and executed notebooks. Viewing saved outputs and independently reproducing them are distinct review activities.
