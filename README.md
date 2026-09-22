# StatsBomb Experiments

这个 repo 包含 xA、xG、xT、VAEP / Atomic VAEP 的 StatsBomb 数据实验。主要 notebook：

- `enriched_xa_features_competition_events.ipynb`
- `enriched_xg_features_competition_events.ipynb`
- `enriched_xt_features_competition_events.ipynb`
- `enriched_vaep_features_competition_events.ipynb`
- `enriched_atomic_vaep_features_competition_events.ipynb`

## 1. 环境准备

在项目根目录运行：

1. 创建虚拟环境

```
python -m venv .venv
```

2. 安装 dependencies

```
python -m pip install --upgrade pip
python -m pip install -r requirement.txt
```

3. 启动 JupyterLab

```
python -m jupyter lab
```

在 JupyterLab 里选择这个项目的 Python 环境作为 notebook kernel。

## 2. 先跑 xA / xG 调参

xA 和 xG notebook 会读取 grid search 选出的最佳 hyperparameters，然后在 notebook 里重新训练 fresh models。因此，第一次跑对应 notebook 前，需要先生成这些结果文件：

- xA notebook 需要 `xa/xa_all_models/xa_original_training_results.json`
- xG notebook 需要 `xg/xgboost_tuning_results/xgboost_xg_tuning_results.json`

生成 xA hyperparameters：

```
python -u xa/train_original_xa_models.py --model-family all --output-dir xa/xa_all_models --overwrite
```

生成 xG hyperparameters：

```
python -u xg/train_xgboost_xg_models.py
```

如果这些文件已经存在，可以直接跳过这一步。

## 3. 运行 notebooks

建议顺序：

1. `enriched_xa_features_competition_events.ipynb`
2. `enriched_xg_features_competition_events.ipynb`
3. `enriched_xt_features_competition_events.ipynb`
4. `enriched_vaep_features_competition_events.ipynb`
5. `enriched_atomic_vaep_features_competition_events.ipynb`

说明：

- xA / xG notebook 不会直接加载训练好的模型，只复用调参得到的 hyperparameters。
- xA / xG / xT / VAEP 会在需要时远程读取 StatsBomb 数据，并把中间结果写入 `vaep_data/` cache。
- Atomic VAEP 依赖标准 VAEP 生成的 SPADL cache；从零开始时先跑标准 VAEP，再跑 Atomic VAEP。