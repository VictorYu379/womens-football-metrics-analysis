# Who Gets the Credit?

**跨性别模型迁移与女足球员价值评估**

[English](README.md) | **简体中文**

本项目是[论文摘要](Who_Gets_the_Credit_Abstract_LaTeX_Style.pdf)的配套研究代码。研究关注：在男足数据上训练的模型，是否会像女足数据训练的模型一样认可女足球员的贡献？论文不只比较事件层面的预测表现，还考察**球员排名的稳定性**，尤其关注后卫、承担混合角色的球员，以及非射门动作的价值。

## 1. 论文概述与结果来源

### 研究问题与实验设计

球员价值评估模型可用于球探工作和人才识别。但预测表现相近，不代表模型对“哪些球员贡献更大”给出相同判断。论文比较 **xG、xA、xT、VAEP 和 Atomic-VAEP** 五种框架，同时分析整体排名、位置差异和动作类型差异。

附带的摘要描述了以下研究设计：

- **数据：** StatsBomb Open Data，包含 **2023/24 赛季的 771 场女足比赛**和 **2015/16 赛季的 1,517 场男足比赛**。这些比赛数来自摘要，不是本次重新下载后统计的结果。
- **评估方式：** 在每种指标内部，分别训练男足和女足模型，并对同一批留出的女足事件打分；训练/测试按比赛划分，再比较球员层面的排名。
- **样本量控制：** xG、xA、VAEP 和 Atomic-VAEP 对男足训练事件下采样，以匹配女足事件数；xT 不采用同样的行级下采样步骤。不同指标对应的事件集合和样本量并不相同。
- **动作层面分析：** 重点关注拦截、接球和推进传球，因为这些非射门贡献可能对后卫和承担混合角色的球员尤为重要。

### 主要发现

摘要强调：大多数指标的球员排名一致性较高，而 Atomic-VAEP 的差异明显更大。下表中的摘要数值与选定实验基线 `824d701` 中已保存的整体 Spearman 输出，在展示精度下一致。整理本概述时没有重新运行实验。

| 框架 | 球员排名的 Spearman 相关系数 |
| --- | --- |
| xG | 0.977 |
| xA | 0.971 |
| xT | 0.975 |
| VAEP | 0.954 |
| Atomic-VAEP | **0.652** |

**摘要还报告了**以下发现：

- Atomic-VAEP 在**中后卫、边后卫/翼卫和前锋**组内的排名相关系数分别为 **0.365、0.413 和 0.877**，说明在该实验快照中，防守角色的排名不稳定性更明显。
- 拦截和接球解释了相当一部分后卫排名的不稳定性，推进传球的作用则弱于预期；解释重点是防守转换，而不只是进攻推进。
- 专门使用女足数据训练并不总能改善预测表现；摘要指出，男足训练与女足训练的 xG 模型具有非常接近的 Brier 分数。摘要报告相关系数在 `p < 0.001` 水平显著，但强调的重点是排名变化的幅度和分布，而不只是显著性。

这些发现提示，在招募决策中迁移使用球员估值模型时，需要特别谨慎地对待后卫和依赖转换动作贡献的球员；不能将其解释为对性别单一因素影响的估计。

### 实验基线

**这套双语材料对应的实验基线为提交 `824d701`。** 在这一版本中，Atomic-VAEP 直接在 notebook 内准备特征并拟合模型。男女足训练组都使用 `HistGradientBoostingClassifier`，配置为 `max_iter=300`、`learning_rate=0.05`、`random_state=42`，不需要独立的 Atomic-VAEP 调参 JSON。

根目录中的 Atomic-VAEP notebook 及其英文副本均保存了整体相关系数 `0.652`，位置分组数值在四舍五入后也与摘要一致。查看这些结果不需要依赖本地 archive。已保存的输出可用于审阅该实验快照；独立复现仍需在记录完整的环境中运行下文流程。

## 2. 审稿人阅读指南

**如果只希望查看结果，可以直接打开下表中的 notebook，无需重新训练或下载数据。** 原始 notebook 完整保留；对应的 `.en.ipynb` 英文副本保留相同的代码、执行计数、元数据和已有输出，仅翻译 Markdown 说明，并对下文提到的 xG 原文歧义明确加注。已有输出来自此前的运行，打开文件阅读并不等于完成了一次新的复现实验。

| 实验 | 原始中文 notebook | 建议查看的结果 |
| --- | --- | --- |
| xA | [xA 实验](enriched_xa_features_competition_events.ipynb) | 第 6–12 节：球员排名、Spearman 相关性、传球特征、传球终点网格和位置分组 |
| xG | [xG 实验](enriched_xg_features_competition_events.ipynb) | 第 5–7、9–12 节：球员排名、StatsBomb xG 参考、男足全量/下采样比较、空间和位置诊断 |
| xT | [xT 实验](enriched_xt_features_competition_events.ipynb) | 第 11–15 节：学习得到的威胁地图、球员排名、Spearman 相关性及可视化 |
| VAEP | [VAEP 实验](enriched_vaep_features_competition_events.ipynb) | 第 7–11 节：动作价值、球员排名、位置分组和动作类型分布 |
| Atomic-VAEP | [Atomic-VAEP 实验](enriched_atomic_vaep_features_competition_events.ipynb) | 第 7–12 节：原子动作价值、球员排名、位置/动作类型分布和事件空间图 |

摘要中的两张图对应这一实验基线中的以下 Atomic-VAEP 分析章节：

| 摘要图像 | 对应的 notebook 分析 |
| --- | --- |
| 图 1：不同动作类型的 Atomic-VAEP 估值差异 | [Atomic-VAEP](enriched_atomic_vaep_features_competition_events.ipynb) 第 11 节：动作类型的 delta 分布 |
| 图 2：不同位置的 Atomic-VAEP 排名变化 | [Atomic-VAEP](enriched_atomic_vaep_features_competition_events.ipynb) 第 10、10.1 节：位置汇总及组内排名变化 |

英文说明及五个英文 notebook 的入口见 [README.md](README.md)。

## 3. 目录结构

```text
README.md / README.zh-CN.md        英文 / 中文审稿指南
Who_Gets_the_Credit_Abstract_LaTeX_Style.pdf   附带的论文摘要及图像
*.ipynb / *.en.ipynb               原始 notebook / 英文副本
xa/                               xA 数据准备、模型和调参
xg/                               xG 数据准备、模型和调参
vaep/                             VAEP 数据准备和概率模型
requirement.txt                   运行依赖
```

Notebook 继续放在项目根目录，以保持现有相对路径有效。两种语言共用同一套 Python 模块、调参结果和数据缓存，不维护独立的英文实验实现。这一实验基线中的 Atomic-VAEP 数据准备和模型拟合直接实现在 notebook 内。

## 4. 准备复现环境

所有命令都在**项目根目录**运行。需要可用的 Python 3；本地 `.venv` 不随仓库分发。

创建虚拟环境：

```shell
python -m venv .venv
```

使用该环境的解释器安装依赖。

**Windows PowerShell：**

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirement.txt
```

**macOS / Linux：**

```shell
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirement.txt
```

下文命令中的 `python` 均指虚拟环境解释器。请先激活环境，或在 Windows 上替换为 `.\.venv\Scripts\python.exe`，在 macOS/Linux 上替换为 `.venv/bin/python`。在 JupyterLab 中选择对应环境作为 notebook kernel。

`requirement.txt` 规定的是依赖版本下限，并非完整的版本锁定文件。正式复现时请记录 Python 和已安装依赖的版本，不要假定任意满足下限的依赖组合都与生成已有输出的环境完全相同。

## 5. 数据和调参前置条件

### 数据与缓存

- 比赛/赛季列表和划分规则定义在 notebook 及其数据模块中。核对研究样本时，请以这些定义为准。
- 缺少所需缓存时，xA、xG、xT 和标准 VAEP 流程会获取 StatsBomb open data 并重建中间表；首次无缓存运行需要网络访问。Atomic-VAEP 本身不会下载原始事件。
- `data/`、`open-data/` 和 `vaep_data/` 被 Git 忽略。新克隆的仓库不应被视为已经包含原机器上的数据或 prepared caches。
- `vaep_data/` 存放可复用的中间数据。从无缓存环境开始时，应先运行标准 VAEP，在该目录生成 `women_spadl_ltr.parquet`、`men2015_spadl_ltr.parquet` 和必需的 `women_pos.parquet`，再运行 Atomic-VAEP。
- Atomic-VAEP 优先读取 `women_atomic_spadl_ltr.parquet` 和 `men2015_atomic_spadl_ltr.parquet`；缺少时，会将现有 SPADL 缓存转换为这些 Atomic-SPADL 缓存。如果两种动作缓存都不存在，或缺少女足球员位置缓存，它会抛出 `FileNotFoundError`，而不是从原始 StatsBomb 事件重新构建。

### 必需的调参 JSON 文件

xA、xG 和标准 VAEP notebook 会读取选中的 hyperparameters，然后**在 notebook 内重新训练 fresh models**，而不是直接加载调参阶段生成的 `.joblib` 模型。xT 直接学习威胁地图；Atomic-VAEP 使用其 notebook 内定义的固定参数拟合概率模型。在这一实验基线中，xT 和 Atomic-VAEP 都不需要调参 JSON。

| 实验 | Notebook 实际读取的路径 | 当前代码快照中的情况 |
| --- | --- | --- |
| xA | `xa/xa_all_models/xa_original_training_results.json` | 此路径下的文件未纳入仓库；重新运行 xA 前需生成 |
| xG | `xg/xgboost_tuning_results/xgboost_xg_tuning_results.json` | 已包含 |
| VAEP | `vaep/hist_gradient_boosting_tuning_results/vaep_training_results.json` | 已包含 |

**如果所需 JSON 已经存在，可以跳过调参。** 缺少文件时，运行对应命令：

```shell
python -u xa/train_original_xa_models.py --model-family all --output-dir xa/xa_all_models
```

```shell
python -u xg/train_xgboost_xg_models.py
```

```shell
python -u vaep/train_vaep_probability_models.py --search-n-jobs 1
```

对 xA 而言，`--model-family xgboost` 已足以支持 notebook；`all` 还会运行逻辑回归比较。保存在其他目录中的 xA 结果不会被 notebook 自动读取。

如果有意重新生成调参结果，请先保留现有产物。xA 和 VAEP 脚本需要显式传入 `--overwrite` 才能覆盖已有产物；xG 也只应在明确希望替换其结果时重新运行。正式复现时不要用 `--quick` 冒烟测试的结果替代完整搜索结果。

模块说明：[xA](xa/README.md)、[xG](xg/README.md)、[VAEP](vaep/README.md)。Atomic-VAEP 的缓存处理、特征构造和模型配置直接说明在其 notebook 中。

## 6. 运行实验

从项目根目录启动 JupyterLab：

```shell
python -m jupyter lab
```

**每个 notebook 使用独立的新 kernel，并从上到下运行。** 这也可以避免在同一 kernel 中混用不同模型目录里的同名辅助模块。

建议顺序：

1. xA：`enriched_xa_features_competition_events.ipynb`
2. xG：`enriched_xg_features_competition_events.ipynb`
3. xT：`enriched_xt_features_competition_events.ipynb`
4. VAEP：`enriched_vaep_features_competition_events.ipynb`
5. Atomic-VAEP：`enriched_atomic_vaep_features_competition_events.ipynb`

中英文版本执行相同代码，无需把两种语言各跑一次。预期输出包括数据划分/训练摘要、球员层面的统计量，以及审稿导航表中列出的表格和图像。调参脚本另行保存搜索产物；生成的数据和缓存不属于语言副本。

无缓存数据准备和完整超参数搜索可能耗时较长。本仓库未提供经过测量的全流程运行时间或最低内存保证。上述 VAEP 调参命令使用单个搜索 worker，以减少内存复制，但这不代表已经确定了固定的内存需求。Atomic-VAEP 直接在 notebook 内拟合已配置的分类器。

## 7. 审稿时需要注意的方法细节

- **实验内对齐：** 每个主要 notebook 内的模型在同一批女足测试事件上比较。不同指标使用不同事件集合和球员筛选条件，不能假定五个 notebook 的球员样本或测试比赛完全相同。
- **数据划分：** xA、xG 和 VAEP 的数据模块在调参/训练前预留女足测试比赛。配置的随机种子为 `42`；xA 的女足测试比例为 `0.20`，xG 和 VAEP 为 `0.30`。xT 和 Atomic-VAEP 在各自 notebook 中定义划分；Atomic-VAEP 同样使用种子 `42` 和 `0.30` 的测试比例。核对可比性时应检查实际 split summary。
- **训练数据规模：** xA、xG、VAEP 和 Atomic-VAEP 包含男足下采样模型与女足模型的比较；xG 还拟合男足全量模型。xT 使用其男足动作池训练，没有同样的行级样本量对齐步骤。
- **超参数：** xA、xG 和标准 VAEP 的女足 fresh models 也复用男足数据选出的参数；VAEP 对 `scores`、`concedes` 分别选参。Atomic-VAEP 则对两个目标和两个训练组使用同一套固定 HGB 配置（`max_iter=300`、`learning_rate=0.05`、`random_state=42`）。xG 原始中文第 5 节仍写了“女组最佳参数”，但实际代码将同一组 `BEST_MEN_XGBOOST_PARAMS` 用于全部三个模型；英文版已明确说明这一点，翻译没有更改实验代码或已有结果。
- **序列完整性：** 标准 VAEP 在完整比赛序列上准备 features/labels 后再下采样。Atomic-VAEP 在每场被选中比赛的完整序列上计算 features/labels，再在逐场处理时保留抽样选中的行。两者的特征窗口均为 3 个动作、标签窗口均为 10 个动作，分别以标准动作和原子动作为单位。
- **结果解释：** Spearman 衡量的是两个模型给出的球员排名一致性，不是真实球员能力的直接度量，也不同于事件层面的预测表现。数据来自单一提供商，男女足样本还存在赛季和赛事覆盖差异，因此不能把迁移差异解释为仅由性别造成。Position-only xT 的一致性也不能证明其他战术或上下文差异不存在。摘要将使用同一赛季、时间对齐的男女足数据作为后续研究的重要方向。

独立复现时，请保留代码版本、环境信息、数据/缓存来源、调参 JSON 和重新执行后的 notebook。阅读已有输出与独立复现这些输出是两种不同的审稿活动。
