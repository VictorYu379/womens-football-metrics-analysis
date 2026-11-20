# Atomic-VAEP：C + adapter 与纯女足模型 D 的比较

这个文件夹是一份可独立读取模型、重新评分和画图的分享包。**主问题：同样有 540 场女足开发比赛，使用已有男足模型加 adapter，与只用女足数据从头训练，预测结果和球员评价有什么不同？**

主比较是 **D vs C + Beta adapter**。未加 adapter 的 C 用来解释概率调整层改变了什么；A 和 C 的新继续训练是辅助对照。不是继续比较所有女足数据量，也不把 D 的球员评分当作正确答案。

先打开 `01_compare_atomic_vaep.ipynb`：第 1 节是模型配置，分析沿用原 `enriched_atomic_vaep_features_competition_events.ipynb` 第 4.1–8.2 节的顺序与模式，第 9 节是辅助比较。只加载模型，不重新训练，不修改原实验目录或原 enriched notebook。

## 1. 数据与共同定义

数据来自原项目的 StatsBomb open-data 比赛及已经转换好的 Atomic-SPADL / Atomic-VAEP 缓存。本包不重新下载事件，也不重新生成训练特征。

| 数据 | 数量 | 用途 |
| --- | --- | --- |
| 女足开发池 | 540 场，1,586,819 条原子动作 | 训练 D；训练 adapter；男足模型的继续训练 |
| 女足最终测试集 | 231 场，669,582 条原子动作 | 所有模型使用完全相同的动作和标签进行评估与估值 |
| 男足完整训练池 | 4,583,701 条原子动作 | 训练 C；A、B 从中取固定子集 |
| 模型输入 | 256 个有序特征 | 使用原项目修正后的 v2 特征，五动作状态窗口 |

540 场开发比赛和 231 场测试比赛不重叠。比赛清单与特征顺序在 `inputs/manifest.json`。这里的“用满 540 场”是开发预算，不意味着每条动作都用于树的拟合：树模型还从开发数据中预留内部验证集，见第 3 节。

每个模型都是一对独立二分类模型：

- `scores.joblib`：预测当前动作所属球队在当前及随后 9 个原子动作内进球的概率。
- `concedes.joblib`：预测当前动作所属球队在相同窗口内失球的概率。
- 两个概率按 Atomic-VAEP 公式转换为动作的进攻贡献、防守贡献和总贡献。因此下文“22 对模型”对应 **44 个 joblib 文件**，不是 22 个单独的分类器。

## 2. ABCD 四个基础模型

| 模型与路径 | 开发数据 | 怎么得到 | 本 notebook 的角色 |
| --- | --- | --- | --- |
| **A**：`models/A/` | 1,586,819 条男足原子动作 | 只用男足数据从头训练 HGB；动作数与女足开发池相同 | 辅助：较小男足数据量基线 |
| **B**：`models/B/` | 3,085,260 条男足原子动作 | 只用男足数据从头训练 HGB | 已导出，默认不加载、不画图 |
| **C**：`models/C/` | 全部 4,583,701 条男足原子动作 | 只用男足数据从头训练 HGB | 辅助：adapter 所依赖的原始男足模型 |
| **D**：`models/D/` | 全部 540 场女足开发比赛，1,586,819 条原子动作 | 不使用男足模型，只用女足数据从头训练 HGB | **主比较：纯女足训练方案** |

A、B 是男足动作行的固定随机子集：对完整男足行索引用 `RandomState(42)` 做一次排列，取对应长度的前缀，再按原行顺序排列，因此 **A ⊂ B ⊂ C**。A 的动作数等于女足开发池的动作数，不是“恰好 540 场完整男足比赛”。男足基础模型不会因 adapter 的代表 seed 不同而重新抽样。

ABCD 均使用 `HistGradientBoostingClassifier`，不是 XGBoost。基础模型的参数如下；继续训练沿用来源模型的树参数和学习率。

| 参数 | scores | concedes |
| --- | --- | --- |
| learning_rate | 0.016 | 0.010 |
| max_leaf_nodes | 127 | 63 |
| min_samples_leaf | 35 | 50 |
| max_iter | 800 | 1200 |
| l2_regularization | 0.75 | 0.75 |
| random_state | 42 | 42 |
| validation_fraction | 0.1 | 0.1 |
| n_iter_no_change / tol | 10 / 1e-7 | 10 / 1e-7 |

`max_iter` 是上限，不是实际轮数；实际轮数、验证曲线和耗时在各模型目录的 `fit.json`。继续训练的 800/1200 是**最多新增轮数**，并非来源树与新增树合计的上限。

## 3. 五种使用女足数据的方法

每种方法分别从 A、B、C 出发，且本包仅保留 **540 场女足预算**。adapter 冻结男足 HGB，只改变它输出的概率；继续训练保留原有树，再用女足数据拟合新增树。两者不是同一类操作。

### 四种 adapter

设 `p` 是男足模型输出的概率，`q` 是调整后的概率，`sigmoid(z) = 1 / (1 + exp(-z))`，`logit(p) = log(p / (1-p))`。计算时先将 `p` 截断到 `[1e-8, 1-1e-8]`。scores、concedes 分别训练自己的系数。

| 方法 | 计算形式 | 每目标新增参数 / 两目标合计 | 解释 |
| --- | --- | --- | --- |
| `intercept` | `q = sigmoid(logit(p) + b)` | 1 / 2 | 只调整整体概率水平，不重新学习原始比赛特征 |
| `logistic` | `q = sigmoid(b + a * logit(p))` | 2 / 4 | 同时调整整体水平和概率的拉伸程度 |
| `beta` | `q = sigmoid(b + a * log(p) - c * log(1-p))` | 3 / 6 | 用两项对数特征，更灵活地调整不同概率区间；是本次主比较所用类型 |
| `context` | `q = sigmoid(b + a * logit(p) + sum(d_t * I[type=t]))` | 8 / 16 | logistic 的 2 个参数，加六类动作各自的偏移量 |

`context` 的六类是 pass、receival、interception、dribble、clearance、tackle；其他动作共用基础项。它不是 MLP，也不直接加入 event position。四种 adapter 都是在已有概率上训练小型调整层，不是重新训练 HGB。上表只统计**新增**参数，男足基础模型仍不可缺少。

- intercept、logistic、beta 在所选全部女足开发动作上拟合二元 log loss。
- context 先按比赛做 80%/20% 的内部划分，选择六个动作偏移量的 L2 正则强度，再用全部开发动作重新拟合。候选正则强度为 `0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1`。
- 所有 adapter 使用带边界的 L-BFGS-B 优化；logistic/context 的斜率为正，beta 的两个对数项系数非负。原始实现和系数限制见 `provenance/training_code_at_export/adapters.py`；推理实现见 `inference.py`。

### 继续训练的验证方式

本包只保留 `continued_match`：预留约 10% 的完整女足比赛作内部验证，同场动作不跨训练／验证两侧。C 的版本用作辅助对照。旧 `continued` 按动作行划分内部验证，现只留在原 `paper_experiments`，不在本分享包中。

ABCD 的原有训练仍使用按动作行的内部验证。新继续训练修正了续训的验证粒度，并没有重新训练 D。树模型提前停止后直接使用保存的模型，不再把内部验证集并回去重训，也不回滚到最低验证 loss 的轮次。以上是现有实验规则，不是本分享包新增的方法。

## 4. 每一对导出模型在哪里？

基础模型的四个路径见第 2 节。以下 15 对全部使用 540 场女足预算；目录中的 `s17` 等表示所保留的代表 seed。

| 模型 | 路径 | 说明 |
| --- | --- | --- |
| A + intercept | `runs/A/intercept/b540_s17/` | 冻结 A，只调整概率截距 |
| A + logistic | `runs/A/logistic/b540_s17/` | 冻结 A，调整概率截距和斜率 |
| A + beta | `runs/A/beta/b540_s17/` | 冻结 A，使用三参数 Beta 层 |
| A + context | `runs/A/context/b540_s43/` | 冻结 A，加入动作类别偏移 |
| A + 新继续训练 | `runs/A/continued_match/b540_s17/` | 从 A 新增树，按完整比赛验证 |
| B + intercept | `runs/B/intercept/b540_s17/` | 冻结 B，只调整概率截距 |
| B + logistic | `runs/B/logistic/b540_s17/` | 冻结 B，调整概率截距和斜率 |
| B + beta | `runs/B/beta/b540_s17/` | 冻结 B，使用三参数 Beta 层 |
| B + context | `runs/B/context/b540_s71/` | 冻结 B，加入动作类别偏移 |
| B + 新继续训练 | `runs/B/continued_match/b540_s17/` | 从 B 新增树，按完整比赛验证 |
| C + intercept | `runs/C/intercept/b540_s17/` | 冻结 C，只调整概率截距 |
| C + logistic | `runs/C/logistic/b540_s17/` | 冻结 C，调整概率截距和斜率 |
| **C + beta** | **`runs/C/beta/b540_s17/`** | **主比较：冻结 C，使用三参数 Beta 层** |
| C + context | `runs/C/context/b540_s71/` | 冻结 C，加入动作类别偏移 |
| C + 新继续训练 | `runs/C/continued_match/b540_s17/` | 辅助对照：从 C 新增树，按完整比赛验证 |

合计 **4 对基础模型 + 3 个男足来源 × 5 种方法 = 19 对 / 38 个 joblib 文件**。本包不包含50–500场版本或旧继续训练模型；原 `paper_experiments` 的所有数据量和四个 seed 的实验结果仍完整保留。

ABCD 和继续训练的 joblib 包含完整 HGB；**adapter 的 joblib 只包含调整层参数**，必须配合同源的 `models/A`、`models/B` 或 `models/C`。`inference.load_pair(HOME, relative_path)` 自动校验并加载对应基础模型，不能直接把 adapter 参数文件当作分类器。

## 5. 为什么选 Beta，为什么保留这个 seed？

**选择 adapter 类型：**在 C、540 场条件下，Beta 的四次平均 test log loss 对 scores 和 concedes 都最低，因此作为主比较。不是先看球员名单，再选最接近 D 的 adapter。候选的均值、标准差和次数在 notebook 第 1.1 节与 `results/adapter_selection.csv`。

**选择代表 seed：**对每个来源／方法的四次实验（17、29、43、71），分别计算两个目标的平均 test loss。每个 seed 的距离为：

```text
distance = 0.5 * (
    abs(scores_loss - mean_scores_loss) / mean_scores_loss
    + abs(concedes_loss - mean_concedes_loss) / mean_concedes_loss
)
```

取距离最小者；差异在 `1e-12` 内时选较小 seed。两个目标始终来自同一个 seed，不挑最佳 seed、不拼接不同实验，也不进行模型集成。`seed_selection.csv` 保留这15个方案的全部60条候选记录及15个最终选择。

540 场时四个 seed 使用相同的完整开发池，不是四批独立比赛；确定性的方案可能得到完全相同的结果，context 的内部正则选择仍可能不同。表中的标准差不能当作独立测试样本的不确定性。

**adapter 类型与代表 seed 都使用了测试结果进行选择。** 本 notebook 是测试后选定模型的描述性分析，不能再把同一批比赛称为独立的选择后验证集。论文需披露这一点；“best”只指这里的候选、预算、指标和测试集，不是普遍最优。

## 6. 如何阅读分析结果？

本次不另起一套简化展示，而是把原 enriched notebook 中的两组男女模型替换为 **C+adapter 与 D**。对照关系如下：

| Notebook 章节 | 继承的分析模式 |
| --- | --- |
| 4.1 概率诊断 | 两个目标分别报告 log loss、Brier、ROC AUC、0.5 阈值错误率和正例比例；稀有正例下，低错误率不等于有用的概率预测 |
| 5 完整比赛估值 | 用 Atomic-VAEP 公式逐场估值，保留门将、进球、零值和负值；之后才排除门将做动作分析 |
| 5.1 动作类型 | 完整均值、中位数、标准差、总差等统计；山脊图显示差异分布，横轴固定 ±0.01、峰高归一；超范围动作计数报告且仍留在统计里 |
| 5.1.1 传球高度 | 用原始事件区分传球和任意球射门；按原子类型 × Ground/Low/High/Unknown 分组，保留所有传球，不只看成功传球 |
| 5.2 场地区域 | shot、pass、receival、dribble、cross、take_on 六图，12×16 网格；射门至少10条、其他类型至少20条；共用对称色标，颜色截在合格网格绝对均值的第98百分位 |
| 6–6.3 总贡献 | 总体与位置内排名表、标注差异球员的散点图、Spearman 条形图、位置差异分布、±10整数名次变化热图、位置散点图 |
| 7–7.4 每90分钟 | 复用同一球员群体，完整重复总贡献分析；两套名次热图共用绝对人数色标；再并排比较两种评分基础 |
| 8–8.1 位置 Top 15 | 分别比较总分和每90分钟名单，报告交集、各自独有球员及完整并集，保留完整位置组内名次 |
| 8.2 归一化造成的名单变化 | 在 D、adapter 各自内部比较总分与每90分钟名单，报告保留／进入／退出，不与模型间差异混淆 |
| 9 辅助分析 | C−D、C+adapter−C 用于解释调整前后；A、新继续训练和更高分钟门槛作为补充 |

主比较的贡献差统一为 **adapter − D**；名次差 `rank_delta = adapter_rank − D_rank`，正数表示 adapter 排得更靠后。`rank_improvement_per90 = total_rank − per90_rank` 则表示同一模型归一化后的名次改善，正数为上升，含义不同。

排名规则也同步原 notebook：降序排名使用 `method='min'`。主指标是 Spearman、Kendall tau-b、平均绝对名次差除以 `人数−1`；Top 10 是补充排名诊断，不等于位置 Top 15。位置人数不足15则排除；Top 15 保留所有完整组内排名≤15的人，**不再用 player_id 强行截成15人**。边界同分使名单扩张时，固定15人的重合百分比留空。完整组名次不会在名单子集上重新计算。

所有球员分析复用非门将、至少100个测试动作且45分钟的同一群体；分钟数包含有出场却没有原子动作的比赛。位置仍为缓存女足事件中最常记录的位置。空间图为动作级分析，不使用球员出场门槛，也不是每90分钟地图。坐标统一为当前动作所属球队向右进攻；传球／带球／过人取起点，接球取接球点，射门取射门点。

原 notebook 有训练集诊断，本分享包没有完整训练输入，因此只报告共同测试集的概率指标，不把验证曲线冒充训练集结果。`main_loss_comparison.csv` 给出 adapter 相对 D 的绝对及百分比 loss 差；这些是点估计，不是显著性结论。模型、概率及 Atomic-VAEP 公式未因调整分析模式而改变。

**D 不是球员贡献真值。** 高相关、名单重合或更接近 D 不能证明球员评价更准确，也不能证明某套阵容表现更好。45 分钟门槛是沿用原分析的选择，不代表评分已稳定。此包不重训模型来消除原有内部验证、提前停止等局限。

## 7. 文件索引与复现

| 路径 | 内容 |
| --- | --- |
| `01_compare_atomic_vaep.ipynb` | 主入口；显式选择、加载、推理、估值、汇总和显示图表 |
| `model_catalog.csv` / `export_manifest.json` | 19 对模型的路径、来源、方法、seed、基础模型依赖、文件校验和及依赖版本 |
| `seed_selection.csv` | 540 场方案的四次 loss、均值、距离和是否选中 |
| `inputs/` | 相同 231 场测试比赛的有序特征、完整动作、球员位置和出场时间快照 |
| `inputs/pass_height_events.parquet` | 测试比赛原始 Pass/Shot 事件的 ID、类型和传球高度；独立运行无需再访问原项目的事件 JSON |
| `models/` / `runs/` | 已复制的模型本体与原始训练记录，不是链接 |
| `references/` | 基础模型及 C 的 540 场代表模型原预测，用于核对重新推理 |
| `provenance/` | 基础模型和540场方案的原始指标、训练协议及数据清单；另保留 adapter 实现和依赖，供一致性测试使用 |
| `inference.py` / `analysis.py` / `plotting.py` | 模型推理、Atomic-VAEP 与统计、图表函数 |
| `results/selection.json` / `results/selected_models.csv` | 本次主模型、辅助模型、路径、seed 与球员门槛 |
| `results/probability_metrics.csv` / `results/main_loss_comparison.csv` | 全部已分析模型的预测指标，以及 D vs adapter 的直接 loss 比较 |
| `results/action_values.parquet` | 完整逐动作概率与进攻／防守／总贡献，可重新分析和画图 |
| `results/action_type_differences.csv` / `results/spatial_differences.csv` | 主比较的动作与场地区域差异 |
| `results/pass_height_differences.csv` | 全部实际存在的传球类型×高度分组，包括不足绘图门槛的组 |
| `results/action_type_source_context.csv` | C−D 与 C+adapter−C 的辅助分解 |
| `results/player_values.csv` / `results/positional_top15.csv` | 球员每 90 分钟评分、总分、排名与各模型名单 |
| `results/total_ranking_summary.csv` / `results/per90_ranking_summary.csv` | 主比较的总体与位置内排名指标 |
| `results/total_rank_movement_counts.csv` / `results/per90_rank_movement_counts.csv` | 每位置、每个整数名次差的人数，以及未显示的超范围人数 |
| `results/top15_scope.csv` / `results/top15_model_comparison.csv` | 位置是否足够15人、最低可能重合率，以及两模型名单交集与独有人数 |
| `results/top15_model_union.csv` / `results/top15_overlap_change.csv` | 两模型完整名单并集，以及总分改成每90分钟后的模型间重合率变化 |
| `results/top15_normalization_summary.csv` / `results/top15_normalization_union.csv` | 各模型内部归一化前后的保留／进入／退出与完整名单并集 |
| `results/figures/` | `01` 概率；`02` 动作分布；`03` 传球高度；`04` 空间；`05`–`08` 总贡献；`09`–`12` 每90分钟；`13`–`14` 辅助比较 |
| `tests/` | 导出、推理、Atomic-VAEP 校验，以及与原 notebook 的排名指标、边界同分、名单并集等一致性测试 |

`provenance` 的指标仅覆盖本包中的基础模型和540场方案；每个适配方案仍保留四个 seed 的结果，未改成只留选中的 seed。训练 notebook、其余训练源码和完整学习曲线留在原 `paper_experiments`。原始 `fit.json` 可能引用未打包的训练行索引或预测缓存；本包文件完整性以 `export_manifest.json` 为准，历史绝对路径不影响这里的推理与分析。

**查看：**直接用 JupyterLab 打开已执行 notebook，或读取 `results/figures/`。**重新运行：**使用兼容的 Python 环境打开 notebook，按顺序运行即可；首次重新推理，之后校验并复用预测缓存。没有训练步骤，不需要父目录的比赛数据、原 notebook 或 `paper_experiments`。

`results/predictions/` 是可以重新生成的预测缓存，本次完整目录分享也保留已有缓存。Notebook 会检查缓存与模型、输入、推理代码是否一致；缺失或失效时只重新预测，不重新训练，并与 `references/` 中的原始预测核对。已保存的逐动作贡献、球员评分、结果表和图仍可直接查看。

运行时环境如下；HGB 的 joblib 必须使用相同 scikit-learn 版本，加载函数会检查。只加载可信来源的 joblib 文件。

| 组件 | 已验证版本 |
| --- | --- |
| Python | 3.13.3 |
| scikit-learn / joblib | 1.9.0 / 1.6.0 |
| numpy / pandas / scipy | 2.5.3 / 3.0.5 / 1.18.1 |
| pyarrow / socceraction | 25.0.1 / 1.4.2 |
| matplotlib / mplsoccer | 3.11.1 / 1.8.0 |
| nbformat / nbclient | 5.11.1 / 0.11.0 |
| ipykernel / jupyterlab | 7.3.0 / 4.6.3 |

这是**已验证的运行环境记录，不是保证能由 pip 一步解析的依赖清单**。旧 `socceraction 1.4.2` 的包元数据要求 NumPy、Pandas 小于 2，而现有环境使用更新版本；`analysis.py` 对 Atomic-VAEP 公式需要的旧 NumPy 名称做兼容处理，并有公式一致性测试。异机重跑时优先复用团队已验证环境，不要盲目升级或降级依赖；新建环境后先运行测试。

在本目录运行测试：

```text
python -B -m unittest discover -s tests -q
```

`export_models.py` 仅供拥有原项目的人重新导出：只选择540场的四种 adapter 和新继续训练，校验模型副本，遇到不同模型内容停止，不覆盖来源模型；指标表按此范围重新生成。它不复制训练 notebook 或整套训练源码，也不生成分析缓存。它不是同事阅读或运行分析的前置步骤。此分享包不包含完整训练数据，不能单独复现所有训练。

## 8. 通过 Git 分享完整目录

本目录的 `.gitignore` 允许跟踪全部文件，并覆盖根目录对 `data/` 等名称的忽略规则。模型、输入、参考预测、指标表、图表、清单以及目录中已有的缓存和自动备份都纳入完整目录分享，不再只上传代码。

`.gitattributes` 将 `.joblib`、`.npy`、`.parquet` 文件交给 Git LFS；其余文件正常保存在 Git 中。目录结构和 notebook 的读取路径不变。该文件也关闭本目录的自动换行转换，保留文件的原始字节，避免跨平台检出后清单中的 SHA-256 校验失败。

同事首次克隆前，应安装 Git LFS 并运行 `git lfs install`，再正常克隆仓库。已有克隆可在仓库内执行：

```text
git lfs install --local
git lfs pull
```

确认 LFS 文件已经下载，再运行 notebook 或测试；只拿到 Git 中的小型指针文件不能代替真实模型与数据。完整上传还受远端账户的 LFS 存储和流量配额约束。
