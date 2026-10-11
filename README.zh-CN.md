# Who Gets the Credit?

**跨性别模型迁移与女足球员价值评估**

本项目是[论文摘要](Who_Gets_the_Credit_Abstract_LaTeX_Style.pdf)的配套研究代码。研究关注：在男足数据上训练的模型，是否会像女足数据训练的模型一样认可女足球员的贡献？论文不只比较事件层面的预测表现，还考察**球员排名的稳定性**，尤其关注后卫、承担混合角色的球员，以及非射门动作的价值。

## 1. 研究概述

**背景：** 球员价值评估模型可用于球探工作和人才识别，但事件层面的预测表现相近，不代表在不同训练数据下，球员排名也同样一致。

**研究问题：** 在男足和女足数据上训练的模型，是否会对同一批女足球员给出相近的估值和排名？我们比较 xG、xA、xT、VAEP 和 Atomic-VAEP 五种框架。

**主要发现：** 摘要报告，xG、xA、xT 和 VAEP 的排名一致性较高（Spearman 为 0.954–0.977），而 Atomic-VAEP 较低（0.652），后卫尤其明显。拦截和接球解释了相当一部分排名不稳定性，提示在迁移模型评估防守与转换贡献时需要谨慎。

## 2. 审稿人阅读指南

**如果只希望查看结果，可以直接打开下表中的 notebook，无需重新训练或下载数据。** 独立复现这些结果则需要运行下文说明的实验流程。

| 实验 | Notebook | 建议查看的结果 |
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

## 3. 目录结构

```text
README.zh-CN.md                    项目概述与审稿指南
Who_Gets_the_Credit_Abstract_LaTeX_Style.pdf   附带的论文摘要及图像
*.ipynb                          实验 notebook
xa/                               xA 数据准备、模型和调参
xg/                               xG 数据准备、模型和调参
vaep/                             VAEP 数据准备和概率模型
requirement.txt                   运行依赖
```

Notebook 继续放在项目根目录，以保持现有相对路径有效。Atomic-VAEP 的数据准备和模型拟合直接实现在 notebook 内。

## 4. 准备复现环境

**实验基线：** `824d701`。

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

## 5. 数据来源与准备

研究使用 StatsBomb Open Data，包含摘要中所述的 2023/24 赛季 771 场女足比赛和 2015/16 赛季 1,517 场男足比赛。

建模前的数据处理包括：

1. **加载事件：** `xa/xa_data.py` 和 `xg/xg_data.py` 加载成功传球和射门数据，优先复用本地缓存，缺少时从 StatsBomb 获取。
2. **构建动作表示：** xT notebook 和 `vaep/vaep_data.py` 准备进攻方向统一为从左到右的 SPADL 动作；Atomic-VAEP 再将缓存中的 SPADL 转换为 Atomic-SPADL。
3. **准备模型输入：** 构建各指标的特征和标签，按比赛预留女足测试集，并在相应实验中对男足训练数据下采样。

数据和中间表缓存在 `data/` 和 `vaep_data/`。从无缓存环境开始时，应先运行标准 VAEP，生成 Atomic-VAEP 所需的 SPADL 和球员位置缓存。

## 6. 运行实验

运行 xA 前，如缺少调参结果，请按 [xA 模块说明](xa/README.md)生成，并指定 `--output-dir xa/xa_all_models`。仓库已包含 xG 和 VAEP 的调参结果，详见 [xG](xg/README.md) 和 [VAEP](vaep/README.md) 模块说明。

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

预期输出包括数据划分与训练摘要、球员层面的统计量，以及审稿导航表中列出的表格和图像。调参脚本另行保存搜索产物。

首次数据准备和完整调参可能耗时较长，运行时间与内存需求取决于环境。

## 7. 审稿时需要注意的方法细节

- **实验内对齐：** 每个主要 notebook 内的模型在同一批女足测试事件上比较。不同指标使用不同事件集合和球员筛选条件，不能假定五个 notebook 的球员样本或测试比赛完全相同。
- **数据划分：** xA、xG 和 VAEP 的数据模块在调参/训练前预留女足测试比赛。配置的随机种子为 `42`；xA 的女足测试比例为 `0.20`，xG 和 VAEP 为 `0.30`。xT 和 Atomic-VAEP 在各自 notebook 中定义划分；Atomic-VAEP 同样使用种子 `42` 和 `0.30` 的测试比例。核对可比性时应检查实际 split summary。
- **训练数据规模：** xA、xG、VAEP 和 Atomic-VAEP 包含男足下采样模型与女足模型的比较；xG 还拟合男足全量模型。xT 使用其男足动作池训练，没有同样的行级样本量对齐步骤。
- **超参数：** 每个实验中的男足训练模型与女足训练模型使用相同的超参数配置。具体设置以相应实验代码为准。
- **序列完整性：** 标准 VAEP 在完整比赛序列上准备 features/labels 后再下采样。Atomic-VAEP 在每场被选中比赛的完整序列上计算 features/labels，再在逐场处理时保留抽样选中的行。两者的特征窗口均为 3 个动作、标签窗口均为 10 个动作，分别以标准动作和原子动作为单位。

独立复现时，请保留代码版本、环境信息、数据/缓存来源、调参 JSON 和重新执行后的 notebook。阅读已有输出与独立复现这些输出是两种不同的审稿活动。

## 8. 结果解读与局限性

- Spearman 衡量的是两个模型给出的球员排名一致性，不是真实球员能力的直接度量，也不同于事件层面的预测表现。
- 数据来自单一提供商，男女足样本还存在赛季和赛事覆盖差异，因此不能把迁移差异解释为仅由性别造成。
- Position-only xT 的一致性也不能证明其他战术或上下文差异不存在。
- 摘要将使用同一赛季、时间对齐的男女足数据作为后续研究的重要方向。
