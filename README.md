# GMCM-25B-Reproduction-

2025 年中国研究生数学建模竞赛 B 题《无线通信系统链路速率建模》的**第一题**可执行复现。

依据仓库已有的优秀论文《机器学习驱动的面向 MIMO-OFDM 的链路速率预测》第5章和附录2，
实现 **CSI → 逐载波 SINR → dB 域 EESM → Isotonic → HGBR → 融合 → 速率标签预测**。
本次仅包含第一题，第二、三题未实现。

- [论文与代码逐项对应](docs/问题一-论文代码对照.md)：公式、段落、代码入口、正文与附录差异。
- [实际复现结果](docs/问题一-复现结果.md)：论文训练指标、本次训练指标、独立设备验证、待预测集结果。
- [原优秀论文](docs/B题-机器学习驱动的面向%20MIMO-OFDM%20的链路速率预测.pdf)。
- [第一题完整预测表](outputs/q1/valid_predictions.csv)。
- [真实派生数据与溯源说明](data/derived/README.md)。

## 运行（Anaconda FirstPythonEnv）

在本项目目录打开 Anaconda Prompt：

```console
conda activate FirstPythonEnv
python scripts/run_q1.py --source derived
```

仓库提供真实数据生成的122维 SINR、标签与逐行来源信息，可直接重跑，无需先下载原始836 MB Excel。
程序会核验派生数据的 SHA-256，然后训练并执行外层按设备隔离、内层重新选参数的完整验证。

Windows 上无需激活环境也可运行（本次实测路径）：

```powershell
& 'D:\Anaconda3\envs\FirstPythonEnv\python.exe' -X utf8 scripts/run_q1.py --source derived
```

依赖版本见 [requirements.txt](requirements.txt)；仅在环境缺少依赖时运行
`python -m pip install -r requirements.txt`。
本次使用现有 FirstPythonEnv 完成计算，无需更换环境。
若新建环境，可参考 [environment.yml](environment.yml)。

```console
python scripts/run_q1.py --source derived --skip-audit
python scripts/run_q1.py --source raw --prepare-only
python scripts/verify_data.py
python scripts/run_tests.py
```

`--skip-audit` 只省略耗时的设备隔离验证，不能据此声称获得独立验证成绩。
默认线程数4，可用 `--threads` 调整。
代码搜索完整参数网格及330次局部扰动，不是小样本演示。

## 从原始 CSI 重建

原始数据来自用户提供的 [Google Drive 文件夹](https://drive.google.com/drive/folders/1e-4Tmme_lDIbw_XVc8GZiD-9bjxRxydC)。
第一题使用 `notxbf_com_excels_f4` 下3个设备的 train / valid，共6个 Excel；
开启波束赋形的另一目录不属于本次范围。

将文件按 [data/manifest.json](data/manifest.json) 的 `path` 保存到 `data/raw/`：

```text
data/raw/notxbf_com_excels_f4/
  341c-f0d4-70be/train/341c-f0d4-70be_0.xlsx
  341c-f0d4-70be/valid/341c-f0d4-70be_0.xlsx
  bc98-2983-907b/train/bc98-2983-907b_0.xlsx
  bc98-2983-907b/valid/bc98-2983-907b_0.xlsx
  d0c1-bfe4-18f0/train/d0c1-bfe4-18f0_0.xlsx
  d0c1-bfe4-18f0/valid/d0c1-bfe4-18f0_0.xlsx
```

然后运行 `python scripts/run_q1.py --source raw`。
原始 Excel 不进入普通 Git；来源、字节数、SHA-256、清洗统计和完整派生数据进入版本管理。
需要重新定义物理预处理时，必须从原始 CSI 重算，而非仅修改派生 SINR。

## 文件结构

| 路径 | 内容 |
|---|---|
| `src/gmcm25b/data.py` | 安全解析8列复数CSI、噪声换算、SINR、清洗、缓存、逐行溯源 |
| `src/gmcm25b/features.py` | 附录dB EESM、31个统计/物理特征、来源编码 |
| `src/gmcm25b/models.py` | EESM搜索、保序回归、HGBR、融合与最近标签映射 |
| `scripts/run_q1.py` | 完整训练、预测、嵌套设备验证、图表和报告 |
| `data/derived/` | 可重跑模型的真实派生数据及校验值 |
| `outputs/q1/metrics.json` | 本次参数、训练/外层验证指标、版本及耗时 |
| `outputs/q1/parameter_search.csv` | 全量模型参数搜索记录 |
| `outputs/q1/audit/` | 每个外层设备验证的参数与指标 |
| `outputs/q1/valid_predictions.csv` | B集每条记录的SINR等效量、连续预测、最近标签及来源 |
| `outputs/q1/required/` | 按设备分开的原始index/mcs两列结果 |
| `outputs/q1/q1_model.joblib` | 运行时生成的模型（可重新生成，不进入Git） |
| `tests/` | 数值正确性、数据保序、分组验证与模型恢复测试 |

逐载波 SINR 在 `data/derived/q1_valid.npz` 的 `sinr` 数组中，
第 i 行与 `q1_valid_rows.csv` 第 i 条记录及预测总表严格对应；第 k 列对应子载波 k（0–121）。
待预测集没有真实标签，不会把预测文件误称为测试集成绩。

## 复现边界

默认 `paper_appendix` 优先采用论文附录行为：dB域EESM、ISO逆频权重、
HGBR按训练R²择优、`var(d)` 分母融合。正文与附录不一致之处、未公开的平滑步骤、
噪声分母epsilon、参数不可辨识性均在对照文档明确说明。
随机种子固定42，模型以全部真实训练数据拟合。报告同时区分训练拟合和外层设备隔离结果，
不以论文给出的目标分数替换实际实验输出，也不承诺未公开实现的逐位复现。

旧的“建模思路与实施方案”文档保留为前期方案；实际已执行的第一题以本README及两份新文档为准。
论文与数据权利归原作者/赛事方，本工程未对第三方材料附加新的许可证。
