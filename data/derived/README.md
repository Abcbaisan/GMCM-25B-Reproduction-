# 第一题的真实派生数据

此目录保存从用户提供的6个原始 Excel 计算的逐载波 SINR，不是模拟数据。
原始 CSI Excel 位于 [Google Drive](https://drive.google.com/drive/folders/1e-4Tmme_lDIbw_XVc8GZiD-9bjxRxydC)，
文件级清单见 [manifest.json](../manifest.json)。原始大文件不直接进入 Git。

- `q1_train.npz`：`sinr` 数组 (N,122)，`y` 为原始 mcs 标签。
- `q1_valid.npz`：`sinr` 数组 (M,122)，无伪造标签。
- `q1_*_rows.csv`：与数组逐行对齐的原始文件、设备、Excel 行号、原始索引、噪声、时间与清洗状态。
- `q1_row_audit.csv.gz`：所有原始行的清洗审计，包含被过滤的训练行（若有）。
- `provenance.json`：以上五个文件的 SHA-256，以及6个原始 Excel 的 SHA-256与清洗审计。

实现采用论文第28页伪代码的信号功率缩放 0.001：
`SINR[k] = sum(abs(H[:,:,k])**2) * 0.001 / (10**((noise_floor-30)/10)/122)`。
论文未披露分母 epsilon，默认设为0：有效底噪范围内分母严格为正。
这是一项复现约定，不能断言它就是原作者的完整实现。
异常训练样本按原因记录后排除；异常官方待预测样本保留行位置。

从派生数据重跑：`python scripts/run_q1.py --source derived`。
从原始文件重算：按清单保存到 `data/raw/` 后运行 `python scripts/run_q1.py --source raw`。
两种路径使用同一数据顺序和模型流程。读取派生文件之前自动核验校验和。

原始 CSI 不包含在此派生文件中；需要修改物理预处理时，应重新从原始 Excel 运行。
数据来源和原论文归原作者/赛事方所有；此项目未为第三方数据附加新的许可证。
