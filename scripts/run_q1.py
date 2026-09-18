"""Run the complete question-1 experiment using raw Excel or verified derived data."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from gmcm25b.data import Dataset, load_dataset
from gmcm25b.models import ModelConfig, fit_model
from gmcm25b.diagnostics import audit_overlap

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def json_write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
        default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
        allow_nan=False) + "\n", encoding="utf-8")

def scores(y, pred):
    return {"n": len(y), "mse": float(mean_squared_error(y,pred)),
            "rmse": float(np.sqrt(mean_squared_error(y,pred))),
            "mae": float(mean_absolute_error(y,pred)),
            "r2": float(r2_score(y,pred))}

def save_derived(train, valid, audit):
    folder = ROOT / "data/derived"
    folder.mkdir(parents=True,exist_ok=True)
    entries = []
    row_source = Path(audit["row_audit_csv"])
    if not row_source.is_absolute():
        row_source = ROOT / row_source
    row_target = folder / "q1_row_audit.csv.gz"
    if row_source != row_target:
        pd.read_csv(row_source,keep_default_na=False,dtype=str).to_csv(
            row_target,index=False,compression={"method":"gzip","mtime":0})
    audit["row_audit_csv"] = row_target.relative_to(ROOT).as_posix()
    entries.append({"path":row_target.name,"sha256":sha256(row_target),
                    "size_bytes":row_target.stat().st_size})
    for name, d in [("train",train),("valid",valid)]:
        path = folder / f"q1_{name}.npz"
        arrays = {"sinr":d.sinr}
        if d.y is not None:
            arrays["y"] = d.y
        np.savez_compressed(path, **arrays)
        meta = folder / f"q1_{name}_rows.csv"
        d.meta.to_csv(meta,index=False,encoding="utf-8")
        entries.extend({"path":p.name,"sha256":sha256(p),"size_bytes":p.stat().st_size}
                       for p in [path,meta])
    json_write(folder / "provenance.json",{
        "description":"Real supplied Excel data; derived linear per-subcarrier SINR, no synthetic rows.",
        "signal_scale":0.001,"noise_epsilon_w":0.0,"subcarriers":122,
        "formula":"sum(abs(H)**2,rx,tx)*0.001/(10**((noise_floor-30)/10)/122)",
        "files":entries,"raw_data_audit":audit})
    print("Derived data exported and checksummed.",flush=True)

def read_derived():
    folder = ROOT / "data/derived"
    provenance = json.loads((folder/"provenance.json").read_text(encoding="utf-8"))
    for f in provenance["files"]:
        p=folder/f["path"]
        if p.stat().st_size != f["size_bytes"] or sha256(p)!=f["sha256"]:
            raise ValueError(f"Derived file checksum mismatch: {p.name}")
    datasets=[]
    for split in ["train","valid"]:
        with np.load(folder/f"q1_{split}.npz",allow_pickle=False) as a:
            x=a["sinr"].copy()
            y=a["y"].copy() if "y" in a else None
        meta=pd.read_csv(folder/f"q1_{split}_rows.csv",keep_default_na=False,
                         dtype={"source_index":str,"sample_id":str,"source_file":str,"device":str})
        if x.shape != (len(meta),122) or (y is not None and len(y)!=len(meta)):
            raise ValueError("Derived metadata/array shape mismatch")
        datasets.append(Dataset(sinr=x,meta=meta,y=y))
    return *datasets,provenance["raw_data_audit"]

def fit(d, indices=None):
    ix=np.arange(len(d.meta)) if indices is None else indices
    return fit_model(d.sinr[ix],d.y[ix],d.meta.device.to_numpy()[ix],
        d.meta.noise_floor.to_numpy(dtype=float)[ix],
        d.meta.source_file.map(lambda f: Path(f).name).to_numpy()[ix],
        config=ModelConfig(),progress=lambda message: print(message,flush=True))

def predict(model,d,indices=None):
    ix=np.arange(len(d.meta)) if indices is None else indices
    return model.predict(d.sinr[ix],d.meta.noise_floor.to_numpy(dtype=float)[ix],
                         d.meta.source_file.map(lambda f: Path(f).name).to_numpy()[ix])

def device_audit(d,out):
    """Outer held-out device never participates in inner parameter/model fitting."""
    groups=d.meta.device.to_numpy()
    count=len(np.unique(groups))
    if count < 3:
        raise ValueError("Nested device audit requires at least three devices")
    oof=d.meta[["sample_id","source_file","excel_row","device"]].copy()
    oof["truth"]=d.y
    records=[]
    for fold,(tr,va) in enumerate(GroupKFold(count).split(d.sinr,d.y,groups),1):
        print(f"Outer device audit {fold}/{count}: training {len(tr)}, held out {len(va)}",flush=True)
        model,metrics,search=fit(d,tr)
        pred=predict(model,d,va)
        for c in pred:
            oof.loc[va,c]=pred[c].to_numpy()
        oof.loc[va,"fold"]=fold
        record={"fold":fold,"held_out_devices":sorted(np.unique(groups[va]).tolist()),
                "n_train":len(tr),"n_test":len(va),"alpha":model.alpha,
                "beta":model.beta,"train_metrics":metrics,
                "held_out":{name:scores(d.y[va],pred[name]) for name in ["iso","hgbr","blend"]}}
        records.append(record)
        json_write(out/f"audit/fold_{fold}.json",record)
        search.to_csv(out/f"audit/fold_{fold}_parameter_search.csv",index=False)
        oof.to_csv(out/"device_oof_predictions.csv",index=False)
    overall={name:scores(d.y,oof[name]) for name in ["iso","hgbr","blend"]}
    json_write(out/"device_holdout_metrics.json",{
        "protocol":"Nested leave-one-device-out; EESM and HGBR refitted/selected exclusively on outer training devices.",
        "notes":"Inner estimator follows the paper appendix, including in-sample HGBR selection/blending; only outer held-out scores assess device transfer. Three devices limit precision.",
        "overall":overall,"folds":records})
    return overall,oof

def figures(y,train_pred,oof,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (out/"figures").mkdir(exist_ok=True)
    fig,axs=plt.subplots(1,2,figsize=(11,4.4),layout="constrained")
    sets=[("Training fit",train_pred["blend"].to_numpy()),
          ("Nested device holdout",oof["blend"].to_numpy() if oof is not None else None)]
    for ax,(title,p) in zip(axs,sets):
        if p is None:
            ax.text(.5,.5,"Not run",ha="center");continue
        ax.hexbin(y,p,gridsize=45,mincnt=1,cmap="viridis")
        bounds=[min(y.min(),p.min()),max(y.max(),p.max())]
        ax.plot(bounds,bounds,"r--",lw=1)
        ax.set(title=title,xlabel="Observed mcs label",ylabel="Predicted mcs label")
    fig.savefig(out/"figures/fit_vs_device_holdout.png",dpi=160)
    plt.close(fig)

def result_report(report, predictions, out):
    training=report["training_scores"]["blend"]
    paper={"n":20369,"mse":1652.976,"rmse":40.657,"mae":30.952,"r2":0.828}
    rows="\n".join(f"| {k} | {paper[k]} | {training[k]:.6g} |" for k in paper)
    audit=report.get("nested_device_holdout")
    atext=("本次未执行设备隔离验证，训练拟合指标不能作为泛化指标。" if audit is None else
           f"全部外层测试样本合并：融合模型 R²={audit['blend']['r2']:.6f}，"
           f"RMSE={audit['blend']['rmse']:.6f}，MAE={audit['blend']['mae']:.6f}。"
           "每一外折都仅使用其余设备重新搜索 EESM 参数、选择 HGBR 和估计融合权重。"
           "这是跨设备迁移检验；只有三个设备，不能代表所有部署环境。")
    counts=predictions.groupby("device",sort=True).agg(rows=("sample_id","size"),
        valid_predictions=("label",lambda s:int(s.notna().sum())))
    table="| 设备 | 待预测行数 | 有效预测 |\n|---|---:|---:|\n"+"\n".join(
        f"| {device} | {int(row.rows)} | {int(row.valid_predictions)} |"
        for device,row in counts.iterrows())
    weight=report["model_metrics"]["blend_weight_hgbr_appendix_variance"]
    weight_note=("本次权重达到1，融合结果等于HGBR输出，不能声称融合额外提升了本次成绩。"
                 if weight==1 else "融合按附录方差分母公式计算。")
    overlap=json.loads((out/"data_overlap_audit.json").read_text(encoding="utf-8"))
    text=f"""# 问题一：实际复现结果

本报告由 `scripts/run_q1.py` 根据真实运行结果生成。仅覆盖问题一。

## 论文结果与本次结果

原文第30页表5-3明确描述训练集拟合指标。下表同样比较训练拟合，不视为独立测试精度。
数据中的 `mcs` 是原始数值标签，程序不将其擅自重编码为整数，也不宣称其单位为 Mbps 或 dB。

| 指标 | 论文表5-3 | 本次训练拟合 |
|---|---:|---:|
{rows}

论文参数 α=2.2、β=0.29648733946825756；本次参数 α={report['parameters']['alpha']:.12g}、
β={report['parameters']['beta']:.12g}。本次参数来自运行搜索，未用论文成绩反推或替换。
原论文主程序、平滑实现、随机状态和噪声分母 ε 未完整公开，因此本工程为依据正文及附录的可执行重建，不承诺逐位或指标完全一致。

HGBR的融合权重为 {weight:.6g}。{weight_note}

## 独立设备隔离验证

{atext}

详见 [指标文件](../outputs/q1/device_holdout_metrics.json)、
[逐样本折外预测](../outputs/q1/device_oof_predictions.csv)。
这部分另行评估论文算法，不能与论文训练表5-3直接比较为优劣。

## 官方无标签 B 集预测

{table}

B 集没有真实标签，因此没有对 B 集计算准确率、R² 或 RMSE。
[总预测表](../outputs/q1/valid_predictions.csv)同时给出每条样本的 EESM 等效量、
ISO、HGBR、连续融合值以及吸附到训练标签集合后的 `label`。
`required/` 下每个 CSV 采用原始索引及 `mcs` 两列。异常行保留并标识，不会静默移位。

## 数据与复现实验

原始6个 Excel 的字节数、SHA-256、清洗统计见
[数据审计](../outputs/q1/data_audit.json)。
[派生数据说明](../data/derived/README.md)记录逐载波 SINR 与源行的对应关系，
可直接重跑模型；从原始 CSI 重建需下载 Drive 中相同文件。
所有版本、种子、耗时与模型细节见 [metrics.json](../outputs/q1/metrics.json)。
默认种子42；源码、派生数据和本次结果已纳入版本管理。

## 数据结构重复检查

训练与待预测集各有20,369个精确唯一的SINR行，跨集合完全相同的SINR行为
{overlap['shared_exact_sinr_rows']}。但去除噪声影响、将每个子载波的天线总能量保留11位有效数字后，
训练/待预测集分别只有 {overlap['splits']['train']['channel_energy_unique_rows_11_significant_digits']} /
{overlap['splits']['valid']['channel_energy_unique_rows_11_significant_digits']} 种能量向量，
待预测集有 {overlap['valid_rows_with_rounded_energy_seen_in_train']} 条记录的能量结构在训练集出现。
因此不能仅凭行数或精确SINR无重叠就认为所有样本独立。
能量求和丢失复数相位和空间结构，此检查不证明原始CSI相同，也不证明标签泄漏。
计算方法及完整统计见 [数据重叠审计](../outputs/q1/data_overlap_audit.json)。

![训练拟合与设备隔离验证](../outputs/q1/figures/fit_vs_device_holdout.png)

论文公式、附录代码与本工程的详细对应见
[问题一-论文代码对照](问题一-论文代码对照.md)。
"""
    (ROOT/"docs/问题一-复现结果.md").write_text(text,encoding="utf-8")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",choices=["auto","raw","derived"],default="auto")
    p.add_argument("--prepare-only",action="store_true")
    p.add_argument("--skip-audit",action="store_true",help="Skip expensive nested device validation; no generalization claim.")
    p.add_argument("--threads",type=int,default=4)
    args=p.parse_args()
    start=time.perf_counter()
    out=ROOT/"outputs/q1";out.mkdir(parents=True,exist_ok=True)
    use_raw=args.source=="raw" or (args.source=="auto" and not (ROOT/"data/derived/provenance.json").exists())
    if use_raw:
        print("Parsing and validating raw Excel data...",flush=True)
        train,valid,audit=load_dataset(ROOT/"data/raw",ROOT/"data/manifest.json",ROOT/"data/processed")
        save_derived(train,valid,audit)
    else:
        print("Verifying and reading derived real data...",flush=True)
        train,valid,audit=read_derived()
    json_write(out/"data_audit.json",audit)
    json_write(out/"data_overlap_audit.json",audit_overlap(train,valid))
    if args.prepare_only:
        print(f"Prepared {len(train.meta)} training and {len(valid.meta)} validation rows.",flush=True)
        return
    with threadpool_limits(limits=args.threads):
        print(f"Fitting paper appendix model on {len(train.meta)} samples...",flush=True)
        model,metrics,search=fit(train)
        search.to_csv(out/"parameter_search.csv",index=False)
        joblib.dump(model,out/"q1_model.joblib")
        trpred=predict(model,train)
        pd.concat([train.meta.reset_index(drop=True),pd.Series(train.y,name="truth"),trpred.reset_index(drop=True)],axis=1).to_csv(out/"training_predictions.csv",index=False)
        good=valid.meta.status.eq("ok").to_numpy()
        vp=pd.DataFrame(index=np.arange(len(valid.meta)),columns=["eff_db","iso","hgbr","blend","label","blend_mse_audit"],dtype=float)
        vp.loc[good,:]=predict(model,valid,np.flatnonzero(good)).to_numpy()
        predictions=pd.concat([valid.meta.reset_index(drop=True),vp],axis=1)
        predictions.to_csv(out/"valid_predictions.csv",index=False)
        (out/"required").mkdir(exist_ok=True)
        for (device,file),sub in predictions.groupby(["device","source_file"],sort=True):
            required=pd.DataFrame({"index":sub.source_index,"mcs":sub.label})
            required.to_csv(out/"required"/f"{device}_{Path(str(file)).stem}_required.csv",index=False)
        overall,oof=(None,None) if args.skip_audit else device_audit(train,out)
        report={"scope":"question_1_only","protocol":"paper_appendix",
          "parameters":{"alpha":model.alpha,"beta":model.beta},
          "model_metrics":metrics,
          "training_scores":{name:scores(train.y,trpred[name]) for name in ["iso","hgbr","blend"]},
          "nested_device_holdout":overall,"n_valid":len(valid.meta),
          "n_valid_predictions":int(good.sum()),"random_seed":42,
          "elapsed_seconds":time.perf_counter()-start,"python":sys.version,
          "platform":platform.platform(),"packages":{name:importlib.metadata.version(name)
            for name in ["numpy","scipy","pandas","openpyxl","scikit-learn","matplotlib","joblib","threadpoolctl"]}}
        json_write(out/"metrics.json",report)
        figures(train.y,trpred,oof,out)
        result_report(report,predictions,out)
    print(json.dumps(report["training_scores"],indent=2),flush=True)
    print(f"Completed; outputs: {out}",flush=True)

if __name__=="__main__":
    main()
