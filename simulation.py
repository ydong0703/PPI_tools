#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Dependency：
  - naive.py     ：naive_estimator(y_labeled, ddof=...)
  - ppi_base.py  ：ppi_estimator(y_labeled, yhat_labeled, yhat_unlabeled) -> {"est","sd"}
  - ppi_safe.py  ：ppi_pp_estimator(y_labeled, yhat_labeled, yhat_unlabeled, ddof=..., eps=...) -> {"est","sd",...}
  - sada.py      ：sada_estimator(...) -> (theta_sada, omega_hat, sd_sada)


Usage：
  python simulation.py --N 200 --label_ratio 0.3 --J 1000 --seed 123 \
      --ddof 0 --lambda_reg 0.0 --use_pinv
"""

import argparse
import numpy as np
import pandas as pd

from naive import naive_estimator
from ppi_base import ppi_estimator
from ppi_safe import ppi_pp_estimator
from sada import sada_estimator


def gen_data(N: int, theta_star: float, gamma: float, rng: np.random.Generator):
    Y = rng.normal(loc=theta_star, scale=1.0, size=N)
    eps1 = rng.normal(0.0, 1.0, size=N)
    eps2 = rng.normal(0.0, 1.0, size=N)
    Yhat1 = gamma * Y + (1.0 - gamma) * eps1
    Yhat2 = (1.0 - gamma) * Y + gamma * eps2
    Yhat = np.column_stack([Yhat1, Yhat2])  # (N,2)
    return Y, Yhat


def split_labeled_unlabeled(Y, Yhat, label_ratio, rng):
    N = len(Y)
    n = int(np.floor(label_ratio * N))
    idx = np.arange(N)
    rng.shuffle(idx)
    L = np.sort(idx[:n])
    U = np.sort(idx[n:])
    y_L = Y[L]
    yhat_L = Yhat[L, :]
    yhat_U = Yhat[U, :]
    return y_L, yhat_L, yhat_U, L, U


def simulate_once(N, theta_star, gamma, label_ratio, ddof, lambda_reg, use_pinv, rng):
    """
    返回一个 list[dict]，每个 dict 对应一种方法的一条记录（含 est、sd、以及 ppi++/SADA 的 ω）
    """
    Y, Yhat = gen_data(N=N, theta_star=theta_star, gamma=gamma, rng=rng)
    y_L, yhat_L, yhat_U, L, U = split_labeled_unlabeled(Y, Yhat, label_ratio, rng)

    # naive
    res_naive = naive_estimator(y_L, ddof=ddof)  # {"est","sd"}
    naive_sd = float(res_naive["sd"])

    # PPI（分别对 yh1/yh2），基准 PPI 的 ω=1，这里不记录（保持 NaN）
    res_ppi_yh1 = ppi_estimator(y_labeled=y_L, yhat_labeled=yhat_L[:, 0], yhat_unlabeled=yhat_U[:, 0])
    res_ppi_yh2 = ppi_estimator(y_labeled=y_L, yhat_labeled=yhat_L[:, 1], yhat_unlabeled=yhat_U[:, 1])

    # PPI++（safe PPI，单预测器）
    res_ppipp_yh1 = ppi_pp_estimator(
        y_labeled=y_L, yhat_labeled=yhat_L[:, 0], yhat_unlabeled=yhat_U[:, 0],
        ddof=ddof, eps=1e-12
    )  # {"est","sd","omega"}
    res_ppipp_yh2 = ppi_pp_estimator(
        y_labeled=y_L, yhat_labeled=yhat_L[:, 1], yhat_unlabeled=yhat_U[:, 1],
        ddof=ddof, eps=1e-12
    )

    # SADA（K=2）
    theta_sada, omega_vec, sd_sada = sada_estimator(
        y_labeled=y_L,
        yhat_labeled=yhat_L,
        yhat_unlabeled=yhat_U,
        ddof=ddof,
        lambda_reg=lambda_reg,
        use_pinv=use_pinv,
        eps=1e-12,
        return_omega=True,
    )  # omega_vec 形状 (2,)

    rows = []

    def add_row(method, group, est, sd, omega1=np.nan, omega2=np.nan):
        sd_ratio = (sd / naive_sd) if naive_sd > 0 else np.nan
        rows.append({
            "method": method, "group": group,
            "est": float(est), "sd": float(sd), "sd.ratio": float(sd_ratio),
            "omega1": float(omega1) if np.isfinite(omega1) else np.nan,
            "omega2": float(omega2) if np.isfinite(omega2) else np.nan,
        })

    # 填充各方法
    add_row("naive", "common", res_naive["est"], res_naive["sd"])
    add_row("ppi",   "yh1",    res_ppi_yh1["est"], res_ppi_yh1["sd"])  # ω=1，不记录
    add_row("ppi",   "yh2",    res_ppi_yh2["est"], res_ppi_yh2["sd"])
    add_row("ppi++", "yh1",    res_ppipp_yh1["est"], res_ppipp_yh1["sd"], omega1=res_ppipp_yh1["omega"])
    add_row("ppi++", "yh2",    res_ppipp_yh2["est"], res_ppipp_yh2["sd"], omega1=res_ppipp_yh2["omega"])

    # SADA 的 ω 是长度 K 的向量（这里 K=2）
    w1 = omega_vec[0] if omega_vec is not None and len(omega_vec) > 0 else np.nan
    w2 = omega_vec[1] if omega_vec is not None and len(omega_vec) > 1 else np.nan
    add_row("SADA", "common", theta_sada, sd_sada, omega1=w1, omega2=w2)

    return rows


def summarize_by_gamma(df_rep: pd.DataFrame, theta_star: float) -> pd.DataFrame:
    """
   
      bias = mean(est) - true
      se   = sd(est)                     
      coverage = mean( est ± 1.96*sd 覆盖真值 )
      sd.mean = mean(sd)
      rmse = sqrt(bias^2 + se^2)
      ARE = rmse / rmse(naive, 同 γ)
    """
    parts = []
    for (g, method, group), sub in df_rep.groupby(["gamma", "method", "group"], dropna=False):
        est = sub["est"].to_numpy()
        sd  = sub["sd"].to_numpy()
        bias = float(est.mean() - theta_star)
        se = float(est.std(ddof=1)) if len(est) > 1 else 0.0
        rmse = float(np.sqrt(bias**2 + se**2))
        # 覆盖率（容错 NaN）
        cover = np.mean(((est - 1.96 * sd) <= theta_star) & ((est + 1.96 * sd) >= theta_star)) if len(est) > 0 else np.nan
        sd_mean = float(np.nanmean(sd)) if len(sd) > 0 else np.nan
        parts.append({
            "gamma": g, "method": method, "group": group,
            "bias": bias, "se": se, "coverage": cover, "sd": sd_mean, "rmse": rmse
        })
    out = pd.DataFrame(parts)

   
    def add_are(grp):
        rmse_naive = float(grp.loc[grp["method"] == "naive", "rmse"].values[0])
        grp = grp.copy()
        grp["ARE"] = grp["rmse"] / rmse_naive if rmse_naive > 0 else np.nan
        return grp

    out = out.groupby("gamma", as_index=False, group_keys=False).apply(add_are).reset_index(drop=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=200, help="总样本数 N")
    parser.add_argument("--J", type=int, default=1000, help="重复次数（Monte Carlo replications）")
    parser.add_argument("--label_ratio", type=float, default=0.3, help="labeled 比例（默认 0.3）")
    parser.add_argument("--theta", type=float, default=0.5, help="真均值 θ*")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--ddof", type=int, default=0, choices=[0, 1], help="方差/协方差的 ddof（0=总体; 1=样本无偏）")
    parser.add_argument("--lambda_reg", type=float, default=0.0, help="SADA 中对 Var(Ŷ) 的岭正则 λ")
    parser.add_argument("--use_pinv", action="store_true", help="SADA 遇到病态协方差矩阵时使用伪逆")
    parser.add_argument("--out_prefix", type=str, default="results", help="输出文件名前缀")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    gammas = np.linspace(0.0, 1.0, 11)  # 0,0.1,...,1.0

    rows = []
    for g in gammas:
        for rep in range(1, args.J + 1):
            sub_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
            # 单次返回多方法的记录
            recs = simulate_once(
                N=args.N,
                theta_star=args.theta,
                gamma=g,
                label_ratio=args.label_ratio,
                ddof=args.ddof,
                lambda_reg=args.lambda_reg,
                use_pinv=args.use_pinv,
                rng=sub_rng,
            )
            # 附加实验信息
            for r in recs:
                r.update({"rep": rep, "gamma": g})
            rows.extend(recs)

    df_rep = pd.DataFrame(rows).sort_values(["gamma", "method", "group", "rep"]).reset_index(drop=True)

    # 汇总
    df_sum = summarize_by_gamma(df_rep, theta_star=args.theta)

    
    rep_path = f"{args.out_prefix}_gamma_rep.csv"        # R: 1simu_gamma_rep.csv
    sum_path = f"{args.out_prefix}_gamma_summary.csv"    # R: 1simu_gamma_summary.csv
    df_rep.to_csv(rep_path, index=False)
    df_sum.to_csv(sum_path, index=False)

    print(f"[OK] 保存逐次结果到: {rep_path}")
    print(f"[OK] 保存汇总结果到: {sum_path}")
    print("\n汇总预览：")
    print(df_sum.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
