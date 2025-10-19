"""
ppi_base.py


------------


  Naive:
    θ̂_nv = (1/n) * Σ_{i=1}^n Y_i

  PPI（单预测器）:
    θ̂_ppi = θ̂_nv + [ mean(Ŷ_U) - mean(Ŷ_L) ]
        


"""

from typing import Tuple
import numpy as np



# 1) 主接口：

#    - y_labeled:       有标签样本的 Y（长度 n）
#    - yhat_labeled:    同一批 L 的预测 Ŷ（长度 n）
#    - yhat_unlabeled:  未标注样本 U 的预测 Ŷ（长度 N-n）

def ppi_estimator(
    y_labeled: np.ndarray,
    yhat_labeled: np.ndarray,
    yhat_unlabeled: np.ndarray,
) -> dict:   # === 修改：返回 dict 而非 float ===
    """

    ----------
    y_labeled : np.ndarray
        一维数组，长度 n。带标签样本的真实观测值 Y_L。
    yhat_labeled : np.ndarray
        一维数组，长度 n。对应 L 的预测值 Ŷ_L（来自某个模型）。
    yhat_unlabeled : np.ndarray
        一维数组，长度 N-n。对应 U 的预测值 Ŷ_U（来自同一个模型）。

    """
    # ---- 输入检查：转为 numpy 数组 ----
    if not isinstance(y_labeled, np.ndarray):
        y_labeled = np.array(y_labeled)
    if not isinstance(yhat_labeled, np.ndarray):
        yhat_labeled = np.array(yhat_labeled)
    if not isinstance(yhat_unlabeled, np.ndarray):
        yhat_unlabeled = np.array(yhat_unlabeled)

    # ---- 维度与长度检查 ----
    if y_labeled.ndim != 1:
        raise ValueError("y_labeled 必须为一维数组")
    if yhat_labeled.ndim != 1:
        raise ValueError("yhat_labeled 必须为一维数组")
    if yhat_unlabeled.ndim != 1:
        raise ValueError("yhat_unlabeled 必须为一维数组")

    n = len(y_labeled)
    if n == 0:
        raise ValueError("y_labeled 不能为空")
    if len(yhat_labeled) != n:
        raise ValueError("yhat_labeled 的长度必须与 y_labeled 相同（同一批 L）")
    if len(yhat_unlabeled) == 0:
        raise ValueError("yhat_unlabeled 不能为空（需要 U 的预测）")

    # ---- 基础统计量 ----
    mean_y_L    = float(np.mean(y_labeled))        # mean(Y_L)
    mean_yhat_L = float(np.mean(yhat_labeled))     # mean(Ŷ_L)
    mean_yhat_U = float(np.mean(yhat_unlabeled))   # mean(Ŷ_U)

    #ppi
    theta_ppi = mean_y_L + (mean_yhat_U - mean_yhat_L)

   
    #  Var(θ̂_ppi) = Var(Ŷ)/(N-n) + Var(Y - Ŷ)/n
    yhat_all = np.concatenate([yhat_labeled, yhat_unlabeled], axis=0)
    N = len(yhat_all)
    var_yhat_all = np.var(yhat_all, ddof=0)       # 总体方差 Var(Ŷ)
    resid_L = y_labeled - yhat_labeled
    var_resid_L = np.var(resid_L, ddof=0)         # Var(Y - Ŷ)
    var_theta = var_yhat_all / (N - n) + var_resid_L / n
    sd_ppi = np.sqrt(var_theta)


    # 返回与 naive 一致的字典形式
    return {"est": float(theta_ppi), "sd": float(sd_ppi)}



# 2) 便捷接口：
#    - y_full:  N 长度的真值向量（仿真中可得）
#    - yhat_full: N 长度的预测向量（同一模型）
#    - n_labeled: 前 n_labeled 个索引视为 L，其余为 U

def ppi_estimator_from_full(
    y_full: np.ndarray,
    yhat_full: np.ndarray,
    n_labeled: int,
) -> float:
    """

    y_full : np.ndarray
        一维数组，长度 N。完整的真值向量（仿真可见，真实半监督场景不可见）。
    yhat_full : np.ndarray
        一维数组，长度 N。完整的单模型预测向量。
    n_labeled : int
        视为有标签样本的数量 n（取前 n 个为 L，剩余为 U）。

    """
    # ---- 输入检查 ----
    if not isinstance(y_full, np.ndarray):
        y_full = np.array(y_full)
    if not isinstance(yhat_full, np.ndarray):
        yhat_full = np.array(yhat_full)

    if y_full.ndim != 1 or yhat_full.ndim != 1:
        raise ValueError("y_full 与 yhat_full 必须为一维数组")
    if len(y_full) != len(yhat_full):
        raise ValueError("y_full 与 yhat_full 长度必须一致")
    N = len(y_full)
    if N == 0:
        raise ValueError("输入向量不能为空")
    if not (1 <= n_labeled < N):
        raise ValueError("n_labeled 必须在 [1, N-1] 区间内")

    # splitting
    L = slice(0, n_labeled)
    U = slice(n_labeled, N)

    y_labeled      = y_full[L]
    yhat_labeled   = yhat_full[L]
    yhat_unlabeled = yhat_full[U]

    # 用前面的
    return ppi_estimator(y_labeled, yhat_labeled, yhat_unlabeled)

##AI测试 不要调用。
# ================== 自测（仿真） ==================
if __name__ == "__main__":
    np.random.seed(0)

    # ---- 仿真参数（与论文 5.1 节一致的风格）----
    N = 200          # 总样本
    n = 60           # 有标签样本
    theta_star = 0.5 # 真均值
    gamma = 0.7      # 控制预测质量的参数（越接近 1，Yhat1 越接近 Y）

    # ---- 生成真值 Y ~ N(theta*, 1) ----
    Y = np.random.normal(loc=theta_star, scale=1.0, size=N)

    # ---- 生成单预测器的预测（示例：仿论文构造）----
    eps = np.random.normal(0.0, 1.0, size=N)
    Yhat = gamma * Y + (1 - gamma) * eps  # 单个预测器的预测 Ŷ

    # ---- 计算 PPI（两种接口等价）----
    theta_ppi_1 = ppi_estimator_from_full(Y, Yhat, n_labeled=n)

    theta_ppi_2 = ppi_estimator(
        y_labeled=Y[:n],
        yhat_labeled=Yhat[:n],
        yhat_unlabeled=Yhat[n:]
    )

    # ---- Naive 作为对照（只用 L 的均值）----
    theta_nv = float(np.mean(Y[:n]))

    print(f"[自测] Naive = {theta_nv:.6f}")
    print(f"[自测] PPI (from_full) = {theta_ppi_1:.6f}")
    print(f"[自测] PPI (semi-supervised API) = {theta_ppi_2:.6f}")
