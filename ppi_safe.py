"""
ppi_pp.py
1) 
   ω̂_opt = ((N - n) / N) * Cov_L(Ŷ, Y) / Var_All(Ŷ)
2) safe PPI (PPI++):
   θ̂_safe = mean(Y_L) + ω̂_opt * ( mean(Ŷ_U) - mean(Ŷ_L) )


------
- naive:   θ̂_nv   = mean(Y_L)


- PPI:     θ̂_ppi  = mean(Y_L) + ( mean(Ŷ_U) - mean(Ŷ_L) )           # ω=1

- PPI++:   θ̂_safe = mean(Y_L) + ω̂_opt * ( mean(Ŷ_U) - mean(Ŷ_L) )   # ω=ω̂_opt

"""

from typing import Tuple
import numpy as np



# 1) Base input
#    
#      - y_labeled:       Y_L   (长度 n)
#      - yhat_labeled:    Ŷ_L   (长度 n)
#      - yhat_unlabeled:  Ŷ_U   (长度 N-n)

def ppi_pp_estimator(
    y_labeled: np.ndarray,
    yhat_labeled: np.ndarray,
    yhat_unlabeled: np.ndarray,
    eps: float = 1e-12,#只是为了防止除以0
) -> float:


    # 数制转换，copy from previous code.
    if not isinstance(y_labeled, np.ndarray):
        y_labeled = np.array(y_labeled)
    if not isinstance(yhat_labeled, np.ndarray):
        yhat_labeled = np.array(yhat_labeled)
    if not isinstance(yhat_unlabeled, np.ndarray):
        yhat_unlabeled = np.array(yhat_unlabeled)

    if y_labeled.ndim != 1 or yhat_labeled.ndim != 1 or yhat_unlabeled.ndim != 1:
        raise ValueError("所有输入必须为一维数组 (1D)")
    n = len(y_labeled)
    if n == 0:
        raise ValueError("y_labeled 不能为空")
    if len(yhat_labeled) != n:
        raise ValueError("yhat_labeled 长度必须与 y_labeled 相同")
    if len(yhat_unlabeled) == 0:
        raise ValueError("yhat_unlabeled 不能为空")

    # All means
    mean_y_L     = float(np.mean(y_labeled))          # mean(Y_L)
    mean_yhat_L  = float(np.mean(yhat_labeled))       # mean(Ŷ_L)
    mean_yhat_U  = float(np.mean(yhat_unlabeled))     # mean(Ŷ_U)

 
    # Var_All(Ŷ)：用 L∪U 全部 Ŷ
    #Usage Mark: np.concatenate((array1, array2, ...), axis=0)用于拼接
    yhat_all = np.concatenate([yhat_labeled, yhat_unlabeled], axis=0)
    #Usage Mark: Σ差平方/（N - ddof）
    #Question Mark: 我们用的是0还是1？
    var_yhat_all = float(np.var(yhat_all, ddof=1)) if len(yhat_all) > 1 else 0.0

    # Cov_L(Ŷ, Y)：仅用 L
    if n > 1:
        cov_yhat_y_L = float(np.cov(yhat_labeled, y_labeled, ddof=1)[0, 1])
    else:
        cov_yhat_y_L = 0.0  # safety check 样本过小无法稳定估计，退化为 0

    # 计算 ω̂_opt
    N = len(yhat_all)
    if var_yhat_all <= eps:
        # ➗0检验
        omega_hat = 0.0
    else:
        omega_hat = ((N - n) / N) * (cov_yhat_y_L / var_yhat_all)

    # ---- PPI++ 估计值 ----
    theta_safe = mean_y_L + omega_hat * (mean_yhat_U - mean_yhat_L)
    return theta_safe


# 2) 仿真实验便捷接口

def ppi_pp_estimator_from_full(
    y_full: np.ndarray,
    yhat_full: np.ndarray,
    n_labeled: int,
    eps: float = 1e-12,
) -> float:
    """
    y_full : np.ndarray
        一维数组（长度 N）。完整真值。
    yhat_full : np.ndarray
        一维数组（长度 N）。完整预测（同一模型的预测）。
    n_labeled : int
        前 n_labeled 个视为 L，其余为 U。
    eps : float
        数值稳定项（见 ppi_pp_estimator）。

    """
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
        raise ValueError("n_labeled 必须在 [1, N-1] 内")

    L = slice(0, n_labeled)
    U = slice(n_labeled, N)

    return ppi_pp_estimator(
        y_labeled=y_full[L],
        yhat_labeled=yhat_full[L],
        yhat_unlabeled=yhat_full[U],
        eps=eps,
    )

if __name__ == "__main__":
    np.random.seed(42)

 
    N = 200
    n = 60
    theta_star = 0.5
    gamma = 0.3  # 预测质量：越接近 1，Ŷ 越接近 Y；越接近 0，Ŷ 越接近噪声

   
    Y = np.random.normal(loc=theta_star, scale=1.0, size=N)
    eps1 = np.random.normal(0.0, 1.0, size=N)
    Yhat = gamma * Y + (1 - gamma) * eps1  

    
    theta_nv   = float(np.mean(Y[:n]))
    theta_pp   = theta_nv + (np.mean(Yhat[n:]) - np.mean(Yhat[:n]))      # 普通 PPI（ω=1）
    theta_safe = ppi_pp_estimator_from_full(Y, Yhat, n_labeled=n)        # PPI++

    print(f"[自测] naive   = {theta_nv:.6f}")
    print(f"[自测] PPI     = {theta_pp:.6f}")
    print(f"[自测] PPI++   = {theta_safe:.6f}")
