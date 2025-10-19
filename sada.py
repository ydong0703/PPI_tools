from typing import Tuple, Optional
import numpy as np


# ============================================================
# 1) 半监督真实场景接口（推荐）
#    输入：
#     - y_labeled:        (n,)          L 的真值 Y
#     - yhat_labeled:     (n, K)        L 的 K 个预测（每列一位预测器）
#     - yhat_unlabeled:   (N-n, K)      U 的 K 个预测
#    输出：
#     - theta_sada: 标量，SADA 均值估计
#     - omega_hat:  (K,)  学到的权重向量（对应命题2的 ω̂_opt）
# ============================================================
def sada_estimator(
    y_labeled: np.ndarray,
    yhat_labeled: np.ndarray,
    yhat_unlabeled: np.ndarray,
    *,
    ddof: int = 1,
    lambda_reg: float = 0.0,
    use_pinv: bool = False,
    eps: float = 1e-12,
    return_omega: bool = True,
) -> Tuple[float, Optional[np.ndarray], float]:
    """
    计算 SADA 均值估计（多预测器 K>=1），严格对应论文式(6)与命题2的均值特例。
    返回 (theta_sada, omega_hat, sd_sada)
    """
    # ---- 基本检查与整形 ----
    y_labeled = np.asarray(y_labeled).reshape(-1)
    yhat_labeled = np.asarray(yhat_labeled)
    yhat_unlabeled = np.asarray(yhat_unlabeled)

    if y_labeled.ndim != 1:
        raise ValueError("y_labeled 必须为一维向量 (n,)")
    if yhat_labeled.ndim != 2 or yhat_unlabeled.ndim != 2:
        raise ValueError("yhat_labeled / yhat_unlabeled 必须为二维矩阵 (·, K)")
    n, K = yhat_labeled.shape
    if n != len(y_labeled):
        raise ValueError("yhat_labeled 的行数必须与 y_labeled 长度一致")
    if yhat_unlabeled.shape[1] != K:
        raise ValueError("yhat_unlabeled 的列数必须等于 yhat_labeled 的列数 (相同的 K)")
    if n == 0 or yhat_unlabeled.shape[0] == 0:
        raise ValueError("L 与 U 都必须非空")

    # ȳ_L（naive 部分）
    ybar_L = float(np.mean(y_labeled))

    # ȳ̂_L, ȳ̂_U
    yhatbar_L = np.mean(yhat_labeled, axis=0)   # (K,)
    yhatbar_U = np.mean(yhat_unlabeled, axis=0) # (K,)
    delta_bar = (yhatbar_U - yhatbar_L)         # (K,)

    # 合并预测
    yhat_all = np.vstack([yhat_labeled, yhat_unlabeled])  # (N, K)
    N = yhat_all.shape[0]

    # 方差矩阵 Var(Ŷ)
    if N > 1:
        Sigma_yhat = np.cov(yhat_all, rowvar=False, ddof=ddof)
    else:
        Sigma_yhat = np.zeros((K, K), dtype=float)

    # 数值稳定性判断
    if not np.isfinite(Sigma_yhat).all():
        omega_hat = np.zeros(K, dtype=float)
    else:
        # 岭正则（可选）
        if lambda_reg > 0.0:
            Sigma_yhat = Sigma_yhat + float(lambda_reg) * np.eye(K)

        # Cov_L(Ŷ, Y)
        if n > 1:
            yL_center = y_labeled - y_labeled.mean()
            YhatL_center = yhat_labeled - yhat_labeled.mean(axis=0)
            cov_vec = (YhatL_center.T @ yL_center) / (n - ddof)
        else:
            cov_vec = np.zeros(K, dtype=float)

        # 安全退化与矩阵可逆性判断
        if K == 1:
            if float(np.abs(Sigma_yhat).max()) <= eps:
                omega_hat = np.zeros(1, dtype=float)
            else:
                inv_Sigma = 1.0 / float(Sigma_yhat)
                omega_hat = ((N - n) / N) * inv_Sigma * cov_vec
        else:
            cond = np.linalg.cond(Sigma_yhat) if np.all(Sigma_yhat) else np.inf
            if not np.isfinite(cond) or cond > 1.0 / max(eps, 1e-15):
                if use_pinv:
                    inv_Sigma = np.linalg.pinv(Sigma_yhat)
                    omega_hat = ((N - n) / N) * (inv_Sigma @ cov_vec)
                else:
                    omega_hat = np.zeros(K, dtype=float)
            else:
                inv_Sigma = np.linalg.inv(Sigma_yhat)
                omega_hat = ((N - n) / N) * (inv_Sigma @ cov_vec)

    # ---- 点估计 ----
    theta_sada = ybar_L + float(omega_hat @ delta_bar)

    # === 新增部分：计算标准误 sd_sada ===
    # Var(θ̂) = (1/n) Var(Y) + [N/(n(N-n))]*ωᵀVar(Ŷ)ω − (2/n)*ωᵀCov(Ŷ,Y)
    var_y_L = float(np.var(y_labeled, ddof=ddof)) if n > 1 else 0.0
    var_theta = (
        (var_y_L / n)
        + (N / (n * (N - n))) * float(omega_hat.T @ (Sigma_yhat @ omega_hat))
        - (2.0 / n) * float(omega_hat.T @ cov_vec)
    )
    sd_sada = float(np.sqrt(max(var_theta, 0.0)))
    # === 新增结束 ===

    if return_omega:
        return theta_sada, omega_hat, sd_sada
    else:
        return theta_sada, None, sd_sada


# ==========================================
# 2) 
#    - y_full:    (N,)      全部真值
#    - yhat_full: (N, K)    全部预测
#    - n_labeled: int       前 n_labeled 作为 L，其余 U
# ============================================================
def sada_estimator_from_full(
    y_full: np.ndarray,
    yhat_full: np.ndarray,
    n_labeled: int,
    *,
    ddof: int = 1,
    lambda_reg: float = 0.0,
    use_pinv: bool = False,
    eps: float = 1e-12,
    return_omega: bool = True,
) -> Tuple[float, Optional[np.ndarray]]:
  
    y_full = np.asarray(y_full).reshape(-1)
    yhat_full = np.asarray(yhat_full)
    if y_full.ndim != 1 or yhat_full.ndim != 2:
        raise ValueError("y_full 应为 (N,), yhat_full 应为 (N, K) 矩阵")
    N = len(y_full)
    if yhat_full.shape[0] != N:
        raise ValueError("y_full 与 yhat_full 行数 (N) 必须一致")
    if not (1 <= n_labeled < N):
        raise ValueError("n_labeled 必须在 [1, N-1] 内")

    L = slice(0, n_labeled)
    U = slice(n_labeled, N)

    return sada_estimator(
        y_labeled=y_full[L],
        yhat_labeled=yhat_full[L, :],
        yhat_unlabeled=yhat_full[U, :],
        ddof=ddof,
        lambda_reg=lambda_reg,
        use_pinv=use_pinv,
        eps=eps,
        return_omega=return_omega,
    )


# ============================== 自测 ==============================
if __name__ == "__main__":
    np.random.seed(0)

    # 仿真参数
    N, n, K = 200, 60, 2
    theta_star = 0.5
   
    Y = np.random.normal(theta_star, 1.0, size=N)
    eps1 = np.random.normal(0, 1, size=N)
    eps2 = np.random.normal(0, 1, size=N)
    gamma = 0.7
    Yhat1 = gamma * Y + (1 - gamma) * eps1
    Yhat2 = (1 - gamma) * Y + gamma * eps2
    Yhat = np.column_stack([Yhat1, Yhat2])  # (N, K)

    # 计算 SADA
    theta_sada, omega_hat, sd_sada = sada_estimator_from_full(
        y_full=Y, yhat_full=Yhat, n_labeled=n, lambda_reg=0.0, use_pinv=True
    )

    # 基线对照
    theta_nv = float(np.mean(Y[:n]))
    print(f"[自测] naive = {theta_nv:.6f}")
    print(f"[自测] SADA  = {theta_sada:.6f}")
    print(f"[自测] omega = {omega_hat}")
