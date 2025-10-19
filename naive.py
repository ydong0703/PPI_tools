import numpy as np

def naive_estimator(Y_labeled: np.ndarray, ddof: int = 0) -> dict:
    """

    θ̂_nv = mean(Y_labeled)
    Var(θ̂_nv) = Var(Y) / n

    result : dict
        {
          "est": θ̂_nv,   # 点估计
          "sd":  ŝ_nv,   # 标准误（sqrt(Var(Y)/n)）
        }
    """
    if not isinstance(Y_labeled, np.ndarray):
        Y_labeled = np.array(Y_labeled)

    if Y_labeled.ndim != 1:
        raise ValueError("Y_labeled 必须是一维数组")

    n = len(Y_labeled)
    if n == 0:
        raise ValueError("Y_labeled 不能为空")

    # 均值估计
    theta_nv = np.mean(Y_labeled)

    # 样本方差
    var_Y = np.var(Y_labeled, ddof=ddof)

    # 标准误（论文中 Var(θ̂)=Var(Y)/n）
    sd_nv = np.sqrt(var_Y / n)

    return {"est": float(theta_nv), "sd": float(sd_nv)}


# ===================== 自测 =====================
if __name__ == "__main__":
    np.random.seed(42)
    Y = np.random.normal(loc=0.5, scale=1.0, size=60)
    res = naive_estimator(Y, ddof=0)

    print("Naive estimator 估计结果 θ̂ =", res["est"])
    print("标准误 sd(θ̂) =", res["sd"])
