
import numpy as np

def naive_estimator(Y_labeled: np.ndarray) -> float:
    """
    计算 Naive Estimator（仅基于带标签数据的样本均值）

    参数说明
    ----------
    Y_labeled : np.ndarray
        一维数组（长度 n），表示带标签样本的真实观测值 Y_1, Y_2, ..., Y_n。
        这些样本来自总体分布 Y ~ N(θ*, σ²)，其中 θ* 是总体均值。

    返回值
    -------
    theta_nv : float
        Naive Estimator 的估计结果，即带标签样本的简单均值：
            θ̂_nv = mean(Y_labeled)
    """
  
    if not isinstance(Y_labeled, np.ndarray):
        Y_labeled = np.array(Y_labeled)

    # one dimension
    if Y_labeled.ndim != 1:
        raise ValueError("必须是一维数组")

    
    n = len(Y_labeled)#empty check
    if n == 0:
        raise ValueError("Y_labeled 不能为 empty ")


    # θ̂_nv = (1/n) Σ_{i=1}^{n} Y_i
    theta_nv = np.mean(Y_labeled)

    
    return theta_nv



if __name__ == "__main__":
    # 固定随机种子，保证可复现性
    np.random.seed(42)

    # 生成模拟 labeled 数据：Y_i ~ N(θ*=0.5, σ²=1)
    Y = np.random.normal(loc=0.5, scale=1.0, size=60)

    # 计算 Naive 估计结果
    theta_hat = naive_estimator(Y)

    # 打印结果
    print("Naive estimator 估计结果 =", theta_hat)