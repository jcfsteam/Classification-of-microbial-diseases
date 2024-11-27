import numpy as np

def shuffle_features_independently(X):
    """
    对输入矩阵X的每一列(特征)独立进行随机打乱
    
    参数:
    X: numpy数组，形状为(n_samples, n_features)
    
    返回:
    shuffled_X: 每列独立打乱后的numpy数组
    """
    # 复制输入数组以避免修改原始数据
    X_shuffled = X.copy()
    
    # 获取样本数和特征数
    n_samples, n_features = X.shape
    
    # 对每一列独立进行打乱
    for col in range(n_features):
        shuffle_idx = np.random.permutation(n_samples)
        X_shuffled[:, col] = X[shuffle_idx, col]
        
    return X_shuffled

# 使用示例:
X_scaled = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
X_shuffled = shuffle_features_independently(X_scaled)

print(X_shuffled)