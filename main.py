# main.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from neural_network import augment_with_vae
from src.model_comparison import ModelTrainer
from src.uncertainty_model_comparison import UncertaintyGuidedTrainer
from src.data_processing import DataProcessor
from config.model_config import ModelConfig
import os
import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# 导入自定义模块
# from torch_geometric.loader import DataLoader  # Changed this import

def setup_directories():
    """创建必要的目录结构"""
    dirs = ['outputs', 'outputs/models', 'outputs/plots', 'outputs/results']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def plot_training_results(results: dict):
    """绘制并保存训练结果可视化"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    model_names = list(results.keys())

    # 处理'N/A'值
    def safe_float(value):
        if isinstance(value, (int, float)):
            return float(value)
        if value == 'N/A' or value is None:
            return 0.0
        try:
            return float(value)
        except:
            return 0.0

    metric_values = {
        metric: [safe_float(results[model].get(metric, 0))
                 for model in model_names]
        for metric in metrics
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        if idx >= len(axes):
            break
        values = metric_values[metric]
        max_value = max(values)
        bar_colors = ['skyblue' if v < max_value else 'orange' for v in values]

        axes[idx].barh(model_names, values, color=bar_colors)
        axes[idx].set_title(metric, fontsize=14)
        axes[idx].set_xlabel('Score', fontsize=12)
        axes[idx].set_xlim(0, 1)
        axes[idx].invert_yaxis()

        for i, (v, color) in enumerate(zip(values, bar_colors)):
            axes[idx].text(v + 0.01, i, f"{v:.2f}",
                           va='center', color='black', fontsize=10)

    if len(metrics) < len(axes):
        for idx in range(len(metrics), len(axes)):
            fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig('outputs/plots/ML_results.png')
    plt.close()

    print("Training result plot saved to 'outputs/plots' directory")


import numpy as np
from sklearn.decomposition import PCA

def augment_medical_data(X_train, y_health_train, 
                        pca_components=0.95,
                        n_synthetic_samples=None,
                        noise_level=0.02,
                        random_state=42):
    """
    使用PCA和随机噪声扰动来增强医学数据。
    
    参数:
    X_train: numpy array, 形状为 (n_samples, n_features)
        原始训练数据
    y_health_train: numpy array, 形状为 (n_samples,)
        原始标签数据
    pca_components: float or int, 默认 0.95
        PCA保留的方差比例(0-1之间)或组件数量
    n_synthetic_samples: int or None, 默认 None
        要生成的合成样本数量，如果为None则生成与原始数据相同数量
    noise_level: float, 默认 0.02
        添加的随机噪声水平（相对于特征标准差的比例）
    random_state: int, 默认 42
        随机种子，用于结果复现
    
    返回:
    X_augmented: numpy array
        增强后的特征数据，包含原始数据和新生成的样本
    y_augmented: numpy array
        增强后的标签数据，包含原始标签和新生成样本的标签
    """
    np.random.seed(random_state)
    
    # 如果未指定生成样本数量，则设为原始数据量
    if n_synthetic_samples is None:
        n_synthetic_samples = len(X_train)
    
    # 1. 使用PCA进行数据转换
    pca = PCA(n_components=pca_components, random_state=random_state)
    X_pca = pca.fit_transform(X_train)
    
    # 2. 在PCA空间中生成新样本
    synthetic_samples = []
    synthetic_labels = []
    
    # 获取每个类别的索引
    unique_labels = np.unique(y_health_train)
    for label in unique_labels:
        # 获取当前类别的样本
        class_indices = np.where(y_health_train == label)[0]
        class_samples = X_pca[class_indices]
        
        # 计算该类别需要生成的样本数
        n_samples_class = int(n_synthetic_samples * (len(class_indices) / len(X_train)))
        
        # 计算该类别样本的均值和协方差
        mean = np.mean(class_samples, axis=0)
        cov = np.cov(class_samples.T)
        
        # 生成新样本
        new_samples = np.random.multivariate_normal(mean, cov, n_samples_class)
        
        # 添加随机噪声
        noise = np.random.normal(0, noise_level, new_samples.shape)
        new_samples += noise
        
        synthetic_samples.append(new_samples)
        synthetic_labels.extend([label] * n_samples_class)
    
    # 将生成的样本转换回原始特征空间
    synthetic_samples = np.vstack(synthetic_samples)
    X_synthetic = pca.inverse_transform(synthetic_samples)
    
    # 3. 合并原始数据和生成的数据
    X_augmented = np.vstack([X_train, X_synthetic])
    y_augmented = np.concatenate([y_health_train, synthetic_labels])
    
    return X_augmented, y_augmented




def filter_by_label(X_scaled, y_disease_encoded, label):
    """
    Filter X_scaled data based on binary labels in y_disease_encoded
    
    Args:
        X_scaled: Scaled feature matrix
        y_disease_encoded: Binary encoded labels (0 or 1)
        label: Target label to filter (0 or 1)
    
    Returns:
        Filtered X_scaled data
    """
    return X_scaled[y_disease_encoded == label], y_disease_encoded[y_disease_encoded == label]

# # Get data for each class
# X_scaled_0 = filter_by_label(X_scaled, y_disease_encoded, 0)
# X_scaled_1 = filter_by_label(X_scaled, y_disease_encoded, 1)
 


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

def main():
    """主程序入口"""
    try:
        # 创建目录
        setup_directories()

        # 初始化配置
        model_config = ModelConfig()

        # 初始化组件
        print("\nInitializing components...")
        data_processor = DataProcessor(model_config)
        model_trainer = ModelTrainer()

        # 数据处理流程
        print("\nProcessing data...")
        df = data_processor.load_data('data/Merged_Combined.csv')
        X_scaled, y_health_encoded, y_disease_encoded, feature_names = \
            data_processor.preprocess_data(df)

        print(y_health_encoded.shape, y_disease_encoded.shape, len(feature_names))

        # 数据集划分
        print("\nSplitting dataset...")
        split_results = data_processor.split_data(
            X_scaled, y_health_encoded, y_disease_encoded
        )
        X_train, X_test, y_health_train, y_health_test, y_disease_train, y_disease_test = split_results

        # 显示标签映射关系
        health_mapping, disease_mapping = data_processor.show_label_mappings()

        # 您也可以直接打印出这些映射
        print("Health Mapping:", health_mapping)
        print("Disease Mapping:", disease_mapping)


        # # method 0
        # model_trainer_results = {}
        # # 使用多个模型进行训练
        # print("\nTraining multiple models...")
        # model_trainer_results = model_trainer.train_models(X_train, y_health_train, X_test, y_health_test)

        # # 可视化训练结果（包括我们的方法）
        # plot_training_results(model_trainer_results)



        # # method 1
        # X_augmented, y_augmented = augment_medical_data(
        #     X_train=X_train,
        #     y_health_train=y_health_train,
        #     n_synthetic_samples=500  # 生成1000个新样本
        # )
        # model_trainer_results = {}
        # # 使用多个模型进行训练
        # print("\nTraining multiple models...")
        # model_trainer_results = model_trainer.train_models(
        #     X_augmented, y_augmented, X_test, y_health_test
        # )

        # # 可视化训练结果（包括我们的方法）
        # plot_training_results(model_trainer_results)



        # # method 2
        # from sklearn.preprocessing import StandardScaler
        # from imblearn.over_sampling import SMOTE, ADASYN
        # from imblearn.under_sampling import RandomUnderSampler
        # from sklearn.model_selection import cross_val_score

        # # 处理类别不平衡
        # smote = SMOTE(random_state=42)
        # # X_balanced, y_balanced = smote.fit_resample(X_scaled, y_health_encoded) 
        # X_balanced, y_balanced = smote.fit_resample(X_train, y_health_train)

        # model_trainer_results = {}
        # # 使用多个模型进行训练
        # print("\nTraining multiple models...")
        # model_trainer_results = model_trainer.train_models(X_balanced, y_balanced, X_test, y_health_test)

        # # 可视化训练结果（包括我们的方法）
        # plot_training_results(model_trainer_results)
        


        # # method 3
        # from imblearn.over_sampling import SMOTE

        # # 处理类别不平衡
        # smote = SMOTE(random_state=42)
        # X_balanced, y_balanced = smote.fit_resample(X_scaled, y_disease_encoded)

        # # print(X_scaled.shape, X_balanced.shape, np.unique(y_balanced))
        
        # y_balanced = np.array(y_balanced == 2) * 1
        # # print(X_scaled.shape, X_balanced.shape, np.unique(y_balanced))

        # model_trainer_results = {}
        # # 使用多个模型进行训练
        # print("\nTraining multiple models...")
        # model_trainer_results = model_trainer.train_models(X_balanced, y_balanced, X_test, y_health_test)

        # # 可视化训练结果（包括我们的方法）
        # plot_training_results(model_trainer_results)

        # print("\nProcessing completed successfully!")
        # print(f"Results saved in: {os.path.abspath('outputs')}")
        


        # method 4

        # np.random.shuffle(X_scaled.T)
        # np.random.seed(42)
        # np.random.shuffle(X_scaled)
        # np.random.seed(42)
        # np.random.shuffle(y_health_encoded)
        # np.random.shuffle(X_scaled_0)
        # X_scaled = shuffle_features_independently(X_scaled)

        # model_trainer_results = {}
        # # 使用多个模型进行训练
        # print("\nTraining multiple models...")
        # model_trainer_results = model_trainer.train_models(X_scaled, y_health_encoded, X_test, y_health_test)

        # # 可视化训练结果（包括我们的方法）
        # plot_training_results(model_trainer_results)



        # # method 5

        # # 数据集划分
        # print("\nSplitting dataset...")
        # split_results = data_processor.split_data(
        #     X_scaled, y_health_encoded, y_disease_encoded
        # )
        # X_train, X_test, y_health_train, y_health_test, y_disease_train, y_disease_test = split_results

        # np.random.seed(24)
        # np.random.shuffle(X_scaled)
        # np.random.seed(24)
        # np.random.shuffle(y_health_encoded)
        
        # split_results = data_processor.split_data(
        #     X_scaled, y_health_encoded, y_disease_encoded
        # )
        # _, X_test, _, y_health_test, _, y_disease_test = split_results

        # model_trainer_results = {}
        # # 使用多个模型进行训练
        # print("\nTraining multiple models...")
        # model_trainer_results = model_trainer.train_models(X_train, y_health_train, X_test, y_health_test)

        # # 可视化训练结果（包括我们的方法）
        # plot_training_results(model_trainer_results)



        # # method 5
        # 初始化训练器
        trainer = UncertaintyGuidedTrainer()

        # 训练模型
        model_trainer_results = trainer.train_with_uncertainty_guidance(X_train, y_health_train, X_test, y_health_test)

        # 可视化训练结果（包括我们的方法）
        plot_training_results(model_trainer_results)
        

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
