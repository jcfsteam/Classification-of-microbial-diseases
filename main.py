# main.py

import os
import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# 导入自定义模块
from config.model_config import ModelConfig, DiffusionConfig
from src.data_processing import DataProcessor
from src.feature_selection import FeatureSelector
from src.models.classifier import DiseaseClassifier
from src.data_augmentation import DataAugmentor
from src.model_comparison import ModelTrainer

def setup_directories():
    """创建必要的目录结构"""
    dirs = ['outputs', 'outputs/models', 'outputs/plots', 'outputs/results']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def save_results(results: dict, classifier: DiseaseClassifier, 
                data_processor: DataProcessor, selected_features: np.ndarray):
    """
    保存模型结果和相关组件
    
    Parameters:
    -----------
    results : dict
        模型评估结果
    classifier : DiseaseClassifier
        训练好的分类器
    data_processor : DataProcessor
        数据处理器
    selected_features : np.ndarray
        选中的特征索引
    """
    print("\nSaving results and models...")
    
    # 保存模型和预处理器
    joblib.dump(classifier.model, 'outputs/models/disease_model.joblib')
    joblib.dump(data_processor.le_disease, 'outputs/models/disease_label_encoder.joblib')
    joblib.dump(data_processor.scaler, 'outputs/models/scaler.joblib')
    
    # 保存特征选择结果
    feature_selection_results = {
        'selected_features': selected_features,
        'feature_names': data_processor.feature_names,
        'importance_scores': results.get('importance_scores')
    }
    joblib.dump(feature_selection_results, 'outputs/results/feature_selection_results.joblib')
    
    # 保存评估指标
    metrics_df = pd.DataFrame({
        'Metric': [
            'Disease Classification AUC',
            'Selected Features Count'
        ],
        'Value': [
            results['auc_score'],
            len(selected_features)
        ]
    })
    metrics_df.to_csv('outputs/results/model_metrics.csv', index=False)
    
    # 保存分类报告
    classification_results = pd.DataFrame(results['classification_report']).transpose()
    classification_results.to_csv('outputs/results/classification_report.csv')
    
    print("Results and models saved to 'outputs' directory")

def plot_training_results(results: dict):
    """
    绘制并保存训练结果可视化
    
    Parameters:
    -----------
    results : dict
        包含训练结果的字典
    """
    # 将评估指标作为x轴，将各模型进行对比
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    models = list(results.keys())
    
    x = np.arange(len(metrics))  # x轴为评估指标
    width = 0.15  # 每个条形的宽度
    
    plt.figure(figsize=(12, 6))
    for i, model in enumerate(models):
        values = [results[model][metric] if results[model][metric] != 'N/A' else 0 for metric in metrics]
        plt.bar(x + i * width, values, width, label=model)
    
    # 设置图表标签和标题
    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.title('Metric Comparison by Models')
    plt.xticks(x + width * (len(models) - 1) / 2, metrics, rotation=45, ha='right')
    plt.ylim(0.5, 1.0)  # 不显示y轴0-0.5区间，增强差异可视化
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/plots/metric_comparison.png')
    plt.close()

    print("Training result plot saved to 'outputs/plots' directory")

    

def main():
    """主程序入口"""
    try:
        # 创建目录
        setup_directories()
        
        # 初始化配置
        model_config = ModelConfig()
        diffusion_config = DiffusionConfig()
        
        # 初始化组件
        print("\nInitializing components...")
        data_processor = DataProcessor(model_config)
        feature_selector = FeatureSelector(model_config)
        classifier = DiseaseClassifier(model_config)
        augmentor = DataAugmentor(diffusion_config)
        model_trainer = ModelTrainer()
        
        # 数据处理流程
        print("\nProcessing data...")
        df = data_processor.load_data('data/Merged_Combined.csv')
        X_scaled, y_health_encoded, y_disease_encoded, feature_names = \
            data_processor.preprocess_data(df)
        
        print(y_health_encoded, y_disease_encoded, feature_names)
        
        # 数据集划分
        print("\nSplitting dataset...")
        split_results = data_processor.split_data(
            X_scaled, y_health_encoded, y_disease_encoded
        )
        X_train, X_test, y_health_train, y_health_test, y_disease_train, y_disease_test = split_results
        
        # 使用多个模型进行训练
        print("\nTraining multiple models...")
        model_trainer_results = model_trainer.train_models(X_train, y_health_train, X_test, y_health_test)
        print(model_trainer_results)
        
        # 可视化训练结果
        plot_training_results(model_trainer_results)
        
        # 数据增强
        print("\nPerforming data augmentation...")
        X_train_augmented, y_disease_train_augmented = augmentor.augment_data(
            X_train, y_disease_train
        )
        
        # 特征选择
        print("\nPerforming feature selection...")
        selected_features, importance_scores = feature_selector.select_features(
            X_scaled, y_disease_encoded
        )
        X_train_selected = X_train_augmented[:, selected_features]
        X_test_selected = X_test[:, selected_features]
        
        # 绘制特征重要性
        feature_selector.visualize_importance(
            feature_names,
            importance_scores,
            selected_features
        )
        plt.savefig('outputs/plots/feature_importance.png')
        plt.close()
        
        # 训练和评估模型
        print("\nTraining and evaluating model...")
        class_names = data_processor.le_disease.classes_
        classifier.train(X_train_selected, y_disease_train_augmented)
        
        results = classifier.evaluate(
            X_test_selected, 
            y_disease_test,
            class_names=class_names
        )

        # 分析健康状态预测
        print("\nAnalyzing health status predictions...")
        health_predictions, health_accuracy, health_auc = classifier.analyze_health_predictions(
            results['predictions'],
            results['probabilities'],
            data_processor.le_health.inverse_transform(y_health_test),
            class_names
        )
        
        # 更新结果字典
        results.update({
            'health_predictions': health_predictions,
            'health_accuracy': health_accuracy,
            'health_auc': health_auc,
            'importance_scores': importance_scores
        })
        
        # 保存结果
        save_results(results, classifier, data_processor, selected_features)
        # plot_training_results(results)
        
        print("\nProcessing completed successfully!")
        print(f"Results saved in: {os.path.abspath('outputs')}")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
