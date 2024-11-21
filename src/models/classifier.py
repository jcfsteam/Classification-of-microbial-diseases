# src/models/classifier.py

from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from config.model_config import ModelConfig

class DiseaseClassifier:
    def __init__(self, config: ModelConfig):
        """
        初始化疾病分类器
        
        Parameters:
        -----------
        config : ModelConfig
            模型配置对象
        """
        self.config = config
        self.model = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            random_state=config.random_state
        )
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        训练模型
        
        Parameters:
        -----------
        X_train : np.ndarray
            训练数据特征
        y_train : np.ndarray
            训练数据标签
        """
        print("Training classifier...")
        self.model.fit(X_train, y_train)
        print("Training completed.")
        

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                class_names: List[str]) -> Dict:
        """
        评估模型性能
        
        Parameters:
        -----------
        X_test : np.ndarray
            测试数据特征
        y_test : np.ndarray
            测试数据标签
        class_names : List[str]
            类别名称列表
            
        Returns:
        --------
        Dict
            包含各种评估指标的字典
        """
        print("\nEvaluating model performance...")
        
        # 获取预测结果
        predictions = self.model.predict(X_test)
        pred_proba = self.model.predict_proba(X_test)
        
        # 计算ROC AUC
        y_test_onehot = np.eye(len(class_names))[y_test]
        auc_score = roc_auc_score(
            y_test_onehot, 
            pred_proba, 
            multi_class='ovr', 
            average='macro'
        )
        
        # 准备评估结果
        results = {
            'predictions': predictions,
            'true_labels': y_test,  # 添加真实标签到结果字典中
            'probabilities': pred_proba,
            'auc_score': auc_score,
            'classification_report': classification_report(
                y_test, 
                predictions, 
                target_names=class_names,
                output_dict=True
            ),
            'confusion_matrix': confusion_matrix(y_test, predictions)
        }
        
        # 打印评估结果
        self._print_evaluation_results(results, class_names)
        
        # 可视化结果
        self._plot_confusion_matrix(results['confusion_matrix'], class_names)
        self._plot_roc_curves(y_test_onehot, pred_proba, class_names)
        
        return results
    
    def _print_evaluation_results(self, results: Dict, 
                                class_names: List[str]) -> None:
        """
        打印评估结果
        
        Parameters:
        -----------
        results : Dict
            评估结果字典，包含预测值、真实标签和评估指标
        class_names : List[str]
            类别名称列表
        """
        print("\nModel Evaluation Results:")
        print("-" * 50)
        print(f"Overall AUC Score: {results['auc_score']:.4f}")
        print("\nClassification Report:")
        print(classification_report(
            results['true_labels'],  # 使用真实标签
            results['predictions'],  # 使用预测值
            target_names=class_names
        ))
    
    def _plot_confusion_matrix(self, cm: np.ndarray, 
                             class_names: List[str]) -> None:
        """绘制混淆矩阵"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    def _plot_roc_curves(self, y_test_onehot: np.ndarray,
                        pred_proba: np.ndarray,
                        class_names: List[str]) -> None:
        """绘制ROC曲线"""
        plt.figure(figsize=(10, 8))
        
        for i in range(len(class_names)):
            fpr, tpr, _ = roc_curve(y_test_onehot[:, i], pred_proba[:, i])
            plt.plot(
                fpr, 
                tpr, 
                label=f'{class_names[i]} (AUC = {roc_auc_score(y_test_onehot[:, i], pred_proba[:, i]):.4f})'
            )
            
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用训练好的模型进行预测
        
        Parameters:
        -----------
        X : np.ndarray
            输入特征
            
        Returns:
        --------
        np.ndarray
            预测结果
        """
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率
        
        Parameters:
        -----------
        X : np.ndarray
            输入特征
            
        Returns:
        --------
        np.ndarray
            预测概率
        """
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> np.ndarray:
        """
        获取特征重要性
        
        Returns:
        --------
        np.ndarray
            特征重要性分数
        """
        return self.model.feature_importances_
    
    def analyze_health_predictions(self, disease_preds: np.ndarray, disease_probs: np.ndarray,
                                 true_health_status: List[str], pred_classes: List[str]) -> (List[str], float, float):
        """
        分析疾病预测结果与健康状态的关系，包括AUC计算
        
        Parameters:
        -----------
        disease_preds : np.ndarray
            疾病预测的标签索引
        disease_probs : np.ndarray
            疾病预测的概率矩阵
        true_health_status : List[str]
            实际健康状态标签（OK 或 NG）
        pred_classes : List[str]
            预测类别的名称列表
        
        Returns:
        --------
        List[str]
            预测的健康状态
        float
            健康状态预测的准确性
        float
            健康状态预测的AUC值
        """
        # 将预测结果转换为健康状态
        health_status_mapping = {
            'Control': 'OK',
            'AD': 'NG',
            'PD': 'NG',
            'ASD': 'NG'
        }
        
        predicted_health = [health_status_mapping[pred_classes[pred]] for pred in disease_preds]
        
        # 计算健康状态的概率
        # 将Control的概率作为OK的概率，其他疾病概率之和作为NG的概率
        control_idx = np.where(np.array(pred_classes) == 'Control')[0][0]
        health_probs = np.zeros((len(disease_probs), 2))  # [OK_prob, NG_prob]
        health_probs[:, 0] = disease_probs[:, control_idx]  # OK probability
        health_probs[:, 1] = 1 - health_probs[:, 0]  # NG probability
        
        # 计算健康状态预测的准确性
        accuracy = np.mean([p == t for p, t in zip(predicted_health, true_health_status)])
        
        # 计算健康状态的AUC
        # 将标签转换为数值（OK=0, NG=1）
        true_health_binary = np.array([0 if status == 'OK' else 1 for status in true_health_status])
        health_auc = roc_auc_score(true_health_binary, health_probs[:, 1])
        
        print("\nHealth Status Prediction Analysis:")
        print("-" * 50)
        print(f"Accuracy in predicting health status: {accuracy:.4f}")
        print(f"AUC in predicting health status: {health_auc:.4f}")
        print("\nClassification Report for Health Status:")
        print(classification_report(true_health_status, predicted_health))
        
        # 绘制健康状态的混淆矩阵
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(true_health_status, predicted_health)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['OK', 'NG'], yticklabels=['OK', 'NG'])
        plt.title('Health Status Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
        # 绘制健康状态的ROC曲线
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(true_health_binary, health_probs[:, 1])
        plt.plot(fpr, tpr, label=f'AUC = {health_auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Health Status Prediction')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()
        
        return predicted_health, accuracy, health_auc
