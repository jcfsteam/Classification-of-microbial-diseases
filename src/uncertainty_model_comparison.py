import time
import numpy as np
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

class UncertaintyGuidedTrainer:
    def __init__(self):
        self.models = {
            'RandomForest': RandomForestClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42),
            'LightGBM': LGBMClassifier(random_state=42),
            'CatBoost': CatBoostClassifier(random_state=42, verbose=0),
            'ExtraTrees': ExtraTreesClassifier(random_state=42),
        }
        self.uncertainty_thresholds = {}
        
    def calculate_prediction_uncertainty(self, model, X):
        """
        计算预测的不确定性
        方法1: 使用预测概率的熵
        方法2: 使用预测概率的方差
        """
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(X)
            
            # 方法1: 计算熵
            uncertainties_entropy = entropy(probas.T)
            
            # 方法2: 计算方差
            uncertainties_var = np.var(probas, axis=1)
            
            # 综合不确定性分数(归一化后的熵和方差的加权和)
            normalized_entropy = (uncertainties_entropy - np.min(uncertainties_entropy)) / (np.max(uncertainties_entropy) - np.min(uncertainties_entropy) + 1e-10)
            normalized_var = (uncertainties_var - np.min(uncertainties_var)) / (np.max(uncertainties_var) - np.min(uncertainties_var) + 1e-10)
            
            return 0.6 * normalized_entropy + 0.4 * normalized_var
        else:
            return np.zeros(len(X))  # 对于不支持概率预测的模型返回零不确定性

    def get_uncertain_samples(self, model, X, uncertainty_threshold=0.7):
        """
        识别高不确定性的样本
        """
        uncertainties = self.calculate_prediction_uncertainty(model, X)
        return uncertainties > uncertainty_threshold, uncertainties

    def train_with_uncertainty_guidance(self, X_train, y_train, X_test, y_test, n_iterations=10):
        """
        使用不确定性引导的迭代训练过程
        """
        results = {}
        classes = np.unique(y_train)
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name} with uncertainty guidance...")
            start_time = time.time()
            
            current_model = model
            X_train_current = X_train.copy()
            y_train_current = y_train.copy()
            
            # 迭代训练过程
            for iteration in range(n_iterations):
                print(f"Iteration {iteration + 1}/{n_iterations}")
                
                # 训练模型
                current_model.fit(X_train_current, y_train_current)
                
                # 识别不确定样本
                uncertain_mask, uncertainties = self.get_uncertain_samples(
                    current_model, 
                    X_train_current,
                    uncertainty_threshold=0.7 - (iteration * 0.1)  # 随着迭代逐渐降低阈值
                )
                
                # 保存当前迭代的不确定性阈值
                self.uncertainty_thresholds[f"{model_name}_iter_{iteration}"] = np.mean(uncertainties)
                
                # 基于不确定性进行采样
                if np.any(uncertain_mask):
                    # 对不确定样本增加权重或者进行重采样
                    uncertain_indices = np.where(uncertain_mask)[0]
                    X_train_current = np.vstack([X_train_current, X_train_current[uncertain_indices]])
                    y_train_current = np.hstack([y_train_current, y_train_current[uncertain_indices]])
            
            training_time = time.time() - start_time
            
            # 最终评估
            start_time = time.time()
            y_pred = current_model.predict(X_test)
            prediction_time = time.time() - start_time
            
            # 计算ROC AUC
            if hasattr(current_model, 'predict_proba'):
                y_test_bin = label_binarize(y_test, classes=classes)
                y_pred_prob = current_model.predict_proba(X_test)
                if len(classes) == 2:
                    roc_auc = roc_auc_score(y_test, y_pred_prob[:, 1])
                else:
                    roc_auc = roc_auc_score(y_test_bin, y_pred_prob, multi_class='ovr')
            else:
                roc_auc = 'N/A'
            
            # 计算其他指标
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # 记录不确定性统计
            final_uncertainties = self.calculate_prediction_uncertainty(current_model, X_test)
            
            results[model_name] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'ROC AUC': roc_auc,
                'Training Time (s)': training_time,
                'Prediction Time (s)': prediction_time,
                'Mean Uncertainty': np.mean(final_uncertainties),
                'Max Uncertainty': np.max(final_uncertainties),
                'Uncertainty Thresholds': self.uncertainty_thresholds
            }
            
            # 打印结果
            print(f"{model_name} Final Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"ROC AUC: {roc_auc}")
            print(f"Training Time: {training_time:.4f} seconds")
            print(f"Prediction Time: {prediction_time:.4f} seconds")
            print(f"Mean Uncertainty: {np.mean(final_uncertainties):.4f}")
            
        return results