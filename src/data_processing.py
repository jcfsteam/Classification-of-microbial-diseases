# src/data_processing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, List
from config.model_config import ModelConfig

class DataProcessor:
    def __init__(self, config: ModelConfig):
        """
        初始化数据处理器
        
        Parameters:
        -----------
        config : ModelConfig
            模型配置对象
        """
        self.config = config
        self.scaler = StandardScaler()
        self.le_health = LabelEncoder()
        self.le_disease = LabelEncoder()
        self.feature_names = None  # 添加特征名称属性
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        加载数据并进行基础清理
        
        Parameters:
        -----------
        filepath : str
            数据文件路径
            
        Returns:
        --------
        pd.DataFrame
            清理后的数据框
        """
        print(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        return df
        
    def preprocess_data(self, df: pd.DataFrame) -> Tuple:
        """
        数据预处理主函数
        
        Parameters:
        -----------
        df : pd.DataFrame
            原始数据框
            
        Returns:
        --------
        Tuple
            处理后的特征矩阵和标签
        """
        print("Preprocessing data...")
        
        # 提取特征和标签
        X = df.iloc[:, :-3]
        y_health = df.iloc[:, -3]  # OK/NG
        y_disease = df.iloc[:, -1]  # AD/PD/ASD/Control
        
        # 保存特征名称
        self.feature_names = X.columns.tolist()
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 标签编码
        y_health_encoded = self.le_health.fit_transform(y_health)
        y_disease_encoded = self.le_disease.fit_transform(y_disease)
        
        print(f"Preprocessed {X.shape[0]} samples with {X.shape[1]} features")
        print(f"Health status classes: {self.le_health.classes_}")
        print(f"Disease classes: {self.le_disease.classes_}")
        
        return X_scaled, y_health_encoded, y_disease_encoded, self.feature_names

    def split_data(self, X: np.ndarray, y_health: np.ndarray, 
                   y_disease: np.ndarray) -> Tuple:
        """
        划分训练集和测试集
        
        Parameters:
        -----------
        X : np.ndarray
            特征矩阵
        y_health : np.ndarray
            健康状态标签
        y_disease : np.ndarray
            疾病类型标签
            
        Returns:
        --------
        Tuple
            训练集和测试集数据
        """
        print("Splitting data into train and test sets...")
        return train_test_split(
            X, y_health, y_disease,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y_disease  # 使用疾病标签进行分层采样
        )
    
    def get_label_encoders(self) -> Tuple[LabelEncoder, LabelEncoder]:
        """
        获取标签编码器
        
        Returns:
        --------
        Tuple[LabelEncoder, LabelEncoder]
            健康状态和疾病类型的标签编码器
        """
        return self.le_health, self.le_disease

    def inverse_transform_labels(self, y_health: np.ndarray,
                               y_disease: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        将编码后的标签转换回原始标签
        
        Parameters:
        -----------
        y_health : np.ndarray
            编码后的健康状态标签
        y_disease : np.ndarray
            编码后的疾病类型标签
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            原始标签
        """
        return (
            self.le_health.inverse_transform(y_health),
            self.le_disease.inverse_transform(y_disease)
        )