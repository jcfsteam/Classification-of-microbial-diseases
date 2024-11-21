# src/feature_selection.py

from typing import Tuple, Optional, List
import numpy as np
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from config.model_config import ModelConfig
import pandas as pd

class FeatureSelector:
    def __init__(self, config: ModelConfig):
        self.config = config
        
    def select_features(self, X: np.ndarray, y: np.ndarray, 
                       method: str = 'combined',
                       n_features_to_select: Optional[int] = None) -> Tuple:
        """
        特征选择主函数
        
        Parameters:
        -----------
        X : np.ndarray
            特征矩阵
        y : np.ndarray
            目标变量
        method : str
            特征选择方法 ('rf_importance', 'rfe', 'combined')
        n_features_to_select : int, optional
            要选择的特征数量
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            选中的特征掩码和特征重要性分数
        """
        if n_features_to_select is None:
            n_features_to_select = X.shape[1] // 4
            
        if method == 'combined':
            return self._combined_selection(X, y, n_features_to_select)
        elif method == 'rf_importance':
            return self._rf_importance_selection(X, y, n_features_to_select)
        else:
            return self._rfe_selection(X, y, n_features_to_select)
    
    def _rf_importance_selection(self, X: np.ndarray, y: np.ndarray,
                               n_features: int) -> Tuple[np.ndarray, np.ndarray]:
        """使用随机森林特征重要性进行特征选择"""
        rf = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            random_state=self.config.random_state
        )
        rf.fit(X, y)
        selector = SelectFromModel(rf, max_features=n_features)
        selector.fit(X, y)
        return selector.get_support(), rf.feature_importances_
    
    def _rfe_selection(self, X: np.ndarray, y: np.ndarray,
                      n_features: int) -> Tuple[np.ndarray, np.ndarray]:
        """使用递归特征消除进行特征选择"""
        rf = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            random_state=self.config.random_state
        )
        selector = RFECV(
            estimator=rf,
            min_features_to_select=n_features,
            step=1,
            cv=5,
            n_jobs=-1
        )
        selector.fit(X, y)
        return selector.support_, selector.ranking_
    
    def _combined_selection(self, X: np.ndarray, y: np.ndarray,
                          n_features: int) -> Tuple[np.ndarray, np.ndarray]:
        """结合随机森林重要性和RFE的特征选择方法"""
        # 第一步：使用随机森林特征重要性进行初步筛选
        mask_rf, importance_rf = self._rf_importance_selection(
            X, y, min(n_features * 2, X.shape[1])
        )
        
        # 第二步：在预选特征上使用RFE进行精细选择
        X_selected = X[:, mask_rf]
        mask_rfe, _ = self._rfe_selection(X_selected, y, n_features)
        
        # 合并最终的特征掩码
        final_mask = np.zeros(X.shape[1], dtype=bool)
        final_mask[mask_rf] = mask_rfe
        
        return final_mask, importance_rf
    
    def visualize_importance(self, feature_names: List[str],
                           importance_scores: np.ndarray,
                           selected_mask: np.ndarray,
                           top_n: int = 20) -> None:
        """
        可视化特征重要性
        
        Parameters:
        -----------
        feature_names : List[str]
            特征名称列表
        importance_scores : np.ndarray
            特征重要性分数
        selected_mask : np.ndarray
            特征选择掩码
        top_n : int
            显示前n个最重要的特征
        """
        plt.figure(figsize=(12, 6))
        
        # 创建特征重要性数据框
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores,
            'Selected': selected_mask
        })
        
        # 按重要性排序并获取前N个特征
        importance_df = importance_df.sort_values('Importance', ascending=True)
        if top_n is not None:
            importance_df = importance_df.tail(top_n)
        
        # 创建颜色映射
        colors = ['#2ecc71' if selected else '#e74c3c' 
                 for selected in importance_df['Selected']]
        
        # 绘制条形图
        plt.barh(range(len(importance_df)), importance_df['Importance'], 
                color=colors)
        plt.yticks(range(len(importance_df)), importance_df['Feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top Feature Importance')
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', label='Selected'),
            Patch(facecolor='#e74c3c', label='Not Selected')
        ]
        plt.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.show()
    
    def print_feature_summary(self, feature_names: List[str], 
                            selected_features: np.ndarray,
                            importance_scores: np.ndarray) -> None:
        """
        打印特征选择摘要
        
        Parameters:
        -----------
        feature_names : List[str]
            特征名称列表
        selected_features : np.ndarray
            选中的特征索引
        importance_scores : np.ndarray
            特征重要性分数
        """
        print("\nFeature Selection Summary:")
        print("-" * 50)
        print(f"Total features: {len(feature_names)}")
        print(f"Selected features: {len(selected_features)}")
        print("\nTop 10 most important features:")
        
        # 创建特征重要性排名
        feature_ranking = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores
        }).sort_values('Importance', ascending=False)
        
        for idx, (feature, importance) in enumerate(
            zip(feature_ranking['Feature'].head(10), 
                feature_ranking['Importance'].head(10)), 1):
            print(f"{idx}. {feature}: {importance:.4f}")