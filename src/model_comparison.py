# trainer.py

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class ModelTrainer:
    def __init__(self):
        self.models = {
            'RandomForest': RandomForestClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42),
            # 'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'LightGBM': LGBMClassifier(random_state=42),
            'CatBoost': CatBoostClassifier(random_state=42, verbose=0),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'AdaBoost': AdaBoostClassifier(random_state=42),
            'ExtraTrees': ExtraTreesClassifier(random_state=42),
            # 'RidgeClassifier': RidgeClassifier(random_state=42),
            # 'SGDClassifier': SGDClassifier(random_state=42),
            # 'SVC': SVC(probability=True, random_state=42),
            'KNeighbors': KNeighborsClassifier(),
            # 'GaussianNB': GaussianNB(),
            # 'BernoulliNB': BernoulliNB(),
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'MLPClassifier': MLPClassifier(random_state=42, max_iter=1000)
        }

    def train_models(self, X_train, y_train, X_test, y_test):
        results = {}
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted')
            roc_auc = roc_auc_score(y_test, y_pred_prob) if y_pred_prob is not None else 'N/A'

            results[model_name] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'ROC AUC': roc_auc
            }

            print(f"{model_name} Accuracy: {accuracy:.4f}")
            print(f"{model_name} Precision: {precision:.4f}")
            print(f"{model_name} Recall: {recall:.4f}")
            print(f"{model_name} F1 Score: {f1:.4f}")
            print(f"{model_name} ROC AUC: {roc_auc}")
        
        return results
