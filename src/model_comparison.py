import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np

class ModelTrainer:
    def __init__(self):
        self.models = {
            'RandomForest': RandomForestClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42),
            # 'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'LightGBM': LGBMClassifier(random_state=42),
            'CatBoost': CatBoostClassifier(random_state=42, verbose=0),
            # 'GradientBoosting': GradientBoostingClassifier(random_state=42),
            # 'AdaBoost': AdaBoostClassifier(random_state=42),
            'ExtraTrees': ExtraTreesClassifier(random_state=42),
            # 'RidgeClassifier': RidgeClassifier(random_state=42),
            # 'SGDClassifier': SGDClassifier(random_state=42),
            # 'SVC': SVC(probability=True, random_state=42),
            # 'KNeighbors': KNeighborsClassifier(),
            # 'GaussianNB': GaussianNB(),
            # 'BernoulliNB': BernoulliNB(),
            # 'DecisionTree': DecisionTreeClassifier(random_state=42),
            # 'MLPClassifier': MLPClassifier(random_state=42, max_iter=1000)
        }

    def train_models(self, X_train, y_train, X_test, y_test):
        results = {}
        classes = np.unique(y_train)

        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            start_time = time.time()

            # Train the model
            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            # Predict and evaluate
            start_time = time.time()
            y_pred = model.predict(X_test)
            prediction_time = time.time() - start_time

            # Handle multi-class ROC AUC
            if hasattr(model, 'predict_proba'):
                y_test_bin = label_binarize(y_test, classes=classes)
                y_pred_prob = model.predict_proba(X_test)
                if len(classes) == 2:
                    roc_auc = roc_auc_score(y_test, y_pred_prob[:, 1])
                else:
                    roc_auc = roc_auc_score(y_test_bin, y_pred_prob, multi_class='ovr')
            else:
                roc_auc = 'N/A'
            # roc_auc = 0

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted')

            results[model_name] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'ROC AUC': roc_auc,
                'Training Time (s)': training_time,
                'Prediction Time (s)': prediction_time
            }

            print(f"{model_name} Accuracy: {accuracy:.4f}")
            print(f"{model_name} Precision: {precision:.4f}")
            print(f"{model_name} Recall: {recall:.4f}")
            print(f"{model_name} F1 Score: {f1:.4f}")
            print(f"{model_name} ROC AUC: {roc_auc}")
            print(f"{model_name} Training Time: {training_time:.4f} seconds")
            print(f"{model_name} Prediction Time: {prediction_time:.4f} seconds")

        return results