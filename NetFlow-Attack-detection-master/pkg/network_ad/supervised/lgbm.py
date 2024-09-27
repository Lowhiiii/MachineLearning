import sys
sys.path.append("../..")
import json
import os

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter

from network_ad.config import LOGS_DIR, TRAIN_DATA_PATH, TEST_DATA_PATH, MULTIClASS_CLASS_NAMES, BINARY_CLASS_NAMES
from network_ad.supervised.lgbm_dataset import LightGBMDataset


class LightGBMClassifier:
    def __init__(self,
                 multiclass=True,
                 num_leaves=16,
                 max_depth=-1,
                 num_estimators=100,
                 learning_rate=0.05,
                 n_kfold_splits=5,
                 log_dir=LOGS_DIR / "lgbm"):
        """Initialize with LightGBM parameters and K-Fold setup."""
        self.params = {
            'objective': 'multiclass' if multiclass else 'binary',
            'metric': 'multi_logloss' if multiclass else 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': num_leaves,
            'max_depth': max_depth,
            'n_estimators': num_estimators,
            'learning_rate': learning_rate,
        }
        self.n_splits = n_kfold_splits
        self.models = []
        self.best_iteration = []
        self.multiclass = multiclass
        self.class_labels = MULTIClASS_CLASS_NAMES if multiclass else BINARY_CLASS_NAMES

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)  # TensorBoard writer

    def train_kfold(self, X, y, save_validation_performance_path="validation_metrics.json"):
        """Perform K-fold cross-validation training and save validation performance."""
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        validation_results = []

        fold = 1
        for train_index, val_index in kf.split(X):
            print(f"Training fold {fold}/{self.n_splits}...")

            # Split data into train and validation sets for this fold
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            # Train LightGBM model
            model = lgb.LGBMClassifier(**self.params, class_weight='balanced', n_jobs=-1)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

            # Save the model and best iteration
            self.models.append(model)
            self.best_iteration.append(model._best_iteration)

            # Evaluate on validation set
            val_preds = model.predict(X_val, num_iteration=model._best_iteration)
            accuracy = accuracy_score(y_val, val_preds)

            # Calculate F1 Score, Precision, and Recall
            labels = np.unique(y_val)  # Get unique labels in the current fold
            classification_metrics = classification_report(y_val, val_preds,
                                                           labels = labels,
                                                           output_dict=True)
            f1_weighted = classification_metrics['weighted avg']['f1-score']
            f1_macro = classification_metrics['macro avg']['f1-score']
            precision = classification_metrics['weighted avg']['precision']
            recall = classification_metrics['weighted avg']['recall']

            validation_results.append({
                "fold": fold,
                "accuracy": accuracy,
                "f1_weighted": f1_weighted,
                "f1_macro": f1_macro,
                "precision": precision,
                "recall": recall
            })

            print(f"Finished fold {fold}/{self.n_splits}, accuracy: {accuracy}, f1_weighted: {f1_weighted}, f1_macro: {f1_macro}")

            # Log confusion matrix to TensorBoard for each fold
            cm = confusion_matrix(y_val, val_preds, labels=self.class_labels)
            self._log_confusion_matrix(cm, f'Confusion Matrix - Fold {fold}')

            # Cm normalized
            cm_norm = confusion_matrix(y_val, val_preds, normalize='true', labels= self.class_labels)
            self._log_confusion_matrix(cm_norm, f'Normalized Confusion Matrix - Fold {fold}', normalize=True)

            fold += 1

        # Save validation results in JSON format
        with open(save_validation_performance_path, 'w') as f:
            json.dump(validation_results, f, indent=4)

    def predict(self, X_test):
        """Make predictions using the average of all K-fold models."""
        preds = np.zeros((len(X_test), len(self.models[0].predict_proba(X_test, num_iteration=self.models[0]._best_iteration)[0])))

        for model in self.models:
            preds += model.predict_proba(X_test, num_iteration=model._best_iteration)

        preds /= len(self.models)
        pred_ids =  np.argmax(preds, axis=1)  # Return the class with the highest probability
        pred_labels = [self.class_labels[i] for i in pred_ids]
        return pred_labels

    def evaluate_test(self, X_test, y_test, save_test_performance_path="test_metrics.json"):
        """Evaluate the model on the test set and save the results."""
        test_preds = self.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, test_preds)

        # Calculate F1 Score, Precision, and Recall
        labels = np.unique(y_test)  # Get unique labels in the test set
        classification_metrics = classification_report(y_test, test_preds,
                                                       labels = labels,
                                                       output_dict=True)
        f1_weighted = classification_metrics['weighted avg']['f1-score']
        f1_macro = classification_metrics['macro avg']['f1-score']
        precision = classification_metrics['weighted avg']['precision']
        recall = classification_metrics['weighted avg']['recall']

        # Save performance metrics to JSON file
        test_results = {
            "accuracy": accuracy,
            "f1_weighted": f1_weighted,
            "f1_macro": f1_macro,
            "precision": precision,
            "recall": recall
        }

        with open(save_test_performance_path, 'w') as f:
            json.dump(test_results, f, indent=4)

        # Log confusion matrix to TensorBoard
        cm = confusion_matrix(y_test, test_preds, labels=self.class_labels)
        self._log_confusion_matrix(cm, 'Test Confusion Matrix')

        # Cm normalized
        cm_norm = confusion_matrix(y_test, test_preds, normalize='true', labels=self.class_labels)
        self._log_confusion_matrix(cm_norm, 'Normalized Test Confusion Matrix', normalize=True)

        print(f"Test accuracy: {accuracy}, f1_weighted: {f1_weighted}, f1_macro: {f1_macro}, precision: {precision}, recall: {recall}")

    def save_model(self, path):
        """Save the trained model."""
        for i, model in enumerate(self.models):
            model.save_model(f"{path}_fold_{i}.txt")

    def load_model(self, path, n_splits):
        """Load the trained model."""
        self.models = []
        for i in range(n_splits):
            model = lgb.Booster(model_file=f"{path}_fold_{i}.txt")
            self.models.append(model)

    def _log_confusion_matrix(self, cm, tag, normalize=False):
        """Log confusion matrix to TensorBoard."""
        fig, ax = plt.subplots(figsize=(8, 6))
        fmt = ".2f" if normalize else "d"
        sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(tag)

        # Convert matplotlib plot to TensorBoard image format
        self.writer.add_figure(tag, fig)
        plt.close(fig)


if __name__ == "__main__":
    MULTICLASS = True

    # 1. Load and prepare dataset
    dataset = LightGBMDataset(
        train_path=TRAIN_DATA_PATH,
        test_path=TEST_DATA_PATH,
        multiclass=MULTICLASS)
    dataset.setup()

    # 2. Initialize LightGBM classifier
    classifier = LightGBMClassifier(
        multiclass=True,
        num_leaves=16,
        max_depth=-1,
        num_estimators=100,
        log_dir=LOGS_DIR / "lgbm"  # Custom TensorBoard log directory
    )

    # 3. Train with K-Fold cross-validation
    classifier.train_kfold(dataset.X_train, dataset.y_train)

    # 4. Evaluate the model on the test set
    classifier.evaluate_test(dataset.X_test, dataset.y_test)

    # 5. Save the model
    classifier.save_model("lightgbm_model")
