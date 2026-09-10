"""Model A — Supervised Resume Category Classifier."""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from src.config import (
    CLASSIFIER_PATH,
    CLASSIFIER_VEC_PATH,
    RANDOM_STATE,
    RESUMES_PROCESSED_PATH,
    TEST_SIZE,
    TFIDF_MAX_FEATURES_CLASSIFIER,
    TFIDF_NGRAM_RANGE,
)
from src.evaluate import evaluate_classifier
from src.features.text_features import clean_text


class ResumeClassifier:
    """Supervised text classifier for predicting candidate domain categories."""

    def __init__(
        self,
        vectorizer: Optional[TfidfVectorizer] = None,
        model: Optional[object] = None,
    ):
        self.vectorizer = vectorizer
        self.model = model
        self.classes_: Optional[np.ndarray] = None
        if model is not None and hasattr(model, "classes_"):
            self.classes_ = model.classes_

    def fit(
        self,
        texts: List[str],
        labels: List[str],
        model_type: str = "logistic_regression",
    ) -> "ResumeClassifier":
        """Fit the TF-IDF vectorizer and chosen supervised model."""
        self.vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES_CLASSIFIER,
            ngram_range=TFIDF_NGRAM_RANGE,
            sublinear_tf=True,
            stop_words="english",
        )
        X = self.vectorizer.fit_transform(texts)

        if model_type == "logistic_regression":
            self.model = LogisticRegression(
                max_iter=1000,
                C=1.0,
                random_state=RANDOM_STATE,
                solver="lbfgs",
            )
        elif model_type == "linear_svm":
            base_svc = LinearSVC(C=1.0, random_state=RANDOM_STATE, dual="auto")
            self.model = CalibratedClassifierCV(estimator=base_svc, cv=3)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.model.fit(X, labels)
        self.classes_ = self.model.classes_
        return self

    def predict(self, raw_text: str) -> Dict[str, object]:
        """Predict domain category with confidence score for a single resume text."""
        if self.model is None or self.vectorizer is None:
            raise RuntimeError("Classifier is not trained or loaded. Call load() or train() first.")

        cleaned = clean_text(raw_text)
        if not cleaned:
            return {
                "predicted_category": "Unknown",
                "confidence": 0.0,
                "top_categories": [],
                "explanation": "Input text is empty or could not be processed.",
            }

        X_input = self.vectorizer.transform([cleaned])
        pred_idx = self.model.predict(X_input)[0]

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_input)[0]
            confidence = float(np.max(probs))
            top_3_indices = np.argsort(probs)[::-1][:3]
            top_categories = [
                {"category": str(self.classes_[i]), "probability": round(float(probs[i]), 4)}
                for i in top_3_indices
            ]
        else:
            confidence = 1.0
            top_categories = [{"category": str(pred_idx), "probability": 1.0}]

        explanation = (
            f"Candidate profile classified as '{pred_idx}' with {confidence * 100:.1f}% confidence "
            f"based on matching domain-specific vocabulary and skills."
        )

        return {
            "predicted_category": str(pred_idx),
            "confidence": round(confidence, 4),
            "top_categories": top_categories,
            "explanation": explanation,
        }

    def save(
        self,
        model_path: Optional[Path] = None,
        vec_path: Optional[Path] = None,
    ) -> None:
        """Serialize model and vectorizer artifacts using joblib."""
        m_path = model_path or CLASSIFIER_PATH
        v_path = vec_path or CLASSIFIER_VEC_PATH
        m_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, m_path)
        joblib.dump(self.vectorizer, v_path)
        print(f"Saved classifier to {m_path}")
        print(f"Saved classifier vectorizer to {v_path}")

    @classmethod
    def load(
        cls,
        model_path: Optional[Path] = None,
        vec_path: Optional[Path] = None,
    ) -> "ResumeClassifier":
        """Load pre-trained model and vectorizer artifacts from disk."""
        m_path = model_path or CLASSIFIER_PATH
        v_path = vec_path or CLASSIFIER_VEC_PATH
        if not m_path.exists() or not v_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {m_path} or {v_path}")
        model = joblib.load(m_path)
        vectorizer = joblib.load(v_path)
        return cls(vectorizer=vectorizer, model=model)


def train_and_evaluate_classifier() -> Dict[str, object]:
    """Train both Logistic Regression and Linear SVM, evaluate real metrics, and persist the best artifact."""
    if not RESUMES_PROCESSED_PATH.exists():
        raise FileNotFoundError(f"Processed resumes not found at {RESUMES_PROCESSED_PATH}. Run preprocess.py first.")

    df = pd.read_csv(RESUMES_PROCESSED_PATH)
    texts = df["cleaned_text"].tolist()
    labels = df["category"].tolist()

    # Stratified Train/Test Split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    # 1. Train and Evaluate Logistic Regression
    print("=" * 60)
    print("Training Model 1: Logistic Regression (TF-IDF)")
    clf_lr = ResumeClassifier()
    clf_lr.fit(X_train_raw, y_train, model_type="logistic_regression")
    X_test_vec_lr = clf_lr.vectorizer.transform(X_test_raw)
    y_pred_lr = clf_lr.model.predict(X_test_vec_lr)
    metrics_lr = evaluate_classifier(
        np.array(y_test),
        y_pred_lr,
        class_names=sorted(list(set(labels))),
        save_cm_plot=True,
        plot_filename="confusion_matrix.png",
    )
    print(f"Logistic Regression Accuracy: {metrics_lr['accuracy']:.4f}")
    print(f"Logistic Regression Weighted F1: {metrics_lr['f1_weighted']:.4f}")

    # 2. Train and Evaluate Linear SVM (Calibrated)
    print("=" * 60)
    print("Training Model 2: Linear SVM (Calibrated, TF-IDF)")
    clf_svm = ResumeClassifier()
    clf_svm.fit(X_train_raw, y_train, model_type="linear_svm")
    X_test_vec_svm = clf_svm.vectorizer.transform(X_test_raw)
    y_pred_svm = clf_svm.model.predict(X_test_vec_svm)
    metrics_svm = evaluate_classifier(
        np.array(y_test),
        y_pred_svm,
        class_names=sorted(list(set(labels))),
        save_cm_plot=False,
    )
    print(f"Linear SVM Accuracy: {metrics_svm['accuracy']:.4f}")
    print(f"Linear SVM Weighted F1: {metrics_svm['f1_weighted']:.4f}")

    # Choose the final primary model (Logistic Regression per spec or highest F1)
    primary_clf = clf_lr
    primary_clf.save()

    return {
        "logistic_regression": metrics_lr,
        "linear_svm": metrics_svm,
        "classes": sorted(list(set(labels))),
    }


if __name__ == "__main__":
    train_and_evaluate_classifier()
