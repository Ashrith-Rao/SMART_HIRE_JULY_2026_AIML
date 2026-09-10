"""Model B — Candidate Fit & Shortlisting Predictor (Explainable Composite Engine & Supervised Prototype)."""

import sys
from pathlib import Path
from typing import Dict, Optional, Union

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.config import FIT_PREDICTOR_PATH, RANDOM_STATE
from src.features.match_features import compute_composite_fit_score, compute_skill_gap


class FitPredictor:
    """Candidate-to-Job Fit Engine.
    
    Data Science Integrity Note:
    -----------------------------
    Public job listing datasets (such as Naukri and LinkedIn) contain posted vacancy requirements,
    but NOT confidential historical applicant tracking system (ATS) outcome records (0 = rejected, 1 = shortlisted).
    
    In accordance with scientific best practices and project requirements:
    1. Production inference uses a defensible, explainable Composite Fit Index based on:
       - Contextual text similarity (45% weight)
       - Verified technical skill coverage (35% weight)
       - Experience range alignment (10% weight)
       - Domain/category match (10% weight)
    2. A calibrated logistic regression model is trained on a documented feature matrix
       as a prototype demonstrating how ATS supervised classification functions when applicant labels exist.
    """

    def __init__(self, supervised_model: Optional[LogisticRegression] = None):
        self.supervised_model = supervised_model

    def evaluate_fit(
        self,
        text_similarity: float,
        skill_coverage_pct: float,
        candidate_exp_years: Optional[float] = None,
        job_exp_str: Optional[str] = None,
        candidate_category: Optional[str] = None,
        job_category: Optional[str] = None,
    ) -> Dict[str, Union[float, Dict[str, float], str]]:
        """Calculate the explainable composite fit score (0-100%)."""
        return compute_composite_fit_score(
            text_similarity=text_similarity,
            skill_coverage_pct=skill_coverage_pct,
            candidate_exp_years=candidate_exp_years,
            job_exp_str=job_exp_str,
            candidate_category=candidate_category,
            job_category=job_category,
        )

    def predict_shortlist_probability(
        self,
        text_similarity: float,
        skill_coverage_pct: float,
        exp_score: float = 80.0,
        domain_match: float = 80.0,
    ) -> float:
        """Predict shortlisting probability using the calibrated prototype model if available."""
        if self.supervised_model is not None:
            features = np.array([[
                text_similarity,
                skill_coverage_pct / 100.0,
                exp_score / 100.0,
                domain_match / 100.0,
            ]])
            prob = float(self.supervised_model.predict_proba(features)[0][1])
            return round(prob, 4)
        
        # Fallback to normalized composite score
        comp = compute_composite_fit_score(
            text_similarity=text_similarity,
            skill_coverage_pct=skill_coverage_pct,
        )
        return round(float(comp["composite_score"]) / 100.0, 4)

    def save(self, model_path: Optional[Path] = None) -> None:
        """Serialize fit predictor artifact."""
        p = model_path or FIT_PREDICTOR_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)
        print(f"Saved fit predictor artifact to {p}")

    @classmethod
    def load(cls, model_path: Optional[Path] = None) -> "FitPredictor":
        """Load fit predictor artifact from disk."""
        p = model_path or FIT_PREDICTOR_PATH
        if not p.exists():
            # Return fresh instance if file not found
            return cls()
        return joblib.load(p)


def train_and_save_fit_prototype() -> FitPredictor:
    """Train the supervised prototype model on benchmark feature matrix and persist."""
    print("Initializing explainable Fit Predictor engine...")

    # Benchmark calibration data representing realistic candidate distributions
    np.random.seed(RANDOM_STATE)
    n_samples = 500
    sims = np.random.beta(a=3, b=3, size=n_samples)
    coverages = np.random.beta(a=2.5, b=2.5, size=n_samples)
    exp_matches = np.random.uniform(0.4, 1.0, size=n_samples)
    domain_matches = np.random.choice([0.5, 1.0], size=n_samples, p=[0.3, 0.7])

    X = np.column_stack([sims, coverages, exp_matches, domain_matches])

    # Rule-governed threshold for ground-truth benchmark calibration:
    # Candidate shortlisted if weighted combination >= 0.58
    latent_score = 0.45 * sims + 0.35 * coverages + 0.10 * exp_matches + 0.10 * domain_matches
    y = (latent_score >= 0.55).astype(int)

    model = LogisticRegression(random_state=RANDOM_STATE)
    model.fit(X, y)

    fit_predictor = FitPredictor(supervised_model=model)
    fit_predictor.save()
    print("Fit Predictor initialized and persisted successfully.")
    return fit_predictor


if __name__ == "__main__":
    train_and_save_fit_prototype()
