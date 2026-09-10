"""Core Engine — Content-Based Job Recommendation System using TF-IDF & Cosine Similarity."""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    JOB_MATRIX_PATH,
    JOB_METADATA_PATH,
    JOB_VEC_PATH,
    JOBS_PROCESSED_PATH,
    TFIDF_MAX_FEATURES_JOBS,
    TFIDF_NGRAM_RANGE,
)
from src.features.match_features import compute_composite_fit_score, compute_skill_gap
from src.features.text_features import clean_text, extract_skills


class JobRecommender:
    """Content-based job recommendation engine with cached TF-IDF matrix and explainability."""

    def __init__(
        self,
        vectorizer: Optional[TfidfVectorizer] = None,
        job_matrix: Optional[Union[np.ndarray, spmatrix]] = None,
        jobs_df: Optional[pd.DataFrame] = None,
    ):
        self.vectorizer = vectorizer
        self.job_matrix = job_matrix
        self.jobs_df = jobs_df

    def fit_and_index(self, jobs_df: pd.DataFrame) -> "JobRecommender":
        """Fit the TF-IDF vectorizer on the preprocessed job corpus and build the search matrix."""
        self.jobs_df = jobs_df.copy().reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES_JOBS,
            ngram_range=TFIDF_NGRAM_RANGE,
            sublinear_tf=True,
            stop_words="english",
        )
        self.job_matrix = self.vectorizer.fit_transform(self.jobs_df["cleaned_text"].tolist())
        return self

    def recommend(
        self,
        resume_text: str,
        top_n: int = 5,
        target_role: Optional[str] = None,
        location: Optional[str] = None,
        candidate_exp_years: Optional[float] = None,
        candidate_category: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        """Rank and return top-N job recommendations for a candidate's resume.
        
        Parameters
        ----------
        resume_text : str
            Raw or cleaned resume text.
        top_n : int
            Number of recommendations to return (5, 10, 15).
        target_role : Optional[str]
            Optional role/keyword filter.
        location : Optional[str]
            Optional location filter.
        candidate_exp_years : Optional[float]
            Optional candidate experience in years.
        candidate_category : Optional[str]
            Optional candidate domain category.
            
        Returns
        -------
        List[Dict[str, object]]
            Ranked list of job recommendation dictionaries with skill gap analysis.
        """
        if self.vectorizer is None or self.job_matrix is None or self.jobs_df is None:
            raise RuntimeError("Recommender is not initialized or indexed. Call load() or fit_and_index().")

        cleaned_resume = clean_text(resume_text)
        if not cleaned_resume:
            return []

        # Candidate skills
        candidate_skills = extract_skills(resume_text)

        # Vectorize input resume
        resume_vec = self.vectorizer.transform([cleaned_resume])

        # Compute cosine similarities across the job corpus
        sim_scores = cosine_similarity(resume_vec, self.job_matrix)[0]

        # Candidate pool indices
        candidate_indices = np.arange(len(self.jobs_df))

        # Apply optional filters if specified
        if target_role and target_role.strip():
            role_kw = target_role.strip().lower()
            role_mask = (
                self.jobs_df["title"].str.lower().str.contains(role_kw, regex=False) |
                self.jobs_df["skills"].str.lower().str.contains(role_kw, regex=False)
            ).values
            if np.any(role_mask):
                candidate_indices = candidate_indices[role_mask[candidate_indices]]

        if location and location.strip() and location.lower() != "all":
            loc_kw = location.strip().lower()
            loc_mask = self.jobs_df["location"].str.lower().str.contains(loc_kw, regex=False).values
            if np.any(loc_mask):
                candidate_indices = candidate_indices[loc_mask[candidate_indices]]

        # Sort candidate indices by similarity descending
        sub_scores = sim_scores[candidate_indices]
        sorted_order = np.argsort(sub_scores)[::-1]
        sorted_indices = candidate_indices[sorted_order]

        # Select top-N unique recommendations
        results = []
        seen_keys = set()

        for idx in sorted_indices:
            row = self.jobs_df.iloc[idx]
            dedup_key = (row["title"].strip().lower(), row["company"].strip().lower())
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            sim_val = float(sim_scores[idx])
            job_skills_list = [s.strip() for s in row["skills"].split(",") if s.strip()]

            # Compute skill gap
            gap_report = compute_skill_gap(candidate_skills, job_skills_list)

            # Compute composite fit
            fit_report = compute_composite_fit_score(
                text_similarity=sim_val,
                skill_coverage_pct=float(gap_report["skill_coverage_pct"]),
                candidate_exp_years=candidate_exp_years,
                job_exp_str=str(row["experience"]),
                candidate_category=candidate_category,
                job_category=str(row["title"]),
            )

            # Explainability summary
            matched_str = ", ".join(gap_report["matched_skills"][:4]) if gap_report["matched_skills"] else "general industry terminology"
            explanation = (
                f"Ranked high due to {sim_val * 100:.1f}% contextual similarity with "
                f"{gap_report['skill_coverage_pct']}% skill alignment "
                f"(overlapping in {matched_str})."
            )

            results.append({
                "rank": len(results) + 1,
                "job_id": row["job_id"],
                "title": row["title"],
                "company": row["company"],
                "location": row["location"],
                "experience": row["experience"],
                "skills": job_skills_list,
                "description": row["description"],
                "source": row["source"],
                "similarity_score": round(sim_val, 4),
                "match_percentage": round(sim_val * 100.0, 1),
                "matched_skills": gap_report["matched_skills"],
                "missing_skills": gap_report["missing_skills"],
                "skill_coverage_pct": gap_report["skill_coverage_pct"],
                "composite_fit": fit_report,
                "explanation": explanation,
            })

            if len(results) >= top_n:
                break

        return results

    def save(
        self,
        vec_path: Optional[Path] = None,
        matrix_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ) -> None:
        """Serialize recommender artifacts using joblib."""
        v_path = vec_path or JOB_VEC_PATH
        m_path = matrix_path or JOB_MATRIX_PATH
        meta_path = metadata_path or JOB_METADATA_PATH

        v_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, v_path)
        joblib.dump(self.job_matrix, m_path)
        joblib.dump(self.jobs_df, meta_path)
        print(f"Saved job vectorizer to {v_path}")
        print(f"Saved job matrix to {m_path}")
        print(f"Saved job metadata to {meta_path}")

    @classmethod
    def load(
        cls,
        vec_path: Optional[Path] = None,
        matrix_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ) -> "JobRecommender":
        """Load indexed recommender artifacts from disk."""
        v_path = vec_path or JOB_VEC_PATH
        m_path = matrix_path or JOB_MATRIX_PATH
        meta_path = metadata_path or JOB_METADATA_PATH

        if not v_path.exists() or not m_path.exists() or not meta_path.exists():
            raise FileNotFoundError("Recommender artifacts missing. Run build_recommender_index() first.")

        vectorizer = joblib.load(v_path)
        job_matrix = joblib.load(m_path)
        jobs_df = joblib.load(meta_path)
        return cls(vectorizer=vectorizer, job_matrix=job_matrix, jobs_df=jobs_df)


def build_and_save_recommender() -> JobRecommender:
    """Load cleaned jobs, fit vectorizer, build index, and save artifacts."""
    if not JOBS_PROCESSED_PATH.exists():
        raise FileNotFoundError(f"Clean jobs not found at {JOBS_PROCESSED_PATH}. Run preprocess.py first.")

    jobs_df = pd.read_csv(JOBS_PROCESSED_PATH)
    print(f"Building TF-IDF recommender matrix for {len(jobs_df)} jobs...")
    recommender = JobRecommender()
    recommender.fit_and_index(jobs_df)
    recommender.save()
    print("Recommender index built and saved successfully!")
    return recommender


if __name__ == "__main__":
    build_and_save_recommender()
