"""Data preprocessing pipeline for resumes and job postings."""

import sys
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from src.config import (
    JOB_CORPUS_TARGET_SIZE,
    JOB_METADATA_PATH,
    JOB_SCHEMA,
    JOBS_PROCESSED_PATH,
    RANDOM_STATE,
    RESUME_SCHEMA,
    RESUMES_PROCESSED_PATH,
)
from src.data.load_data import load_raw_jobs, load_raw_resumes
from src.features.text_features import clean_text, extract_skills


def preprocess_resumes(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Clean and structure the raw resume dataset."""
    df = raw_df.copy()
    df = df.rename(columns={"Category": "category", "Resume": "resume_text"})

    # Drop nulls
    df = df.dropna(subset=["category", "resume_text"])
    df["category"] = df["category"].astype(str).str.strip()

    # Clean text
    df["cleaned_text"] = df["resume_text"].apply(clean_text)

    # Filter out empty or too short texts (< 15 words)
    df = df[df["cleaned_text"].str.split().str.len() >= 15].copy().reset_index(drop=True)

    return df[RESUME_SCHEMA]


def preprocess_jobs(raw_df: pd.DataFrame, target_size: int = JOB_CORPUS_TARGET_SIZE) -> pd.DataFrame:
    """Normalize, clean, deduplicate, and vectorize job postings."""
    df = raw_df.copy()

    # Map available column names to standard schema
    col_mapping = {
        "jobtitle": "title",
        "company": "company",
        "joblocation_address": "location",
        "skills": "skills",
        "jobdescription": "description",
        "experience": "experience",
        "uniq_id": "job_id",
    }
    for old_col, new_col in col_mapping.items():
        if old_col in df.columns:
            df = df.rename(columns={old_col: new_col})

    for col in ["title", "company", "location", "skills", "description", "experience"]:
        if col not in df.columns:
            df[col] = ""

    # Clean text columns
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["company"] = df["company"].fillna("Confidential").astype(str).str.strip()
    df["location"] = df["location"].fillna("Pan-India / Remote").astype(str).str.strip()
    df["skills_raw"] = df["skills"].fillna("").astype(str).str.strip()
    df["description"] = df["description"].fillna("").astype(str).str.strip()
    df["experience"] = df["experience"].fillna("Not Specified").astype(str).str.strip()
    df["source"] = "Naukri"

    # Drop records where either title or description is missing/empty
    df = df[(df["title"] != "") & (df["description"] != "")].copy()

    # Deduplicate before intensive processing
    df = df.drop_duplicates(subset=["title", "company", "description"]).copy()

    # Sample target size now for performance and reproducibility
    if len(df) > target_size:
        df = df.sample(n=target_size, random_state=RANDOM_STATE).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # Extract deterministic normalized skills from both skills_raw and description
    print(f"Extracting normalized skills for {len(df)} jobs...")
    def extract_combined_skills(row) -> str:
        combined_text = f"{row['title']} {row['skills_raw']} {row['description']}"
        skills_list = extract_skills(combined_text)
        return ", ".join(skills_list)

    df["skills"] = df.apply(extract_combined_skills, axis=1)

    # Build cleaned combined text for TF-IDF
    def build_cleaned_corpus_text(row) -> str:
        text = f"{row['title']} {row['title']} {row['skills']} {row['description']}"
        return clean_text(text)

    df["cleaned_text"] = df.apply(build_cleaned_corpus_text, axis=1)

    # Filter out entries with inadequate text length (< 10 words)
    df = df[df["cleaned_text"].str.split().str.len() >= 10].copy().reset_index(drop=True)
    df["job_id"] = [f"job_{i+1:05d}" for i in range(len(df))]

    return df[JOB_SCHEMA]


def build_processed_datasets() -> None:
    """Run full pipeline: load raw, clean, validate, and save processed artifacts."""
    print("Loading raw resume dataset...")
    raw_resumes = load_raw_resumes()
    resumes_clean = preprocess_resumes(raw_resumes)
    RESUMES_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    resumes_clean.to_csv(RESUMES_PROCESSED_PATH, index=False)
    print(f"Saved cleaned resumes to {RESUMES_PROCESSED_PATH}: {resumes_clean.shape}")

    print("Loading raw jobs dataset...")
    raw_jobs = load_raw_jobs()
    jobs_clean = preprocess_jobs(raw_jobs)
    JOBS_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    jobs_clean.to_csv(JOBS_PROCESSED_PATH, index=False)
    print(f"Saved cleaned jobs to {JOBS_PROCESSED_PATH}: {jobs_clean.shape}")

    # Also save job metadata pickle for fast Streamlit loading
    JOB_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(jobs_clean, JOB_METADATA_PATH)
    print(f"Saved job metadata to {JOB_METADATA_PATH}")


if __name__ == "__main__":
    build_processed_datasets()
