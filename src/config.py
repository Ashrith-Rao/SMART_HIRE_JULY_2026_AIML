"""Configuration and constants for SmartHire."""

from pathlib import Path

# Base Paths (platform-independent)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
APP_DIR = PROJECT_ROOT / "app"

# Raw Data Paths
RESUME_RAW_PATH = RAW_DATA_DIR / "UpdatedResumeDataSet.csv"
NAUKRI_RAW_PATH = RAW_DATA_DIR / "naukri_com-job_sample.csv"

# Processed Data Paths
RESUMES_PROCESSED_PATH = PROCESSED_DATA_DIR / "resumes_clean.csv"
JOBS_PROCESSED_PATH = PROCESSED_DATA_DIR / "jobs_clean.csv"
JOB_METADATA_PATH = MODELS_DIR / "job_metadata.pkl"

# Model Artifact Paths
CLASSIFIER_PATH = MODELS_DIR / "classifier.pkl"
CLASSIFIER_VEC_PATH = MODELS_DIR / "classifier_vectorizer.pkl"
JOB_VEC_PATH = MODELS_DIR / "job_vectorizer.pkl"
JOB_MATRIX_PATH = MODELS_DIR / "job_matrix.pkl"
CLUSTERING_MODEL_PATH = MODELS_DIR / "clustering_model.pkl"
CLUSTER_METADATA_PATH = MODELS_DIR / "cluster_metadata.pkl"
FIT_PREDICTOR_PATH = MODELS_DIR / "fit_predictor.pkl"

# Machine Learning Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.20
TFIDF_MAX_FEATURES_CLASSIFIER = 5000
TFIDF_MAX_FEATURES_JOBS = 10000
TFIDF_NGRAM_RANGE = (1, 2)
N_CLUSTERS = 8

# Job Corpus Target Size for high-performance inference
JOB_CORPUS_TARGET_SIZE = 4000

# Canonical Schema Definitions
JOB_SCHEMA = [
    "job_id",
    "title",
    "company",
    "location",
    "skills",
    "description",
    "experience",
    "source",
    "cleaned_text",
]

RESUME_SCHEMA = [
    "category",
    "resume_text",
    "cleaned_text",
]
