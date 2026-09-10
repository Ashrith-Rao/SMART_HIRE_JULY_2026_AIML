"""Load raw resume and job dataset files with schema validation."""

from pathlib import Path
from typing import Optional
import pandas as pd
from src.config import NAUKRI_RAW_PATH, RESUME_RAW_PATH


def load_raw_resumes(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw UpdatedResumeDataSet.csv.
    
    Expected columns: 'Category', 'Resume'
    """
    file_path = path or RESUME_RAW_PATH
    if not file_path.exists():
        raise FileNotFoundError(f"Raw resume dataset not found at: {file_path}")

    df = pd.read_csv(file_path)
    required_cols = {"Category", "Resume"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Resume dataset missing required columns: {required_cols - set(df.columns)}. "
            f"Found columns: {df.columns.tolist()}"
        )
    return df


def load_raw_jobs(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw Naukri job listings CSV.
    
    Expected columns from Naukri dump: jobtitle, company, skills, jobdescription, etc.
    """
    file_path = path or NAUKRI_RAW_PATH
    if not file_path.exists():
        raise FileNotFoundError(f"Raw jobs dataset not found at: {file_path}")

    df = pd.read_csv(file_path, low_memory=False)
    return df
