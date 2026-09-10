"""Skill gap computation and explainable candidate fit scoring."""

import re
from typing import Dict, List, Optional, Set, Tuple, Union
from src.features.text_features import normalize_skill


def compute_skill_gap(
    resume_skills: Union[List[str], Set[str]],
    job_skills: Union[List[str], Set[str]],
) -> Dict[str, Union[List[str], float, int]]:
    """Compute matched skills, missing skills, and skill coverage percentage.
    
    Parameters
    ----------
    resume_skills : Union[List[str], Set[str]]
        Normalized skills extracted from the candidate's resume.
    job_skills : Union[List[str], Set[str]]
        Normalized skills required or extracted from the target job posting.
        
    Returns
    -------
    dict
        {
            "matched_skills": List[str],
            "missing_skills": List[str],
            "skill_coverage_pct": float,
            "total_job_skills": int,
            "total_candidate_skills": int,
        }
    """
    res_set = {normalize_skill(s) for s in resume_skills if s and normalize_skill(s)}
    job_set = {normalize_skill(s) for s in job_skills if s and normalize_skill(s)}

    matched = sorted(list(res_set & job_set))
    missing = sorted(list(job_set - res_set))

    if not job_set:
        # If job lists no explicit skills, coverage cannot be penalized
        coverage_pct = 0.0 if not res_set else 100.0
    else:
        coverage_pct = round((len(matched) / len(job_set)) * 100.0, 1)

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "skill_coverage_pct": coverage_pct,
        "total_job_skills": len(job_set),
        "total_candidate_skills": len(res_set),
    }


def parse_experience_range(exp_str: Optional[str]) -> Tuple[float, float]:
    """Parse minimum and maximum years of experience from strings like '2 - 5 yrs', '3-7 Yrs', '5 yrs'."""
    if not isinstance(exp_str, str) or not exp_str.strip():
        return (0.0, 30.0)
    
    matches = re.findall(r"\b(\d+)\b", exp_str)
    if len(matches) >= 2:
        return (float(matches[0]), float(matches[1]))
    elif len(matches) == 1:
        val = float(matches[0])
        return (val, val + 2.0)
    return (0.0, 30.0)


def compute_composite_fit_score(
    text_similarity: float,
    skill_coverage_pct: float,
    candidate_exp_years: Optional[float] = None,
    job_exp_str: Optional[str] = None,
    candidate_category: Optional[str] = None,
    job_category: Optional[str] = None,
) -> Dict[str, Union[float, Dict[str, float], str]]:
    """Calculate an explainable composite fit score (0-100%) based on defensible features.
    
    Components:
    - Text Similarity (TF-IDF Cosine Similarity): 45% weight
    - Skill Coverage Percentage: 35% weight
    - Experience Compatibility: 10% weight
    - Domain Category Compatibility: 10% weight
    
    Returns
    -------
    dict
        {
            "composite_score": float,
            "fit_tier": str,  # High Fit, Moderate Fit, Developing Fit
            "components": {
                "text_similarity_contribution": float,
                "skill_coverage_contribution": float,
                "experience_contribution": float,
                "domain_match_contribution": float,
            },
            "explanation": str
        }
    """
    # 1. Similarity contribution (0-100 scale, weight 0.45)
    sim_clamped = max(0.0, min(1.0, float(text_similarity)))
    sim_score = sim_clamped * 100.0

    # 2. Skill coverage contribution (0-100 scale, weight 0.35)
    cov_clamped = max(0.0, min(100.0, float(skill_coverage_pct)))

    # 3. Experience compatibility (0-100 scale, weight 0.10)
    if candidate_exp_years is not None and job_exp_str:
        min_exp, max_exp = parse_experience_range(job_exp_str)
        if candidate_exp_years < min_exp:
            diff = min_exp - candidate_exp_years
            exp_score = max(0.0, 100.0 - (diff * 20.0))
        elif candidate_exp_years > max_exp:
            exp_score = 90.0  # Over-qualified slightly penalized or high
        else:
            exp_score = 100.0
    else:
        # Neutral baseline if experience is unspecified
        exp_score = 75.0

    # 4. Category compatibility (0-100 scale, weight 0.10)
    if candidate_category and job_category:
        cat_match = candidate_category.strip().lower() in job_category.strip().lower() or \
                    job_category.strip().lower() in candidate_category.strip().lower()
        cat_score = 100.0 if cat_match else 50.0
    else:
        cat_score = 75.0

    # Weighted Composite
    composite = (
        0.45 * sim_score +
        0.35 * cov_clamped +
        0.10 * exp_score +
        0.10 * cat_score
    )
    composite = round(max(0.0, min(100.0, composite)), 1)

    if composite >= 75.0:
        tier = "High Fit (Strong Shortlist Candidate)"
    elif composite >= 50.0:
        tier = "Moderate Fit (Competitive with Core Skills)"
    else:
        tier = "Developing Fit (Skill Gap to Bridge)"

    explanation = (
        f"Score reflects {sim_clamped * 100:.1f}% vocabulary/contextual similarity, "
        f"{cov_clamped:.1f}% mandatory skill coverage, "
        f"and {exp_score:.0f}% experience & domain alignment."
    )

    return {
        "composite_score": composite,
        "fit_tier": tier,
        "components": {
            "text_similarity_pct": round(sim_score, 1),
            "skill_coverage_pct": round(cov_clamped, 1),
            "experience_score": round(exp_score, 1),
            "domain_match_score": round(cat_score, 1),
        },
        "explanation": explanation,
    }
