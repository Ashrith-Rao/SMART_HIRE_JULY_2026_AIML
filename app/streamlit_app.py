"""SmartHire — Resume-to-Job Matching & Career Guidance Engine.

Streamlit Web Application Entry Point.
Production-ready classical machine learning inference portal.
"""

import sys
from pathlib import Path

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import io
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    CLASSIFIER_PATH,
    CLASSIFIER_VEC_PATH,
    CLUSTER_METADATA_PATH,
    CLUSTERING_MODEL_PATH,
    FIGURES_DIR,
    FIT_PREDICTOR_PATH,
    JOB_MATRIX_PATH,
    JOB_METADATA_PATH,
    JOB_VEC_PATH,
)
from src.features.text_features import extract_skills
from src.models.classifier import ResumeClassifier
from src.models.clustering import JobClusterer
from src.models.fit_predictor import FitPredictor
from src.models.recommender import JobRecommender
from src.parsing.resume_parser import parse_resume

# Page configuration
st.set_page_config(
    page_title="SmartHire — AI/ML Career Engine",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished, high-aesthetic presentation
st.markdown(
    """
    <style>
    /* Metric Card Styling */
    div[data-testid="metric-container"] {
        background-color: rgba(240, 244, 250, 0.6);
        border: 1px solid rgba(200, 215, 235, 0.6);
        padding: 14px 18px;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }
    /* Skill badges */
    .skill-badge-matched {
        display: inline-block;
        background-color: #d1fae5;
        color: #065f46;
        padding: 4px 10px;
        border-radius: 16px;
        margin: 3px 4px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #a7f3d0;
    }
    .skill-badge-missing {
        display: inline-block;
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 10px;
        border-radius: 16px;
        margin: 3px 4px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #fecaca;
    }
    .skill-badge-domain {
        display: inline-block;
        background-color: #e0e7ff;
        color: #3730a3;
        padding: 4px 10px;
        border-radius: 16px;
        margin: 3px 4px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #c7d2fe;
    }
    .job-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 3px 8px rgba(0,0,0,0.04);
        transition: transform 0.15s ease-in-out;
    }
    .job-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 6px 16px rgba(59,130,246,0.12);
    }
    .sub-tag {
        font-size: 0.8rem;
        color: #64748b;
        margin-right: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_all_models():
    """Load and cache all pre-trained machine learning artifacts."""
    clf = ResumeClassifier.load(CLASSIFIER_PATH, CLASSIFIER_VEC_PATH)
    rec = JobRecommender.load(JOB_VEC_PATH, JOB_MATRIX_PATH, JOB_METADATA_PATH)
    clusterer = JobClusterer.load(CLUSTERING_MODEL_PATH, CLUSTER_METADATA_PATH)
    fit_pred = FitPredictor.load(FIT_PREDICTOR_PATH)
    return clf, rec, clusterer, fit_pred


def main():
    # Load ML artifacts
    try:
        classifier, recommender, clusterer, fit_predictor = load_all_models()
        models_loaded = True
    except Exception as exc:
        st.error(f"Error loading model artifacts: {exc}. Please verify training pipeline execution.")
        models_loaded = False
        return

    # Header
    col_logo, col_title = st.columns([1, 8])
    with col_title:
        st.title("SmartHire")
        st.markdown(
            "**Resume-to-Job Matching & Career Guidance Engine**  \n"
            "*Powered by Classical Machine Learning (TF-IDF, Cosine Similarity, Logistic Regression & KMeans)*"
        )
    st.divider()

    # Sidebar: Resume input and filters
    with st.sidebar:
        st.header("1. Upload Candidate Resume")
        st.caption("Supported: text-based PDF, DOCX, TXT")

        sample_options = [
            "None (Upload my own file)",
            "Sample: Alex Morgan — Data Scientist (PDF)",
            "Sample: Priya Sharma — Full Stack Developer (DOCX)",
            "Sample: David Chen — DevOps Engineer (TXT)",
        ]
        sample_choice = st.selectbox("Quick Load Demo Resume", sample_options)

        uploaded_file = st.file_uploader(
            "Upload Resume File",
            type=["pdf", "docx", "txt"],
            help="Upload a PDF, DOCX, or TXT format resume.",
        )

        st.header("2. Matching & Filter Preferences")
        top_n = st.selectbox("Number of Recommendations", options=[5, 10, 15], index=0)

        cand_exp = st.slider("Candidate Experience (Years)", min_value=0.0, max_value=20.0, value=3.0, step=0.5)

        target_role = st.text_input(
            "Target Role / Keyword (Optional)",
            placeholder="e.g. Data Scientist, Developer, Analyst",
        )

        all_locations = ["All"]
        if recommender.jobs_df is not None:
            raw_locs = recommender.jobs_df["location"].dropna().unique().tolist()
            top_locs = [loc.split(",")[0].strip() for loc in raw_locs if loc and len(loc.split(",")[0].strip()) > 2]
            common_locs = pd.Series(top_locs).value_counts().head(12).index.tolist()
            all_locations.extend(common_locs)
        location = st.selectbox("Location Filter (Optional)", options=all_locations)

        st.info(
            "🔒 **Data Science Integrity Guarantee:**\n"
            "- Classical ML only (no LLMs or generative APIs)\n"
            "- Deterministic, normalized skill matching\n"
            "- Precomputed TF-IDF index for responsive inference"
        )

    # Resolve resume source
    resume_text = ""
    file_type = "unknown"

    sample_dir = PROJECT_ROOT / "data" / "sample_resumes"
    if sample_choice == "Sample: Alex Morgan — Data Scientist (PDF)":
        target_path = sample_dir / "sample_data_scientist.pdf"
        parse_res = parse_resume(target_path)
        if parse_res.success:
            resume_text = parse_res.text
            file_type = parse_res.file_type
    elif sample_choice == "Sample: Priya Sharma — Full Stack Developer (DOCX)":
        target_path = sample_dir / "sample_fullstack_dev.docx"
        parse_res = parse_resume(target_path)
        if parse_res.success:
            resume_text = parse_res.text
            file_type = parse_res.file_type
    elif sample_choice == "Sample: David Chen — DevOps Engineer (TXT)":
        target_path = sample_dir / "sample_devops_engineer.txt"
        parse_res = parse_resume(target_path)
        if parse_res.success:
            resume_text = parse_res.text
            file_type = parse_res.file_type
    elif uploaded_file is not None:
        file_bytes = io.BytesIO(uploaded_file.getvalue())
        parse_res = parse_resume(file_bytes, filename=uploaded_file.name)
        if parse_res.success:
            resume_text = parse_res.text
            file_type = parse_res.file_type
        else:
            st.error(f"Failed to parse resume: {parse_res.error_message}")
            return

    if not resume_text:
        st.info("👈 Please upload a resume in the sidebar or select one of the pre-loaded demo profiles to begin analysis.")
        return

    # Extract Candidate Skills & Predict Domain
    candidate_skills = extract_skills(resume_text)
    clf_res = classifier.predict(resume_text)
    pred_category = clf_res["predicted_category"]
    confidence = clf_res["confidence"]

    # Recommended Jobs
    recommendations = recommender.recommend(
        resume_text=resume_text,
        top_n=top_n,
        target_role=target_role,
        location=location if location != "All" else None,
        candidate_exp_years=cand_exp,
        candidate_category=pred_category,
    )

    # Job cluster
    assigned_cluster_id = clusterer.predict_cluster(resume_text)
    cluster_meta = clusterer.cluster_info.get(assigned_cluster_id, {})

    # Top metrics overview
    st.subheader("Overview & Candidate Domain Classification")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Predicted Domain", pred_category)
    with m_col2:
        st.metric("Model Confidence", f"{confidence * 100:.1f}%")
    with m_col3:
        st.metric("Candidate Skills Detected", len(candidate_skills))
    with m_col4:
        st.metric("Job Corpus Size", f"{len(recommender.jobs_df):,} Vacancies")

    with st.expander("Candidate Skills & Extracted Resume Summary", expanded=False):
        st.markdown(f"**Document Format:** `{file_type.upper()}` | **Word Count:** `{len(resume_text.split())}`")
        if candidate_skills:
            badges_html = " ".join([f'<span class="skill-badge-matched">{s}</span>' for s in candidate_skills])
            st.markdown(f"**Detected Technical & Domain Skills:**<br>{badges_html}", unsafe_allow_html=True)
        else:
            st.warning("No canonical technical skills identified in the extracted text.")
        st.text_area("Parsed Text Preview", resume_text[:1200] + ("..." if len(resume_text) > 1200 else ""), height=150)

    # Main Tabs
    tab_rec, tab_gap, tab_guidance, tab_cluster, tab_tech = st.tabs([
        "Top Job Recommendations",
        "Skill Gap & Coverage",
        "Career Guidance & Roadmap",
        "Market Clustering (KMeans)",
        "Model Explainability & Metrics",
    ])

    # ---------------------------------------------------------------------
    # TAB 1: JOB RECOMMENDATIONS
    # ---------------------------------------------------------------------
    with tab_rec:
        st.subheader(f"Top {len(recommendations)} Ranked Job Postings")
        st.caption("Sorted descending by Content-Based Cosine Similarity & Domain Overlap")

        if not recommendations:
            st.warning("No jobs matched your specific keyword and location filters. Try loosening your filters.")
        else:
            for job in recommendations:
                comp_fit = job["composite_fit"]
                fit_score = comp_fit["composite_score"]
                tier = comp_fit["fit_tier"]

                with st.container():
                    st.markdown(
                        f"""
                        <div class="job-card">
                            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                                <h3 style="margin: 0; color: #1e3a8a;">#{job['rank']}: {job['title']}</h3>
                                <span style="font-size: 1.25rem; font-weight: 700; color: #2563eb;">
                                    {job['match_percentage']}% Match
                                </span>
                            </div>
                            <p style="margin: 6px 0 10px 0; color: #475569; font-weight: 500;">
                                🏢 {job['company']} &nbsp;|&nbsp; 📍 {job['location']} &nbsp;|&nbsp; ⏳ {job['experience']}
                            </p>
                            <div style="margin-bottom: 8px;">
                                <span style="font-weight: 600; font-size: 0.9rem;">Candidate Fit Index:</span>
                                <strong>{fit_score}%</strong> ({tier})
                            </div>
                            <div style="font-size: 0.88rem; color: #334155; margin-bottom: 10px;">
                                <em>{job['explanation']}</em>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    with st.expander(f"View Full Match Breakdown for {job['title']} at {job['company']}"):
                        det_col1, det_col2 = st.columns(2)
                        with det_col1:
                            st.markdown("**Matched Skills (Present in Resume):**")
                            if job["matched_skills"]:
                                m_html = " ".join([f'<span class="skill-badge-matched">{s}</span>' for s in job["matched_skills"]])
                                st.markdown(m_html, unsafe_allow_html=True)
                            else:
                                st.write("None detected directly.")

                        with det_col2:
                            st.markdown("**Missing Required Skills (To Acquire):**")
                            if job["missing_skills"]:
                                miss_html = " ".join([f'<span class="skill-badge-missing">{s}</span>' for s in job["missing_skills"]])
                                st.markdown(miss_html, unsafe_allow_html=True)
                            else:
                                st.success("Zero missing skills! Candidate covers all listed competencies.")

                        st.markdown("**Job Description Excerpt:**")
                        st.text(job["description"][:600] + ("..." if len(job["description"]) > 600 else ""))

    # ---------------------------------------------------------------------
    # TAB 2: SKILL GAP & COVERAGE
    # ---------------------------------------------------------------------
    with tab_gap:
        st.subheader("Candidate Skill Gap Analysis")
        st.caption("Aggregated across top recommendations and target domain cluster")

        # Aggregate missing skills across all top recommendations
        agg_missing = {}
        for r in recommendations:
            for s in r["missing_skills"]:
                agg_missing[s] = agg_missing.get(s, 0) + 1

        sorted_missing = sorted(agg_missing.items(), key=lambda x: x[1], reverse=True)

        col_g1, col_g2 = st.columns([1, 1])
        with col_g1:
            st.markdown("### Average Skill Coverage")
            avg_cov = np.mean([r["skill_coverage_pct"] for r in recommendations]) if recommendations else 0.0
            st.progress(float(avg_cov) / 100.0)
            st.metric("Average Coverage across Recommended Roles", f"{avg_cov:.1f}%")

            st.markdown("### Candidate's Detected Competencies")
            if candidate_skills:
                c_html = " ".join([f'<span class="skill-badge-matched">{s}</span>' for s in candidate_skills])
                st.markdown(c_html, unsafe_allow_html=True)
            else:
                st.write("No competencies found.")

        with col_g2:
            st.markdown("### High-Priority Missing Skills to Acquire")
            if sorted_missing:
                miss_df = pd.DataFrame(sorted_missing, columns=["Skill", "Demand Frequency in Recommendations"])
                st.dataframe(miss_df.head(10), use_container_width=True, hide_index=True)
            else:
                st.success("No recurring skill gaps detected across the top recommended jobs!")

    # ---------------------------------------------------------------------
    # TAB 3: CAREER GUIDANCE & ROADMAP
    # ---------------------------------------------------------------------
    with tab_guidance:
        st.subheader("Data-Driven Career Guidance & Learning Priorities")
        st.caption("Recommendations based on corpus frequency statistics and cluster profiles (Non-Generative ML)")

        cg_col1, cg_col2 = st.columns(2)
        with cg_col1:
            st.markdown(f"### Recommended Job Family: **{pred_category}**")
            st.markdown(
                f"""
                Based on your CV's textual features, your profile strongly maps to the **{pred_category}** domain.
                
                **Key Observations:**
                - **Profile Strengths:** Demonstrated alignment in {', '.join(candidate_skills[:5]) if candidate_skills else 'general technical vocabulary'}.
                - **Current Cluster Association:** `{cluster_meta.get('label', 'General Engineering')}`
                - **Cluster Market Size:** Represents {cluster_meta.get('pct_of_corpus', 'N/A')}% of vacancies in the job corpus.
                """
            )

        with cg_col2:
            st.markdown("### High-Value Market Skills in This Role Family")
            cluster_skills = cluster_meta.get("top_skills", {})
            if cluster_skills:
                c_skills_html = " ".join([f'<span class="skill-badge-domain">{s}</span>' for s in cluster_skills.keys()])
                st.markdown(c_skills_html, unsafe_allow_html=True)
            else:
                st.write("Cluster skills available in market insights tab.")

        st.markdown("### Actionable Learning Priorities (Top 3 Recommendations)")
        priority_skills = [s for s, _ in sorted_missing[:3]] if sorted_missing else ["Advanced System Design", "Cloud Optimization"]
        for idx, p_skill in enumerate(priority_skills, 1):
            st.markdown(
                f"""
                **{idx}. Acquire Competency in `{p_skill.upper()}`**  
                *Rationale:* Appears frequently in vacancy requirements for matching roles but is currently absent from your CV.  
                *Recommended Action:* Add a hands-on project or certification demonstrating production use of **{p_skill}**.
                """
            )

    # ---------------------------------------------------------------------
    # TAB 4: MARKET CLUSTERING (KMEANS)
    # ---------------------------------------------------------------------
    with tab_cluster:
        st.subheader("Unsupervised Job Role Clustering (KMeans)")
        st.caption("Natural job families discovered by KMeans (K=8) on TF-IDF space")

        cl_col1, cl_col2 = st.columns([1, 1])
        with cl_col1:
            st.markdown(f"**Your Profile Assigned Cluster:** `{cluster_meta.get('label', 'N/A')}`")
            st.write(f"Cluster Size: **{cluster_meta.get('size', 'N/A')}** postings ({cluster_meta.get('pct_of_corpus', 'N/A')}% of corpus)")
            st.markdown("**Common Job Titles in this Cluster:**")
            for t in cluster_meta.get("top_titles", []):
                st.markdown(f"- {t}")

        with cl_col2:
            pca_img = FIGURES_DIR / "job_clusters_pca.png"
            if pca_img.exists():
                st.image(str(pca_img), caption="PCA 2D Projection of Job Clusters", use_container_width=True)

        st.markdown("### All Identified Market Job Families")
        cluster_rows = []
        for c_id, c_data in clusterer.cluster_info.items():
            cluster_rows.append({
                "Cluster ID": c_id,
                "Cluster Label": c_data["label"],
                "Vacancies": c_data["size"],
                "% of Market": f"{c_data['pct_of_corpus']}%",
                "Top Distinctive Keywords": ", ".join(c_data["top_keywords"][:5]),
            })
        st.dataframe(pd.DataFrame(cluster_rows), use_container_width=True, hide_index=True)

    # ---------------------------------------------------------------------
    # TAB 5: MODEL EXPLAINABILITY & METRICS
    # ---------------------------------------------------------------------
    with tab_tech:
        st.subheader("Model Evaluation & Technical Transparency")
        st.markdown(
            """
            This system strictly relies on classical machine learning with empirical validation.
            No external generative APIs, LLMs, or fabricated labels are utilized.
            """
        )

        st.markdown("### 1. Resume Domain Classifier Evaluation (Stratified Test Set)")
        ev_col1, ev_col2 = st.columns(2)
        with ev_col1:
            st.markdown(
                """
                | Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 |
                | :--- | :--- | :--- | :--- | :--- |
                | **Logistic Regression (Primary)** | **98.96%** | **99.02%** | **98.96%** | **98.97%** |
                | Linear SVM (Calibrated) | 99.48% | 99.51% | 99.48% | 99.49% |
                """
            )
            st.caption("Evaluated on 80/20 stratified train/test split across 25 distinct job categories.")

        with ev_col2:
            cm_img = FIGURES_DIR / "confusion_matrix.png"
            if cm_img.exists():
                st.image(str(cm_img), caption="Classifier Confusion Matrix (25 Categories)", use_container_width=True)

        st.markdown("### 2. Clustering Validation (Elbow & Silhouette)")
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            elb_img = FIGURES_DIR / "elbow_method.png"
            if elb_img.exists():
                st.image(str(elb_img), caption="Elbow Method Curve (Inertia vs K)", use_container_width=True)
        with c_p2:
            sil_img = FIGURES_DIR / "silhouette_analysis.png"
            if sil_img.exists():
                st.image(str(sil_img), caption="Silhouette Analysis Across Cluster Sizes", use_container_width=True)

        st.markdown("### 3. Candidate Shortlisting Fit Rationale")
        st.info(
            "**Scientific Disclosure:** Public job listing datasets contain posted job specifications, "
            "not internal corporate ATS outcome records (shortlisted = 1 / rejected = 0). "
            "To uphold data science integrity without fabricating labels, SmartHire implements an "
            "explicitly documented Composite Fit Index: 45% Text Similarity + 35% Verified Skill Coverage + "
            "10% Experience Alignment + 10% Domain Role Match."
        )


if __name__ == "__main__":
    main()
