"""Unsupervised Job Family Clustering using TF-IDF, KMeans, and PCA Visualization."""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from src.config import (
    CLUSTER_METADATA_PATH,
    CLUSTERING_MODEL_PATH,
    FIGURES_DIR,
    JOBS_PROCESSED_PATH,
    N_CLUSTERS,
    RANDOM_STATE,
)
from src.features.text_features import extract_skills


class JobClusterer:
    """KMeans clustering model for grouping job postings into natural role families."""

    def __init__(
        self,
        n_clusters: int = N_CLUSTERS,
        model: Optional[KMeans] = None,
        vectorizer: Optional[TfidfVectorizer] = None,
        cluster_info: Optional[Dict[int, Dict[str, object]]] = None,
        pca_model: Optional[PCA] = None,
    ):
        self.n_clusters = n_clusters
        self.model = model
        self.vectorizer = vectorizer
        self.cluster_info = cluster_info or {}
        self.pca_model = pca_model

    def fit(self, jobs_df: pd.DataFrame) -> "JobClusterer":
        """Fit TF-IDF and KMeans on job descriptions and extract cluster interpretations."""
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        X = self.vectorizer.fit_transform(jobs_df["cleaned_text"].tolist())

        print(f"Fitting KMeans with k={self.n_clusters}...")
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=RANDOM_STATE,
            n_init=10,
            max_iter=300,
        )
        labels = self.model.fit_predict(X)
        jobs_df_copy = jobs_df.copy()
        jobs_df_copy["cluster"] = labels

        # Extract top keywords, top titles, and frequent skills per cluster
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        order_centroids = self.model.cluster_centers_.argsort()[:, ::-1]

        self.cluster_info = {}
        for cluster_idx in range(self.n_clusters):
            top_keywords = feature_names[order_centroids[cluster_idx, :8]].tolist()
            cluster_subset = jobs_df_copy[jobs_df_copy["cluster"] == cluster_idx]

            # Most common job titles
            top_titles = cluster_subset["title"].value_counts().head(5).index.tolist()

            # Aggregate skills in this cluster
            all_skills = []
            for s_str in cluster_subset["skills"].dropna():
                for s in s_str.split(","):
                    s_clean = s.strip()
                    if s_clean:
                        all_skills.append(s_clean)
            skill_counts = pd.Series(all_skills).value_counts().head(8).to_dict()

            # Descriptive cluster label based on top titles and keywords
            label_name = f"Cluster {cluster_idx}: " + " / ".join(top_keywords[:3]).title()

            self.cluster_info[cluster_idx] = {
                "cluster_id": cluster_idx,
                "label": label_name,
                "size": int(len(cluster_subset)),
                "pct_of_corpus": round((len(cluster_subset) / len(jobs_df_copy)) * 100, 1),
                "top_keywords": top_keywords,
                "top_titles": top_titles,
                "top_skills": skill_counts,
            }

        # Fit 2D PCA for visual projection
        self.pca_model = PCA(n_components=2, random_state=RANDOM_STATE)
        pca_coords = self.pca_model.fit_transform(X.toarray())
        jobs_df_copy["pca_x"] = pca_coords[:, 0]
        jobs_df_copy["pca_y"] = pca_coords[:, 1]

        return self

    def predict_cluster(self, text: str) -> int:
        """Assign an input text (e.g. resume) to the closest job cluster."""
        if self.vectorizer is None or self.model is None:
            raise RuntimeError("Clusterer is not trained or loaded.")
        vec = self.vectorizer.transform([text])
        return int(self.model.predict(vec)[0])

    def save(
        self,
        model_path: Optional[Path] = None,
        meta_path: Optional[Path] = None,
    ) -> None:
        """Save clustering artifacts to disk."""
        m_path = model_path or CLUSTERING_MODEL_PATH
        meta_p = meta_path or CLUSTER_METADATA_PATH

        m_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "vectorizer": self.vectorizer,
            "pca_model": self.pca_model,
            "n_clusters": self.n_clusters,
        }
        joblib.dump(payload, m_path)
        joblib.dump(self.cluster_info, meta_p)
        print(f"Saved clustering model to {m_path}")
        print(f"Saved cluster metadata to {meta_p}")

    @classmethod
    def load(
        cls,
        model_path: Optional[Path] = None,
        meta_path: Optional[Path] = None,
    ) -> "JobClusterer":
        """Load clustering artifacts from disk."""
        m_path = model_path or CLUSTERING_MODEL_PATH
        meta_p = meta_path or CLUSTER_METADATA_PATH
        if not m_path.exists() or not meta_p.exists():
            raise FileNotFoundError(f"Clustering artifacts not found at {m_path} or {meta_p}")

        payload = joblib.load(m_path)
        cluster_info = joblib.load(meta_p)
        return cls(
            n_clusters=payload["n_clusters"],
            model=payload["model"],
            vectorizer=payload["vectorizer"],
            cluster_info=cluster_info,
            pca_model=payload.get("pca_model"),
        )


def analyze_and_build_clusters(
    k_min: int = 4,
    k_max: int = 12,
    selected_k: int = N_CLUSTERS,
) -> Tuple[JobClusterer, Dict[str, object]]:
    """Run elbow analysis and silhouette score computation across K, then fit and save selected clusterer."""
    if not JOBS_PROCESSED_PATH.exists():
        raise FileNotFoundError(f"Processed jobs not found at {JOBS_PROCESSED_PATH}")

    jobs_df = pd.read_csv(JOBS_PROCESSED_PATH)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    vec = TfidfVectorizer(max_features=2500, stop_words="english", sublinear_tf=True)
    X = vec.fit_transform(jobs_df["cleaned_text"].tolist())

    k_values = list(range(k_min, k_max + 1))
    inertias = []
    silhouettes = []

    print("Running Elbow and Silhouette analysis across K...")
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=5, max_iter=200)
        labels = km.fit_predict(X)
        inertias.append(float(km.inertia_))
        sil = float(silhouette_score(X, labels, sample_size=1500, random_state=RANDOM_STATE))
        silhouettes.append(sil)
        print(f"K={k}: Inertia={km.inertia_:.1f}, Silhouette={sil:.4f}")

    # Plot 1: Elbow Method (Inertia vs K)
    plt.figure(figsize=(9, 5))
    plt.plot(k_values, inertias, "bo-", linewidth=2, markersize=7)
    plt.title("Elbow Method for Optimal Job Clusters (Inertia vs K)", fontsize=13, fontweight="bold")
    plt.xlabel("Number of Clusters (K)", fontsize=11)
    plt.ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.axvline(x=selected_k, color="red", linestyle="--", label=f"Selected K = {selected_k}")
    plt.legend()
    plt.tight_layout()
    elbow_path = FIGURES_DIR / "elbow_method.png"
    plt.savefig(elbow_path, dpi=200)
    plt.close()
    print(f"Saved elbow plot to {elbow_path}")

    # Plot 2: Silhouette Score vs K
    plt.figure(figsize=(9, 5))
    plt.plot(k_values, silhouettes, "gs-", linewidth=2, markersize=7)
    plt.title("Silhouette Analysis Across Cluster Sizes", fontsize=13, fontweight="bold")
    plt.xlabel("Number of Clusters (K)", fontsize=11)
    plt.ylabel("Average Silhouette Coefficient", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.axvline(x=selected_k, color="red", linestyle="--", label=f"Selected K = {selected_k}")
    plt.legend()
    plt.tight_layout()
    sil_path = FIGURES_DIR / "silhouette_analysis.png"
    plt.savefig(sil_path, dpi=200)
    plt.close()
    print(f"Saved silhouette plot to {sil_path}")

    # Fit final clusterer with selected K
    clusterer = JobClusterer(n_clusters=selected_k)
    clusterer.fit(jobs_df)
    clusterer.save()

    # Plot 3: 2D PCA Visualization of Clusters
    labels = clusterer.model.labels_
    pca = clusterer.pca_model
    X_sample = clusterer.vectorizer.transform(jobs_df["cleaned_text"].tolist()[:1000]).toarray()
    pca_2d = pca.transform(X_sample)
    labels_sample = labels[:1000]

    plt.figure(figsize=(11, 7))
    scatter = plt.scatter(
        pca_2d[:, 0],
        pca_2d[:, 1],
        c=labels_sample,
        cmap="tab10",
        alpha=0.6,
        s=35,
        edgecolors="none",
    )
    plt.title(f"PCA 2D Projection of Job Postings (K={selected_k} Clusters)", fontsize=13, fontweight="bold")
    plt.xlabel("Principal Component 1", fontsize=11)
    plt.ylabel("Principal Component 2", fontsize=11)
    plt.colorbar(scatter, label="Cluster ID")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    pca_path = FIGURES_DIR / "job_clusters_pca.png"
    plt.savefig(pca_path, dpi=200)
    plt.close()
    print(f"Saved PCA cluster plot to {pca_path}")

    results = {
        "k_values": k_values,
        "inertias": inertias,
        "silhouettes": silhouettes,
        "selected_k": selected_k,
        "selected_silhouette": silhouettes[k_values.index(selected_k)],
        "cluster_info": clusterer.cluster_info,
    }
    return clusterer, results


if __name__ == "__main__":
    analyze_and_build_clusters()
