"""NLP text preprocessing, token cleaning, and deterministic skill extraction."""

import re
from typing import Dict, List, Set

# Canonical Skill Lexicon grouped by domain
CANONICAL_SKILLS: Set[str] = {
    # Data Science & Machine Learning
    "python", "r", "machine learning", "deep learning", "natural language processing",
    "computer vision", "data analysis", "data science", "statistics", "data mining",
    "scikit-learn", "tensorflow", "pytorch", "keras", "pandas", "numpy", "scipy",
    "matplotlib", "seaborn", "plotly", "tableau", "power bi", "excel", "bigquery",
    "spark", "hadoop", "hive", "kafka", "dbt", "airflow", "sql", "nosql",
    "xgboost", "lightgbm", "spacy", "nltk", "opencv", "feature engineering",
    "time series", "ab testing", "predictive modeling", "clustering",

    # Web & Software Development
    "java", "c++", "c#", ".net", "javascript", "typescript", "html", "css",
    "react.js", "angular", "vue.js", "node.js", "express.js", "django", "flask",
    "fastapi", "spring boot", "ruby", "rails", "php", "laravel", "go", "rust",
    "scala", "kotlin", "swift", "rest api", "graphql", "microservices", "soap",
    "redux", "bootstrap", "tailwind", "next.js", "jquery",

    # Databases & Storage
    "postgresql", "mysql", "oracle", "sql server", "sqlite", "mongodb", "redis",
    "cassandra", "elasticsearch", "dynamodb", "snowflake", "redshift",

    # Cloud, DevOps & Infrastructure
    "aws", "azure", "google cloud", "docker", "kubernetes", "terraform",
    "ansible", "jenkins", "git", "github", "gitlab", "ci/cd", "linux", "bash",
    "powershell", "devops", "cloud computing", "serverless", "prometheus", "grafana",

    # Testing & Quality
    "selenium", "junit", "pytest", "cypress", "postman", "unit testing",
    "manual testing", "automation testing", "qa",

    # Management, Business & Domain
    "agile", "scrum", "jira", "project management", "product management",
    "business analysis", "communication", "leadership", "crm", "sap", "erp",
    "financial modeling", "accounting", "recruiting", "talent acquisition",
    "human resources", "sales", "marketing", "seo", "content writing"
}

# Skill Alias and Normalization Mapping
SKILL_ALIASES: Dict[str, str] = {
    # Data & ML
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "artificial intelligence": "machine learning",
    "ai": "machine learning",
    "scikit learn": "scikit-learn",
    "scikit": "scikit-learn",
    "sklearn": "scikit-learn",
    "powerbi": "power bi",
    "ms excel": "excel",
    "microsoft excel": "excel",
    "advanced excel": "excel",
    "pyspark": "spark",
    "apache spark": "spark",
    "apache kafka": "kafka",
    "apache hadoop": "hadoop",
    "tf": "tensorflow",
    "torch": "pytorch",
    "t-sql": "sql",
    "pl/sql": "sql",
    "mysql database": "mysql",
    "postgres": "postgresql",
    "postgre sql": "postgresql",
    "mongo": "mongodb",
    "ms sql": "sql server",
    "mssql": "sql server",

    # Programming & Web
    "js": "javascript",
    "ts": "typescript",
    "react": "react.js",
    "reactjs": "react.js",
    "node": "node.js",
    "nodejs": "node.js",
    "vue": "vue.js",
    "vuejs": "vue.js",
    "angularjs": "angular",
    "express": "express.js",
    "expressjs": "express.js",
    "nextjs": "next.js",
    "cpp": "c++",
    "cplusplus": "c++",
    "c sharp": "c#",
    "csharp": "c#",
    "dot net": ".net",
    "dotnet": ".net",
    "asp.net": ".net",
    "spring": "spring boot",
    "golang": "go",
    "restful api": "rest api",
    "restful apis": "rest api",
    "rest apis": "rest api",
    "rest": "rest api",

    # Cloud & DevOps
    "amazon web services": "aws",
    "gcp": "google cloud",
    "google cloud platform": "google cloud",
    "k8s": "kubernetes",
    "k8": "kubernetes",
    "cicd": "ci/cd",
    "continuous integration": "ci/cd",
    "github actions": "github",

    # Business & General
    "hr": "human resources",
    "talent management": "talent acquisition",
    "business intelligence": "data analysis",
    "bi": "data analysis",
    "pm": "project management",
}

# Technical tokens containing special characters that must be preserved during token cleaning
PROTECTED_TOKENS = {
    "c++": "__TECH_CPP__",
    "c#": "__TECH_CSHARP__",
    ".net": "__TECH_DOTNET__",
    "node.js": "__TECH_NODEJS__",
    "react.js": "__TECH_REACTJS__",
    "vue.js": "__TECH_VUEJS__",
    "next.js": "__TECH_NEXTJS__",
    "express.js": "__TECH_EXPRESSJS__",
    "ci/cd": "__TECH_CICD__",
    "scikit-learn": "__TECH_SKLEARN__",
    "spring boot": "__TECH_SPRINGBOOT__",
}
REVERSE_PROTECTED_TOKENS = {v: k for k, v in PROTECTED_TOKENS.items()}


def clean_text(text: str, preserve_tech_tokens: bool = True) -> str:
    """Clean raw resume or job text for NLP vectorization.
    
    Parameters
    ----------
    text : str
        Input raw text.
    preserve_tech_tokens : bool
        Whether to protect tokens like C++, C#, .NET, Node.js during cleaning.
        
    Returns
    -------
    str
        Cleaned lowercase text.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    cleaned = text.strip()

    # Protect known special technical tokens if requested
    if preserve_tech_tokens:
        for token, placeholder in PROTECTED_TOKENS.items():
            pattern = re.compile(rf"(?i)\b{re.escape(token)}\b" if token.isalnum() else rf"(?i){re.escape(token)}")
            cleaned = pattern.sub(placeholder, cleaned)

    # 1. Remove URLs
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)

    # 2. Remove email addresses
    cleaned = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", cleaned)

    # 3. Remove phone numbers and long numeric sequences
    cleaned = re.sub(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", " ", cleaned)

    # 4. Remove HTML tags if present
    cleaned = re.sub(r"<.*?>", " ", cleaned)

    # 5. Remove miscellaneous punctuation but keep alphanumeric and placeholders
    cleaned = re.sub(r"[^\w\s\_\-\.]", " ", cleaned)

    # Restore protected tokens
    if preserve_tech_tokens:
        for placeholder, token in REVERSE_PROTECTED_TOKENS.items():
            cleaned = cleaned.replace(placeholder, f" {token} ")

    # 6. Lowercase and collapse extra whitespace
    cleaned = cleaned.lower()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def normalize_skill(skill: str) -> str:
    """Normalize a skill name against known aliases and canonical forms."""
    if not isinstance(skill, str):
        return ""
    normalized = skill.strip().lower()
    # Normalize common separators
    normalized = re.sub(r"[\s_]+", " ", normalized)
    # Check alias dictionary
    if normalized in SKILL_ALIASES:
        return SKILL_ALIASES[normalized]
    return normalized


def extract_skills(text: str) -> List[str]:
    """Extract and normalize all recognized skills from resume or job text.
    
    Parameters
    ----------
    text : str
        Input resume or job description text.
        
    Returns
    -------
    List[str]
        Deduplicated, sorted list of normalized canonical skills found in the text.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    # First clean gently
    text_lower = text.lower()
    # Replace commas, slashes, pipes, bullet points with spaces
    padded_text = re.sub(r"[,/|;•·\n\r\t()\[\]{}]", " ", text_lower)
    padded_text = f"  {padded_text}  "

    found_skills: Set[str] = set()

    # Check multi-word and special token skills first
    all_targets = sorted(list(CANONICAL_SKILLS | set(SKILL_ALIASES.keys())), key=lambda s: len(s), reverse=True)

    for target in all_targets:
        # Build boundary regex based on characters
        if target in ("c++", "c#", ".net", "node.js", "react.js", "vue.js", "next.js", "ci/cd"):
            pattern = rf"(?:^|[\s,;:(]){re.escape(target)}(?:$|[\s,;:).!?])"
        elif len(target) <= 2:
            # Single or two-letter tokens like 'r', 'go', 'qa' require strict word boundaries
            pattern = rf"\b{re.escape(target)}\b"
        else:
            pattern = rf"\b{re.escape(target)}\b"

        if re.search(pattern, padded_text):
            canonical = normalize_skill(target)
            if canonical in CANONICAL_SKILLS:
                found_skills.add(canonical)
            else:
                found_skills.add(canonical)

    return sorted(list(found_skills))
