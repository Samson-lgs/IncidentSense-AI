from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .features import build_text


class SimilarIncidentIndex:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=8000)
        self.matrix = None
        self.rows = None

    def fit(self, df):
        self.rows = df[["incident_id", "root_cause", "log_message", "service", "severity"]].copy()
        self.matrix = self.vectorizer.fit_transform(build_text(df))
        return self

    def query(self, incident, top_k=5):
        if self.matrix is None:
            raise RuntimeError("Similarity index is not fitted")
        q = self.vectorizer.transform(build_text(incident))
        scores = cosine_similarity(q, self.matrix)[0]
        order = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in order:
            row = self.rows.iloc[int(i)]
            results.append(
                {
                    "incident_id": row["incident_id"],
                    "root_cause": row["root_cause"],
                    "service": row["service"],
                    "severity": row["severity"],
                    "log_message": row["log_message"],
                    "similarity": round(float(scores[i]), 4),
                }
            )
        return results
