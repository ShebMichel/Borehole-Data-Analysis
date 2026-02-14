from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

CANDIDATE_MODELS = {
    'logistic': LogisticRegression(max_iter=1000, random_state=42),
    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'gradient_boost': GradientBoostingClassifier(n_estimators=100, random_state=42),
}
