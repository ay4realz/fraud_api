# app/fraud_model.py
import numpy as np
import pandas as pd
import xgboost as xgb

# ==============================================================
# Rule-based feature engineering and scoring
# ==============================================================

class EnhancedAdaptiveRiskScorer:
    def __init__(self):
        pass

    def extract_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Basic balance anomalies
        df["balance_error_orig"] = (df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]).abs()
        df["balance_error_dest"] = (df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]).abs()

        # Transaction type indicators
        df["is_transfer"] = (df["type"] == "TRANSFER").astype(int)
        df["is_cashout"] = (df["type"] == "CASH_OUT").astype(int)

        # Account emptied flag
        df["orig_balance_emptied"] = ((df["oldbalanceOrg"] > 0) & (df["newbalanceOrig"] == 0)).astype(int)

        # High amount transactions
        df["high_amount"] = (df["amount"] > df["amount"].quantile(0.95)).astype(int)

        # Round amount flag
        df["round_amount"] = ((df["amount"] % 1000) == 0).astype(int)

        # Balance ratios
        df["balance_ratio_orig"] = np.where(
            df["oldbalanceOrg"] > 0, df["newbalanceOrig"] / df["oldbalanceOrg"], 0
        )
        df["balance_ratio_dest"] = np.where(
            df["oldbalanceDest"] > 0, df["newbalanceDest"] / df["oldbalanceDest"], 0
        )

        # Destination frequency (within this batch)
        dest_counts = df["nameDest"].value_counts()
        df["dest_frequency"] = df["nameDest"].map(dest_counts)

        # Timing bursts (within this batch)
        df["burst_activity"] = df.groupby("step")["amount"].transform(
            lambda x: (x > x.quantile(0.9)).astype(int)
        )

        return df

    def compute_risk_score(self, df: pd.DataFrame) -> np.ndarray:
        df = self.extract_enhanced_features(df)

        weights = {
            "is_transfer": 0.3,
            "is_cashout": 0.3,
            "balance_error_orig": 0.1,
            "balance_error_dest": 0.1,
            "orig_balance_emptied": 0.2,
            "high_amount": 0.2,
            "round_amount": 0.1,
            "balance_ratio_orig": 0.1,
            "balance_ratio_dest": 0.1,
            "dest_frequency": 0.05,
            "burst_activity": 0.15,
        }

        scores = np.zeros(len(df))
        for feat, w in weights.items():
            if feat in df.columns:
                vals = df[feat].astype(float).fillna(0).values
                if vals.max() > 0:
                    vals = vals / (vals.max() + 1e-9)
                scores += w * vals

        if scores.max() > 0:
            scores = scores / scores.max()

        return scores


# ==============================================================
# Hybrid model: Rule scorer + XGBoost
# ==============================================================

class HybridFraudDetectionModel:
    def __init__(self, rule_weight: float = 0.5, categorical_cols=None):
        self.rule_scorer = EnhancedAdaptiveRiskScorer()
        self.xgb_model = None
        self.rule_weight = rule_weight
        self.categorical_cols = categorical_cols or ["type", "nameOrig", "nameDest"]

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical string fields into numeric codes."""
        df = df.copy()
        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype("category").cat.codes
        return df

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train the hybrid model."""
        # Encode categoricals
        X = self._encode_categoricals(X)

        # Add rule-based features
        X_enhanced = self.rule_scorer.extract_enhanced_features(X.copy())

        self.xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss"
        )
        self.xgb_model.fit(X_enhanced, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return fraud probability for each row."""
        X = self._encode_categoricals(X)
        X_enhanced = self.rule_scorer.extract_enhanced_features(X.copy())

        xgb_probs = self.xgb_model.predict_proba(X_enhanced)[:, 1]
        rule_scores = self.rule_scorer.compute_risk_score(X_enhanced)

        combined = self.rule_weight * rule_scores + (1 - self.rule_weight) * xgb_probs
        return combined

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Return binary fraud labels (0/1)."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

