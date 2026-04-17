import os
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

class FeatureScaler:
    """
    Standardizes features by removing the mean and scaling to unit variance.
    Must be fit on the train split only to avoid data leakage.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, features: np.ndarray):
        """Fit scaler on a NumPy array of features (expected train portion)."""
        self.scaler.fit(features)
        self.is_fitted = True
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Apply the fitted scaler to features (train, val, or test)."""
        if not self.is_fitted:
            raise ValueError("Scaler is not fitted yet. Call 'fit' on train data first.")
        # StandardScaler returns float64 by default, cast back to float32
        return self.scaler.transform(features).astype(np.float32)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit and transform in one step (for train data)."""
        return self.fit(features).transform(features)

    def save(self, path: str):
        """Save the fitted scaler to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted scaler.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.scaler, path)

    @classmethod
    def load(cls, path: str) -> 'FeatureScaler':
        """Load a fitted scaler from disk."""
        instance = cls()
        instance.scaler = joblib.load(path)
        instance.is_fitted = True
        return instance
