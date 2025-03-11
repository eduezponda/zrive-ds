import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from typing import Dict, Tuple

class PushModel:
    MODEL_COLUMNS = [
        "ordered_before",
        "abandoned_before",
        "normalised_price",
        "set_as_regular",
        "active_smoozed",
        "discount_pct",
        "days_since_purchase_variant_id",
        "avg_days_to_buy_variant_id",
        "std_days_to_buy_variant_id",
        "days_since_purchase_product_type",
        "avg_days_to_buy_product_type",
        "std_days_to_buy_product_type",
        "global_popularity",
        "people_ex_baby",
        "count_adults",
        "count_children",
        "count_babies",
        "count_pets",
        "user_order_seq",
    ]

    TARGET_COLUMN = "outcome"

    def __init__(
        self,
        classifier_parametrisation: Dict,
        calibration_parametrisation: Dict,
        prediction_threshold: int,
    ) -> None:
        """
        Instantiate the push model which is a calibrated Gradient Boosting Classifier.

        Args:
            classifier_parametrisation: 
                {
                    "GradientBoostingTree parameter": value,
                    "Gradient Boosting Tree parameter 2": value,
                    ...
                }

            calibration_parametrisation: 
                {
                    "Calibrated Classifier CV parameter": value,
                    "CalibratedClassifier CV parameter 2": value,
                    ...
                }

            prediction_threshold: Probability threshold above which a prediction is considered as 1.
        """
        self.clf = CalibratedClassifierCV(
            GradientBoostingClassifier(**classifier_parametrisation),
            method='sigmoid',  # Método de calibración
            cv=calibration_parametrisation.get("cv", 3)  # Número de folds para la calibración
        )
        self.prediction_threshold = prediction_threshold

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[:, self.MODEL_COLUMNS]
    
    def _extract_label(self, df: pd.DataFrame) -> pd.Series:
        return df.loc[:, self.TARGET_COLUMN]
    
    def _feature_label_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        return self._extract_features(df), self._extract_label(df)
    
    def fit(self, df: pd.DataFrame) -> None:
        """Fits the model
        Args:
            df: dataframe containing both the features and the labels. Refer to this class MODEL_COLUMNS and TARGET_COLUMN
        """
        features, labels = self._feature_label_split(df)
        self.clf.fit(features, labels)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Retrieves binary predictions for input X

        Args:
            df: DataFrame to predict

        Returns:
            predictions: pd.Series with predictions for X and same indices as X,
            if it has any.
        """
        features = self._extract_features(df)
        probs = self.predict_proba(features)
        predictions = (probs > self.prediction_threshold).astype(int)
        return predictions
    

    def predict_proba (self, df: pd.DataFrame) -> pd.Series: 
        """Retrieves probability predictions for input X

        Args:
            - df: DataFrame to predict

        Returns:
            - predictions: pd. Series with predictions for X and same indices as X, if it has any.
        """
    
        features = self._extract_features(df)
        predictions = self.clf.predict_proba(features)[:, 1]
        predictions = pd.Series(predictions, name="predictions")

        if hasattr(features, "index"):
            predictions.index = features.index

        return predictions
