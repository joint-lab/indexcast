"""
Market classification models.

Authors:
- Erik Arnold <ernold@uvm.edu>
- JGY <jyoung22@uvm.edu>
"""
from os import path

import joblib
from sentence_transformers import SentenceTransformer

from models.markets import Market


class H5N1Classifier:
    """Classifier for H5N1 markets."""

    def __init__(self):
        """Initialize H5N1Classifier."""
        self.model = SentenceTransformer('Alibaba-NLP/gte-base-en-v1.5', trust_remote_code=True)
        # classifier trained from the market_classification_summary.ipynb in notebooks directory
        base_dir = path.dirname(path.abspath(__file__))
        joblib_path = path.join(base_dir, "binary", "classifier_pipeline.joblib")
        self.classifier = joblib.load(joblib_path)
        joblib_path2 = path.join(base_dir, "binary", "second_classifier_pipeline.joblib")
        self.second_classifier = joblib.load(joblib_path2)


    def predict(self, market: Market) -> bool:
        """
        Decide whether a market is about H5N1.

        Args:
            market (Market): The market to classify.

        Returns:
            bool: True if the market is about H5N1, False otherwise.

        """
        encoded_market = self.model.encode(market.question).reshape(1, -1)
        if self.classifier.predict(encoded_market) == 1:
            if self.second_classifier.predict(encoded_market) == 1:
                return True
        else:
            return False
