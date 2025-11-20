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
        # initial classifier trained from the market_classification_summary.ipynb
        # in notebooks directory this classifier gets all markets that could be relating to H5N1
        base_dir = path.dirname(path.abspath(__file__))
        path_to_initial_classifier = path.join(base_dir, "binary",
                                               "initial_classifier_pipeline.joblib")
        self.initial_classifier = joblib.load(path_to_initial_classifier)
        # second classifier trained from the second_classifier.ipynb in notebooks directory
        # this classifier prunes the results of the original for relevance to H5N1
        path_to_relevance_classifier = path.join(base_dir, "binary",
                                                 "relevance_classifier_pipeline.joblib")
        self.relevance_classifier = joblib.load(path_to_relevance_classifier)


    def predict(self, market: Market) -> bool:
        """
        Decide whether a market is about H5N1.

        Args:
            market (Market): The market to classify.

        Returns:
            bool: True if the market is about H5N1, False otherwise.

        """
        encoded_market = self.model.encode(market.question).reshape(1, -1)
        if self.initial_classifier.predict(encoded_market) == 1:
            return self.relevance_classifier.predict(encoded_market) == 1
        else:
            return False


