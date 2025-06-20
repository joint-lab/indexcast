"""
Market classification models.

Authors:
- Erik Arnold <ernold@uvm.edu>
- JGY <jyoung22@uvm.edu>
"""
import numpy as np
from models.markets import Market
from sentence_transformers import SentenceTransformer
import joblib

class H5N1Classifier:
    def __init__(self):
        self.model = SentenceTransformer('Alibaba-NLP/gte-base-en-v1.5', trust_remote_code=True)
        # classifier trained from the market_classification_summary.ipynb in notebooks directory
        self.classifier = joblib.load("binary/classifier_pipeline.joblib")


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
            return True
        else:
            return False
