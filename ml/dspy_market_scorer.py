"""
DSPy-based market scoring model.

This class loads a previously optimized DSPy program (few-shot teleprompter)
and exposes a simple `.predict()` method for use in pipelines.

Authors:
- Erik Arnold <ernold@uvm.edu>
- JGY <jyoung22@uvm.edu>
"""



from pathlib import Path

import dspy


class DSPyMarketScorer:
    """Wrapper class for using a saved DSPy optimized program in a production pipeline."""

    def __init__(self):
        """
        Load the saved optimized DSPy program.

        Args:
            filename (str, optional): Specific JSON filename inside the json/ folder.
                                      If None, automatically loads the first JSON file.

        """
        dspy.settings.configure(
            lm=dspy.LM(
                model="gpt-4.1-mini",
                provider="openai",
                temperature=0,
                top_p=1,
            )
        )
        # Directory containing model file, this was trained via ipynb
        folder = Path(__file__).parent / "relevance_model"
        if not folder.exists():
            raise FileNotFoundError("Folder not found: relevance_model")
        # Load the saved ChainOfThought program directly
        self.program = dspy.load(folder)


    def predict(self, index_question: str, market_title: str):
        """
        Run the optimized DSPy Chain-of-Thought scorer.

        Args:
            index_question (str): Reference "index" question.
            market_title (str): The forecasting market's question/title.

        Returns:
            dict: { rationale, label, score }

        """
        output = self.program(
            index_question=index_question,
            market_title=market_title
        )
        return {
            "rationale": getattr(output, "rationale", None),
            "label": getattr(output, "label", None),
            "score": float(getattr(output, "score", None)) if getattr(output, "score", None)
                                                              is not None else None
        }