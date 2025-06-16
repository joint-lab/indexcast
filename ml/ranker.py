import instructor
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

class MarketRelevance(BaseModel):
    """Structured response model for market relevance scoring."""
    reasoning: str = Field(description="Reasoning for the assigned relevance score.")
    relevance_score: float = Field(description="Score from 0 to 1 indicating market relevance.")

    @field_validator("relevance_score", mode="after")
    @classmethod
    def validate_relevance_score(cls, relevance_score: float) -> float:
        if not (0.0 <= relevance_score <= 1.0):
            raise ValueError("Relevance score must be between 0 and 1.")
        return relevance_score


def create_client_from_api_key(key_path: str) -> str:
    """Load OpenAI API key from a given file path."""
    with open(key_path, "r") as f:
        api_key=f.read().strip()
    return instructor.from_openai(OpenAI(api_key=api_key))

def relevance_score(prompt: str, market_description: str, client: instructor.Instructor) -> MarketRelevance:
    """
    Rank the relevance of a market description to a given prompt.

    Args:
        prompt: The system-level instruction or prompt for ranking.
        market_description: A description of the market.
        client: An Instructor-enhanced OpenAI client.
    Returns:
        A MarketRelevance object containing reasoning and score.
    """
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": market_description}
        ],
        response_model=MarketRelevance,
        max_retries=3,
        temperature=0.3
    )
    return response

