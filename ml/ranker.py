"""
Reranker for market relevance.

Authors:
- JGY <jyoung22@uvm.edu>
- Erik Arnold <ewarnold@uvm.edu>
"""

import instructor
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field


class MarketRelevance(BaseModel):
    """Structured response model for market relevance scoring."""

    reasoning: str = Field(description="Reasoning for the assigned relevance score.")
    relevance_score: float = Field(ge=0, le=1, description="Score from 0 to 1 indicating "
                                                           "market relevance.")

class DiseaseInformation(BaseModel):
    """Structured model for Disease information."""

    disease: str = Field(description="What disease we are interested in.")
    date: str = Field(description="The date we are interested in.")
    overall_index_question: str = Field(description="Overall index question.")

def get_prompt(prompt_template_file: str, disease_data: DiseaseInformation) -> str:
    """
    Use a template file to generate a prompt.

    Args:
        prompt_template_file: template file to use.
        disease_data: disease_information about the event.

    Returns:
        A MarketRelevance object containing reasoning and score.

    """
    env = Environment(loader=FileSystemLoader('.'), autoescape=True)
    template = env.get_template(prompt_template_file)
    return template.render(disease = disease_data.disease,
                           date = disease_data.date,
                           overall_index_question = disease_data.overall_index_question)


def relevance_score(prompt: str, market_text_representation: str,
                    client: instructor.Instructor) -> MarketRelevance:
    """
    Rank the relevance of a market description to a given prompt.

    Args:
        prompt: The system-level instruction or prompt for ranking.
        market_text_representation: A text representation of the market.
        client: An Instructor-enhanced OpenAI client.

    Returns:
        A MarketRelevance object containing reasoning and score.

    """
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": market_text_representation}
        ],
        response_model=MarketRelevance,
        max_retries=3,
        temperature=0.3
    )
    return response


