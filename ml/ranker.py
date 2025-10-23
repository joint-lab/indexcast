"""
Reranker for market relevance.

Authors:
- JGY <jyoung22@uvm.edu>
- Erik Arnold <ewarnold@uvm.edu>
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from os import path

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
    date: datetime = Field(description="The date we are interested in.")
    overall_index_question: str = Field(description="Overall index question.")

def get_prompt(prompt_template_file: str, disease_data: DiseaseInformation) -> str:
    """
    Use a template file to generate a prompt.

    Args:
        prompt_template_file: template file to use.
        disease_data: disease_information about the event.

    Returns:
        A rendered prompt.

    """
    base_dir = path.dirname(path.abspath(__file__))
    templates_dir = path.join(base_dir, "prompts")
    env = Environment(
        loader=FileSystemLoader(templates_dir),
        autoescape=True
    )
    template = env.get_template(prompt_template_file)
    return template.render(disease = disease_data.disease,
                           date = disease_data.date,
                           overall_index_question = disease_data.overall_index_question)


def get_relevance(prompt: str, market_text_representation: str,
                    client: instructor.Instructor) -> tuple[list, list, float]:
    """
    Rank the relevance of a market description to a given prompt.

    Uses in-memory parallel batching.
    """
    messages_list = [
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": market_text_representation},
        ]
        for _ in range(10)
    ]

    def call_model(messages):
        # One API call per message set
        return client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            response_model=MarketRelevance,
            max_retries=3,
            temperature=0.3,
        )

    responses = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(call_model, m): m for m in messages_list}
        for future in as_completed(futures):
            responses.append(future.result())

    # Compute final aggregates
    if not responses:
        raise RuntimeError("No successful responses from relevance model")

    scores = [r.relevance_score for r in responses]
    reasonings = [r.reasoning for r in responses]
    average_score = sum(scores) / len(scores)

    return reasonings, scores, average_score


