"""
Initial labeler for markets.

Authors:
- JGY <jyoung22@uvm.edu>
- Erik Arnold <ewarnold@uvm.edu>
"""

from os import path

import instructor
from pydantic import BaseModel, Field, field_validator


# Pydantic model for market classification
class Label(BaseModel):
    """Class for market labeling."""

    reasoning: str = Field(description="The reasoning behind the labels")
    labels: list[int] = Field(
        description=(
            "List of labels that apply to the market: "
            "-1 for none_of_the_above, "
            "1 for h5n1_outbreak_12mo"
            "4 for next_national_election_democratic_party, "
            "3 for annual_war_deaths_exceed_average, "
            "2 for ai_frontier_milestone_12mo"
        )
    )

    @field_validator("labels", mode="after")
    @classmethod
    def validate_labels(cls, labels: list[int]) -> list[int]:
        """Validate the labels."""
        allowed_labels = [-1, 1, 2, 3, 4]
        for label in labels:
            if label not in allowed_labels:
                raise ValueError("Labels must be one or more of: -1, 1, 2, 3, or 4")
        # Enforce exclusivity of -1
        if -1 in labels and len(labels) > 1:
            raise ValueError("If -1 (none_of_the_above) is used, it must be the only label")
        return labels


def get_initial_labeling_prompt() -> str:
    """
    Get the prompt for initial labeling.

    Returns:
        A rendered prompt.

    """
    base_dir = path.dirname(path.abspath(__file__))
    prompts_dir = path.join(base_dir, "prompts")
    filepath = path.join(prompts_dir, "initial_labeling_prompt.j2")
    with open(filepath, encoding="utf-8") as f:
            return f.read()


def get_initial_label(prompt: str, market_question: str, client: instructor.Instructor) -> Label:
    """
    Rank the relevance of a market description to a given prompt.

    Args:
        prompt: The system-level instruction or prompt for ranking.
        market_question: Question of the market.
        client: An Instructor-enhanced OpenAI client.

    Returns:
        A response which has reasoning behind the labeling and a list of labels.

    """
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": market_question}
        ],
        response_model=Label,
        max_retries=3,
        temperature=0.1
    )
    return response