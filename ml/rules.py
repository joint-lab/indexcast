"""
Rule generator for index.

Authors:
- JGY <jyoung22@uvm.edu>
- Erik Arnold <ewarnold@uvm.edu>
"""

from datetime import datetime
from os import path

import instructor
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field, create_model, field_validator, model_validator

from ml.formula import (
    Formula,
    validate_literal_count,
    validate_market_ids,
    validate_rule_depth,
)

DEFAULT_MODEL = "gpt-4.1"
DEFAULT_TEMPERATURE = 0.8
DEFAULT_MAX_RETRIES = 3
MAX_NUMBER_OF_RULES = 30


class FormulaItem(BaseModel):
    """Top-level wrapper for a formula with reasoning."""

    reasoning: str
    rule: Formula

    @model_validator(mode="after")
    def check_nonempty(self):
        """Make sure reasoning isn't empty."""
        if not self.reasoning.strip():
            raise ValueError("Reasoning cannot be empty")
        return self

def response_with_n(n: int) -> type[BaseModel]:
    """Enforce 1:n rules returned from the model."""
    return create_model(
        f"ResponseWith{n}",
        content=(list[FormulaItem], Field(min_length=1, max_length=n)),
    )


# Data Model for Prompt Input
class PromptInformation(BaseModel):
    """Structured model for Disease information."""

    date: datetime = Field(description="The date we are interested in.")
    overall_index_question: str = Field(description="Overall index question.")
    max_num_of_rules: int = Field(description="Max number of rules to be generated.")

def get_rules(
    prompt: str,
    max_num_rules: int,
    client: instructor.Instructor,
    allowed_market_ids: set[str],
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> list[FormulaItem]:
    """
    Get rules using eligible markets.

    Args:
        prompt: The system-level instruction or prompt for ranking.
        max_num_rules: maximum number of rules to be generated.
        client: An Instructor-enhanced OpenAI client.
        model: The model to use for generation.
        allowed_market_ids: set of valid market IDs.
        model: The model to use for generation.
        temperature: The temperature used for generation.

    Returns:
        A list of FormulaItems.

    """
    response_model = response_with_n(max_num_rules)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": prompt}],
        response_model=response_model,
        temperature=temperature,
        max_retries=DEFAULT_MAX_RETRIES,
    )

    rules = response.content
    rules = rules[:MAX_NUMBER_OF_RULES]
    for rule_item in rules:
        r = rule_item.rule

        validate_rule_depth(r)
        validate_literal_count(r)
        validate_market_ids(r, allowed_market_ids)
    return rules

def get_rules_prompt(
    prompt_template_file: str,
    prompt_data: PromptInformation,
    markets: dict,
) -> str:
    """
    Use a template file to generate a prompt.

    Args:
        prompt_template_file: template file to use.
        prompt_data: prompt_information about the event.
        markets: Dictionary of market_id -> market info.
        existing_rules: List of existing rule strings to avoid duplicates.

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
    return template.render(
        date=prompt_data.date,
        overall_index_question=prompt_data.overall_index_question,
        max_num_of_rules=prompt_data.max_num_of_rules,
        markets=markets,
    )


#######################################################
# Weights for the rules
#######################################################


class WeightScore(BaseModel):
    """
    Represent the evaluation of a rule's predictive power for forecasting outbreaks.

    Attributes:
        reasoning (str): A text explanation providing context or justification for the given score.
        weight_score (float): A score between 0 and 1

    """

    reasoning: str = Field(description="The reasoning behind the score")
    weight_score: float = Field(description="The strength score of the rule in "
                                            "predicting an outbreak in the next 12 "
                                            "months on the scale of 0 to 1")

    @field_validator("weight_score", mode="after")
    @classmethod
    def validate_strength_score(cls, weight_score: float) -> float:
        """
        Validate that the strength score falls within the acceptable range of 0 to 1.

        Args:
            weight_score (float): The score to validate.

        Returns:
            float: The validated score if within range.

        Raises:
            ValueError: If the score is less than 0 or greater than 1.

        """
        if weight_score < 0 or weight_score > 1:
            raise ValueError("Strength score must be between 0 and 1")
        return weight_score

def get_weight(prompt: str, market_text_representation: str,
                    client: instructor.Instructor) -> tuple[list, list, float]:
    """
    Weight of a rule using a given prompt and a rule representation.

    Args:
        prompt: The system-level instruction or prompt for ranking.
        market_text_representation: A text representation of the market.
        client: An Instructor-enhanced OpenAI client.

    Returns:
        A float average score for ten responses.

    """
    scores = []
    reasonings = []
    for _ in range(5):
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": market_text_representation}
            ],
            response_model=WeightScore,
            max_retries=3,
            temperature=0.3
        )
        scores.append(response.weight_score)
        reasonings.append(response.reasoning)

    # Calculate average score
    average_score = sum(scores) / len(scores)
    return reasonings, scores, average_score
