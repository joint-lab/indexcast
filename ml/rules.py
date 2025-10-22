"""
Rule generator for index.

Authors:
- JGY <jyoung22@uvm.edu>
- Erik Arnold <ewarnold@uvm.edu>
"""

import re
from datetime import datetime
from os import path
from typing import Annotated, Literal

import instructor
from jinja2 import Environment, FileSystemLoader
from pydantic import AliasChoices, BaseModel, Field, field_validator

# Constants for rule validation
MAX_RULE_DEPTH = 2
MAX_LITERALS_PER_RULE = 3
MIN_CHILDREN_PER_NODE = 2
MAX_CHILDREN_PER_NODE = 3
DEFAULT_MODEL = "gpt-4.1"
DEFAULT_TEMPERATURE = 0.8
DEFAULT_MAX_RETRIES = 3


# ── Boolean‑AST schema ───────────────────────────────────────────────

class Var(BaseModel):
    """
    Represents a variable node in a Boolean formula AST (Abstract Syntax Tree).

    Attributes:
        node_type (Literal["variable"]): Must always be the string
                                        'variable'. Used for type discrimination.
        variable_name (str): The name of the variable, provided under the alias 'var'.

    """

    node_type: Literal["variable"] = Field(description="Must be exactly 'variable'")
    variable_name: str = Field(alias="var", description="Variable name")

    model_config = dict(populate_by_name=True, extra="forbid")


class Op(BaseModel):
    """
    Represents an operator node in a Boolean formula AST.

    Attributes:
        node_type (Literal["and", "or", "not", "xor", "nand", "nor"]):
            The Boolean operator represented by this node.
        arguments (list[Formula]):
            The operands or sub-formulas for this operation.
            Accepts keys like 'arguments', 'args', 'children', or the singular 'child'.

    Validators:
        coerce_single: Ensures that a single 'child' provided as a dict is wrapped in a list.
        arity_ok: Validates the correct number of child formulas based on the operator type.

    """

    node_type: Literal["and", "or", "not", "xor", "nand", "nor"]
    # accept many synonyms AND the sloppy `"child": {...}` form
    arguments: list["Formula"] = Field(
        validation_alias=AliasChoices("arguments", "args", "children", "child"),
        description="Sub‑formulas of this operation",
    )

    @field_validator("arguments", mode="before")
    @classmethod
    def coerce_single(cls, v):
        """
        Wrap a single 'child' dict in a list to standardize the 'arguments' format.

        Allows sloppy single-child input.
        """
        # allow `"child": { … }` by wrapping it in a list
        return [v] if isinstance(v, dict) else v

    @field_validator("arguments")
    @classmethod
    def arity_ok(cls, v, info):
        """
        Validate that.

            - 'not' has exactly one argument.
            - All other operators have at least two arguments.

        Raises:
            ValueError if the arity is invalid.

        """
        op = info.data["node_type"]
        if op == "not" and len(v) != 1:
            raise ValueError("NOT needs exactly one child")
        if op != "not" and len(v) < 2:
            raise ValueError(f"{op.upper()} needs ≥2 children")
        return v

    model_config = dict(populate_by_name=True, extra="forbid")


type Formula = Annotated[Var | Op, Field(discriminator="node_type")]
Op.model_rebuild()


# ── response schema ──────────────────────────────────────────────────

class FormulaItem(BaseModel):
    """
    Represent a single item in a response that explains a Boolean formula.

    Attributes:
        reasoning (str):
            A textual explanation of the logical structure or intent behind the formula.
        verbalization (str):
            A natural language version or paraphrase of the formula.
        rule (Formula):
            The actual Boolean formula (AST structure) being described.

    """

    reasoning: str
    verbalization: str
    rule: Formula


# --- Data Model for Prompt Input ---
class PromptInformation(BaseModel):
    """Structured model for Disease information."""

    disease: str = Field(description="What disease we are interested in.")
    date: datetime = Field(description="The date we are interested in.")
    overall_index_question: str = Field(description="Overall index question.")
    num_of_rules: int = Field(description="Number of rules to be generated.")



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
        disease=prompt_data.disease,
        date=prompt_data.date,
        overall_index_question=prompt_data.overall_index_question,
        num_of_rules=prompt_data.num_of_rules,
        markets=markets,
    )


def get_rules_chunked(
    prompt: str,
    valid_market_ids: set[str],
    client: instructor.Instructor,
    total_rules: int,
    chunk_size: int = 10,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_retries: int = DEFAULT_MAX_RETRIES
) -> list[FormulaItem]:
    """
    Generate rules in smaller chunks to avoid JSON parsing issues.
    
    Args:
        prompt: Base prompt template
        valid_market_ids: Set of valid market IDs
        client: Instructor client
        total_rules: Total number of rules to generate
        chunk_size: Number of rules per chunk
        model: Model to use
        temperature: Temperature setting
        max_retries: Max retries per chunk
        
    Returns:
        List of validated FormulaItems

    """
    all_rules = []
    chunks_needed = (total_rules + chunk_size - 1) // chunk_size

    prompts = []
    for _chunk_idx in range(chunks_needed):
        rules_in_chunk = min(chunk_size, total_rules - len(all_rules))
        if rules_in_chunk <= 0:
            break

        chunk_prompt = prompt.replace(f"{total_rules} NEW", f"{rules_in_chunk} NEW")
        chunk_prompt = chunk_prompt.replace("{{num_of_rules}}", str(rules_in_chunk))
        prompts.append(chunk_prompt)

    # Batch process all chunk prompts at once
    responses = client.chat.completions.batch(
        model=model,
        messages=[[{"role": "system", "content": p}] for p in prompts],
        response_model=list[FormulaItem],
        max_retries=max_retries,
        temperature=temperature,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.3
    )

    # Flatten the nested results
    for chunk in responses:
        all_rules.extend(chunk)

    return all_rules


def get_rules_single_chunk(
    prompt: str,
    valid_market_ids: set[str], 
    client: instructor.Instructor,
    num_rules: int,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_retries: int = DEFAULT_MAX_RETRIES
) -> list[FormulaItem]:
    """Generate a single chunk of rules with robust error handling."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": prompt}],
        response_model=list[FormulaItem],
        max_retries=1,
        temperature=temperature,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.3
    )

    return response


def get_rules(
    prompt: str, 
    valid_market_ids: set[str],
    client: instructor.Instructor,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_retries: int = DEFAULT_MAX_RETRIES
) -> list[FormulaItem]:
    """
    Get rules using eligible markets with improved validation and chunking.

    Args:
        prompt: The system-level instruction or prompt for ranking.
        valid_market_ids: Set of valid market IDs for validation.
        client: An Instructor-enhanced OpenAI client.
        model: The model to use for generation.
        temperature: Temperature for generation.
        max_retries: Maximum number of retries on failure.

    Returns:
        A list of FormulaItems.
        
    Raises:
        Exception: If rule generation fails after all retries.

    """
    # Extract number of rules from prompt
    num_match = re.search(r'(\d+) NEW', prompt)
    total_rules = int(num_match.group(1)) if num_match else 30
    
    # Use chunked generation for large batches
    if total_rules > 15:
        return get_rules_chunked(
            prompt, valid_market_ids, client, total_rules,
            chunk_size=10, model=model, temperature=temperature, max_retries=max_retries
        )
    else:
        return get_rules_single_chunk(
            prompt, valid_market_ids, client, total_rules,
            model, temperature, max_retries
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
    messages = [
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": market_text_representation}
        ]
        for _ in range(5)
    ]

    # batch the structured calls in memory
    responses = client.chat.completions.batch(
        model="gpt-4.1",
        messages=messages,
        response_model=WeightScore,
        max_retries=3,
        temperature=0.3,
    )

    # Extract results
    scores = [r.weight_score for r in responses]
    reasonings = [r.reasoning for r in responses]
    average_score = sum(scores) / len(scores)

    return reasonings, scores, average_score


# Helper functions for working with the new schema
def stringify_formula(formula: Formula, session) -> str:
    """Get a string representation of the given formula."""
    if formula.node_type == "variable":
        from sqlmodel import select

        from models.markets import Market
        
        market = session.exec(
            select(Market).where(Market.id == formula.variable_name.strip('"').strip("'"))
        ).first()
        
        return market.question if market else f"[Unknown: {formula.variable_name}]"
    
    elif formula.node_type in ["and", "or", "xor", "nand", "nor"]:
        children = [stringify_formula(child, session) for child in formula.arguments]
        joined = f" {formula.node_type.upper()} ".join(f"({child})" for child in children)
        return joined
    elif formula.node_type == "not":
        child_str = stringify_formula(formula.arguments[0], session)
        return f"NOT ({child_str})"
    else:
        return "[Unknown node type]"


def extract_literals_from_formula(formula: Formula) -> list[str]:
    """
    Extract all variable names from a formula, stripping surrounding quotes.

    Args:
        formula: A Formula instance containing the rule tree

    Returns:
        List of cleaned variable names found in the formula

    """
    literals = []

    def traverse(node: Formula):
        if node.node_type == "variable":
            # Strip leading/trailing quotes (single or double)
            cleaned_name = node.variable_name.strip('"').strip("'")
            literals.append(cleaned_name)
        elif hasattr(node, "arguments"):
            for child in node.arguments:
                traverse(child)

    traverse(formula)
    return literals