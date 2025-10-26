"""
Rule generator for index.

Authors:
- JGY <jyoung22@uvm.edu>
- Erik Arnold <ewarnold@uvm.edu>
"""

import re
from datetime import datetime
from os import path
from typing import Annotated, Literal, Union

import instructor
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field, field_validator, model_validator

# Constants for rule validation
MAX_RULE_DEPTH = 2
MAX_LITERALS_PER_RULE = 3
MIN_CHILDREN_PER_NODE = 2
MAX_CHILDREN_PER_NODE = 3
DEFAULT_MODEL = "gpt-4.1"
DEFAULT_TEMPERATURE = 0.8
DEFAULT_MAX_RETRIES = 3


# ── Variable node ──────────────────────────────────────────────────
class Var(BaseModel):
    """Represents a variable in the formula."""
    node_type: Literal["variable", "var"] = Field(default="variable")
    var: str = Field(description="Variable name")

    @property
    def variable_name(self) -> str:
        """Backward compatibility property."""
        return self.var

    @model_validator(mode='after')
    def normalize_node_type(self):
        """Ensure node_type is always 'variable' after validation."""
        self.node_type = "variable"
        return self

    model_config = dict(populate_by_name=True, extra="ignore")


# ── Operator node ──────────────────────────────────────────────────
class Op(BaseModel):
    """Represents a logical operator with child formulas."""
    node_type: Literal["and", "or", "not", "xor", "nand", "nor", "common"]
    arguments: list['Formula'] = Field(min_length=1)

    @model_validator(mode='before')
    @classmethod
    def normalize_before(cls, data):
        """Normalize node_type and arguments before validation."""
        if isinstance(data, dict):
            # Normalize node_type aliases
            if 'node_type' in data:
                nt = str(data['node_type']).strip().lower()
                if nt == 'common':
                    data['node_type'] = 'and'
                else:
                    data['node_type'] = nt

            # Handle arguments normalization
            if 'arguments' in data:
                args = data['arguments']

                # If arguments is not a list, wrap it
                if not isinstance(args, list):
                    data['arguments'] = [args]
                else:
                    # Flatten if it's a list containing a single list
                    if len(args) == 1 and isinstance(args[0], list):
                        data['arguments'] = args[0]

        return data

    @field_validator("arguments")
    @classmethod
    def check_arity(cls, v, info):
        """Validate operator arity rules."""
        op = info.data.get("node_type")
        if op == "not" and len(v) != 1:
            raise ValueError(f"NOT requires exactly one child, got {len(v)}")
        if op in ["and", "or", "xor", "nand", "nor", "common"] and len(v) < 2:
            raise ValueError(f"{op.upper()} requires at least 2 children, got {len(v)}")
        return v

    model_config = dict(extra="ignore")


# ── Unknown fallback ───────────────────────────────────────────────
class Unknown(BaseModel):
    """Fallback for unrecognized formula structures."""
    node_type: Literal["unknown"] = "unknown"
    content: dict = Field(default_factory=dict)

    model_config = dict(extra="ignore")


# ── Discriminated union ────────────────────────────────────────────
Formula = Annotated[
    Union[Var, Op, Unknown],
    Field(discriminator="node_type"),
]

# Rebuild Op model to resolve ForwardRef
Op.model_rebuild()


# ── Helper function for cleaning dict keys ────────────────────────
def clean_dict_keys(d):
    """Recursively clean dictionary keys by stripping whitespace."""
    if not isinstance(d, dict):
        return d

    cleaned = {}
    for k, v in d.items():
        clean_key = k.strip() if isinstance(k, str) else k
        if isinstance(v, dict):
            cleaned[clean_key] = clean_dict_keys(v)
        elif isinstance(v, list):
            cleaned[clean_key] = [clean_dict_keys(item) if isinstance(item, dict) else item for item in v]
        else:
            cleaned[clean_key] = v

    return cleaned


# ── Wrapper with top-level normalization ───────────────────────────
class FormulaItem(BaseModel):
    """Top-level wrapper for a formula with reasoning and verbalization."""
    reasoning: str
    verbalization: str
    rule: Formula

    @model_validator(mode='before')
    @classmethod
    def normalize_and_validate(cls, data):
        """Clean keys and validate structure before processing."""
        if not isinstance(data, dict):
            return data

        # Clean all keys recursively first
        data = clean_dict_keys(data)

        # CRITICAL: Check if rule is empty/invalid before proceeding
        if 'rule' in data:
            rule = data['rule']

            # If rule is empty string, None, or empty dict, skip this item
            if not rule or rule == "" or (isinstance(rule, dict) and not rule):
                raise ValueError("Rule field is empty or invalid")

            # If rule is a dict, normalize it
            if isinstance(rule, dict):
                data['rule'] = cls._normalize_formula_dict(rule)

        # Validate required fields are not empty
        if not data.get('reasoning', '').strip():
            raise ValueError("Reasoning field is empty")
        if not data.get('verbalization', '').strip():
            raise ValueError("Verbalization field is empty")

        return data

    @staticmethod
    def _normalize_formula_dict(d: dict) -> dict:
        """Recursively normalize a formula dictionary."""
        if not isinstance(d, dict):
            return d

        # Strip whitespace from all keys first
        normalized = {k.strip().lower(): v for k, v in d.items()}

        # Handle node_type variations
        if 'nodetype' in normalized:
            normalized['node_type'] = normalized.pop('nodetype')

        if 'node_type' in normalized:
            nt = str(normalized['node_type']).strip().lower()
            # Map common aliases
            alias_map = {
                'var': 'variable',
                'variable': 'variable',
                'common': 'and',
            }
            normalized['node_type'] = alias_map.get(nt, nt)

        # Handle variable name aliases
        if normalized.get('node_type') in ['variable', 'var']:
            if 'variable' in normalized and 'var' not in normalized:
                normalized['var'] = normalized.pop('variable')
            if 'variable_name' in normalized and 'var' not in normalized:
                normalized['var'] = normalized.pop('variable_name')

        # Recursively normalize arguments with double-nesting fix
        if 'arguments' in normalized:
            args = normalized['arguments']
            if isinstance(args, list):
                # Check for double-nesting [[...]]
                if len(args) == 1 and isinstance(args[0], list):
                    args = args[0]

                normalized['arguments'] = [
                    FormulaItem._normalize_formula_dict(arg) if isinstance(arg, dict) else arg
                    for arg in args
                ]
            elif isinstance(args, dict):
                normalized['arguments'] = [FormulaItem._normalize_formula_dict(args)]

        # Fallback to unknown if no valid node_type
        if 'node_type' not in normalized or normalized['node_type'] not in [
            'variable', 'var', 'and', 'or', 'not', 'xor', 'nand', 'nor', 'common', 'unknown'
        ]:
            return {
                'node_type': 'unknown',
                'content': d  # Preserve original
            }

        return normalized


# ── Response wrapper that filters out invalid items ────────────────
class Response(BaseModel):
    """Response containing a list of formula items."""
    content: list[FormulaItem]

    @model_validator(mode='before')
    @classmethod
    def filter_invalid_items(cls, data):
        """Filter out invalid items before validation."""
        if isinstance(data, dict) and 'content' in data:
            if isinstance(data['content'], list):
                # Filter out items with validation errors
                valid_items = []
                for item in data['content']:
                    try:
                        # Quick check for empty/invalid items
                        if not isinstance(item, dict):
                            continue
                        if not item.get('reasoning', '').strip():
                            continue
                        if not item.get('verbalization', '').strip():
                            continue
                        if not item.get('rule') or item.get('rule') == '':
                            continue
                        valid_items.append(item)
                    except Exception:
                        # Skip items that fail basic validation
                        continue

                data['content'] = valid_items

        return data

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

    for _chunk_idx in range(chunks_needed):
        rules_in_chunk = min(chunk_size, total_rules - len(all_rules))
        if rules_in_chunk <= 0:
            break

        # Update prompt for this chunk size
        chunk_prompt = prompt.replace(f"{total_rules} NEW", f"{rules_in_chunk} NEW")
        chunk_prompt = chunk_prompt.replace("{{num_of_rules}}", str(rules_in_chunk))

        try:
            chunk_rules = get_rules_single_chunk(
                chunk_prompt, valid_market_ids, client, rules_in_chunk,
                model, temperature, max_retries
            )
            all_rules.extend(chunk_rules)

        except Exception as e:
            raise RuntimeError("Failed to extract rules") from e
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
        response_model=Response,
        max_retries=1,
        temperature=temperature,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.3
    )

    # Return the filtered content list
    return response.content


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


# Helper functions for working with the new schema
def stringify_formula(formula: Formula, session) -> str:
    """Get a string representation of the given formula."""
    if formula.node_type == "variable":
        from sqlmodel import select

        from models.markets import Market
        
        market = session.exec(
            select(Market).where(Market.id == formula.var.strip('"').strip("'"))
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
            cleaned_name = node.var.strip('"').strip("'")
            literals.append(cleaned_name)
        elif hasattr(node, "arguments"):
            for child in node.arguments:
                traverse(child)

    traverse(formula)
    return literals