"""
Rule generator for index.

Authors:
- JGY <jyoung22@uvm.edu>
- Erik Arnold <ewarnold@uvm.edu>
"""

import json
import re
from datetime import datetime
from os import path
from typing import Annotated, Literal

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


# --- Data Model for Prompt Input ---
class PromptInformation(BaseModel):
    """Structured model for Disease information."""

    disease: str = Field(description="What disease we are interested in.")
    date: datetime = Field(description="The date we are interested in.")
    overall_index_question: str = Field(description="Overall index question.")
    num_of_rules: int = Field(description="Number of rules to be generated.")


def _calculate_node_depth(node: "RuleNode", level: int = 1) -> int:
    """
    Calculate the maximum depth of a rule node tree.
    
    Args:
        node: The rule node to analyze
        level: Current depth level (starts at 1)
        
    Returns:
        Maximum depth of the node tree

    """
    if isinstance(node, LiteralNode):
        return level
    elif isinstance(node, NotNode):
        if level >= MAX_RULE_DEPTH:
            return level
        return _calculate_node_depth(node.child, level + 1)
    elif hasattr(node, "children") and node.children:
        if level >= MAX_RULE_DEPTH:
            return level
        return max(_calculate_node_depth(child, level + 1) for child in node.children)
    return level


def _validate_compound_node(node: "AndNode | OrNode", node_type: str) -> "AndNode | OrNode":
    """
    Validate compound nodes (AND/OR) for depth and literal constraints.
    
    Args:
        node: The compound node to validate
        node_type: Type of node for error messages ("and" or "or")
        
    Returns:
        The validated node
        
    Raises:
        ValueError: If validation fails

    """
    max_depth = max(_calculate_node_depth(child) for child in node.children)
    if max_depth > MAX_RULE_DEPTH:
        raise ValueError(f"Nesting too deep; max depth is {MAX_RULE_DEPTH}.")

    all_literals = {lit.name for lit in node.flatten()}
    if len(all_literals) == 0:
        raise ValueError(f"{node_type.capitalize()}Node must contain at least one literal")
    if len(all_literals) > MAX_LITERALS_PER_RULE:
        raise ValueError(
            f"Too many unique literals in '{node_type}' node: "
            f"{len(all_literals)}. Max is {MAX_LITERALS_PER_RULE}."
        )
    return node


class LiteralNode(BaseModel):
    """A literal node representing a single market variable."""
    
    type: Literal["literal"]
    name: str

    def flatten(self) -> list["LiteralNode"]:
        """Return a list containing only this literal node."""
        return [self]


class AndNode(BaseModel):
    """An AND node combining multiple rule nodes."""
    
    type: Literal["and"]
    children: list["RuleNode"] = Field(..., min_items=MIN_CHILDREN_PER_NODE,
                                       max_items=MAX_CHILDREN_PER_NODE)

    def flatten(self) -> list["LiteralNode"]:
        """Return all literal nodes from all children."""
        return [lit for child in self.children for lit in child.flatten()]

    @model_validator(mode="after")
    def validate_and_node(self):
        """Validate the AND node structure and constraints."""
        return _validate_compound_node(self, "and")


class OrNode(BaseModel):
    """An OR node combining multiple rule nodes."""
    
    type: Literal["or"]
    children: list["RuleNode"] = Field(..., min_items=MIN_CHILDREN_PER_NODE,
                                       max_items=MAX_CHILDREN_PER_NODE)

    def flatten(self) -> list["LiteralNode"]:
        """Return all literal nodes from all children."""
        return [lit for child in self.children for lit in child.flatten()]

    @model_validator(mode="after")
    def validate_or_node(self):
        """Validate the OR node structure and constraints."""
        return _validate_compound_node(self, "or")


class NotNode(BaseModel):
    """A NOT node negating a single rule node."""
    
    type: Literal["not"]
    child: "RuleNode"

    def flatten(self) -> list["LiteralNode"]:
        """Return all literal nodes from the child."""
        return self.child.flatten()

    @model_validator(mode="after")
    def validate_not_node(self):
        """Validate the NOT node has a child."""
        if not self.child:
            raise ValueError("NotNode must have a single child")
        return self


# Now define RuleNode after all the types it depends on
RuleNode = Annotated[
    LiteralNode | AndNode | OrNode | NotNode,
    Field(discriminator="type")
]


class LogicalRule(BaseModel):
    """Logical rule with both a rule and reasoning."""

    reasoning: str = Field(description="Reasoning for rule creation.")
    rule: RuleNode


# Rebuild forward references to resolve strings
LiteralNode.model_rebuild()
AndNode.model_rebuild()
OrNode.model_rebuild()
NotNode.model_rebuild()
LogicalRule.model_rebuild()


def get_rules_prompt(
    prompt_template_file: str, 
    prompt_data: PromptInformation,
    markets: dict,
    existing_rules: list[str] = None
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
        existing_rules=existing_rules or [],
    )


def clean_and_parse_json(raw_response: str) -> tuple[list[dict] | None, str]:
    """
    Clean and parse JSON response from LLM, handling common malformations.
    
    Args:
        raw_response: Raw string response from LLM
        
    Returns:
        Tuple of (parsed_data, error_message)

    """
    try:
        # Remove any text before the first [ and after the last ]
        json_match = re.search(r'\[.*\]', raw_response, re.DOTALL)
        if not json_match:
            return None, "No JSON array found in response"
            
        json_str = json_match.group(0)
        
        # Common fixes for malformed JSON
        # Fix trailing commas
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        # Fix single quotes to double quotes
        json_str = re.sub(r"(?<!\\)'", '"', json_str)
        # Fix unescaped quotes in strings (basic attempt)
        json_str = re.sub(r'"([^"]*?)"([^":,}\]\s])', r'"\1\\"\2', json_str)
        
        # Try to parse
        parsed = json.loads(json_str)
        
        if not isinstance(parsed, list):
            return None, "Response is not a JSON array"
            
        return parsed, ""
        
    except json.JSONDecodeError as e:
        return None, f"JSON parsing error: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error parsing JSON: {str(e)}"


def validate_rule_structure(rule_data: dict, valid_market_ids: set[str]) -> tuple[bool, str]:
    """
    Validate rule structure before Pydantic parsing.
    
    Args:
        rule_data: Raw rule dictionary from LLM
        valid_market_ids: Set of valid market IDs
        
    Returns:
        Tuple of (is_valid, error_message)

    """
    try:
        if "rule" not in rule_data:
            return False, "Missing 'rule' field"
            
        rule = rule_data["rule"]
        if "type" not in rule:
            return False, "Missing 'type' field in rule"
            
        rule_type = rule["type"]
        
        if rule_type == "literal":
            if "name" not in rule:
                return False, "Missing 'name' field in literal"
            if rule["name"] not in valid_market_ids:
                return False, f"Invalid market ID: {rule['name']}"
                
        elif rule_type in ["and", "or"]:
            if "children" not in rule:
                return False, f"Missing 'children' field in {rule_type}"
            children = rule["children"]
            if not isinstance(children, list):
                return False, f"Children must be a list in {rule_type}"
            if len(children) < MIN_CHILDREN_PER_NODE or len(children) > MAX_CHILDREN_PER_NODE:
                return False, (f"{rule_type} must have "
                               f"{MIN_CHILDREN_PER_NODE}-{MAX_CHILDREN_PER_NODE} children")
                
            # Recursively validate children
            for child in children:
                child_valid, child_error = validate_rule_structure({"rule": child},
                                                                   valid_market_ids)
                if not child_valid:
                    return False, f"Invalid child in {rule_type}: {child_error}"
                    
        elif rule_type == "not":
            if "child" not in rule:
                return False, "Missing 'child' field in not"
            child_valid, child_error = validate_rule_structure({"rule": rule["child"]},
                                                               valid_market_ids)
            if not child_valid:
                return False, f"Invalid child in not: {child_error}"
        else:
            return False, f"Unknown rule type: {rule_type}"
            
        return True, ""
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def get_rules_chunked(
    prompt: str,
    valid_market_ids: set[str],
    client: instructor.Instructor,
    total_rules: int,
    chunk_size: int = 10,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_retries: int = DEFAULT_MAX_RETRIES
) -> list[LogicalRule]:
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
        List of validated LogicalRules

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
) -> list[LogicalRule]:
    """Generate a single chunk of rules with robust error handling."""
    for attempt in range(max_retries + 1):
        try:
            # Try with Instructor first (structured output)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": prompt}],
                response_model=list[LogicalRule],
                max_retries=1,
                temperature=temperature,
                top_p=0.9,
                frequency_penalty=0.5,
                presence_penalty=0.3
            )
            
            # Validate each rule
            validated_rules = []
            for rule in response:
                rule_dict = {"rule": rule.rule.model_dump()}
                is_valid, error = validate_rule_structure(rule_dict, valid_market_ids)
                if is_valid:
                    validated_rules.append(rule)
                    
            if validated_rules:
                return validated_rules
            else:
                raise ValueError("No valid rules in structured response")
                
        except Exception as struct_error:
            # Fallback to raw text generation with manual parsing
            try:
                raw_response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": prompt}],
                    temperature=temperature,
                    max_tokens=4000
                )
                
                raw_text = raw_response.choices[0].message.content
                parsed_data, parse_error = clean_and_parse_json(raw_text)
                
                if parsed_data is None:
                    raise ValueError(f"JSON parsing failed: {parse_error}")
                    
                # Manually create LogicalRule objects
                validated_rules = []
                for rule_data in parsed_data:
                    try:
                        if "reasoning" not in rule_data or "rule" not in rule_data:
                            continue
                            
                        # Validate structure first
                        is_valid, error = validate_rule_structure(rule_data, valid_market_ids)
                        if not is_valid:
                            continue
                            
                        # Create LogicalRule object
                        logical_rule = LogicalRule(
                            reasoning=rule_data["reasoning"],
                            rule=rule_data["rule"]
                        )
                        validated_rules.append(logical_rule)
                        
                    except Exception as e:
                        raise RuntimeError("Failed to extract rules") from e
                if validated_rules:
                    return validated_rules
                else:
                    raise ValueError("No valid rules from manual parsing")
                    
            except Exception as fallback_error:
                if attempt == max_retries:
                    raise Exception(
                        f"Both structured and fallback generation failed after "
                        f"{max_retries + 1} attempts. Last errors: "
                        f"Structured: {struct_error}, Fallback: {fallback_error}"
                    ) from fallback_error
                    
                # Adjust parameters for retry
                temperature = min(1.0, temperature + 0.1)

    raise Exception("Unexpected error in rule generation")


def get_rules(
    prompt: str, 
    valid_market_ids: set[str],
    client: instructor.Instructor,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_retries: int = DEFAULT_MAX_RETRIES
) -> list[LogicalRule]:
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
        A list of LogicalRules.
        
    Raises:
        Exception: If rule generation fails after all retries.

    """
    # Extract number of rules from prompt
    import re
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