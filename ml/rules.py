"""Rule generator for index.

Authors:
- JGY <jyoung22@uvm.edu>
- Erik Arnold <ewarnold@uvm.edu>
"""

from datetime import datetime
from os import path
from typing import Annotated, Literal

import instructor
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field, model_validator

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
    """Calculate the maximum depth of a rule node tree.
    
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
    """Validate compound nodes (AND/OR) for depth and literal constraints.
    
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
    children: list["RuleNode"] = Field(..., min_items=MIN_CHILDREN_PER_NODE, max_items=MAX_CHILDREN_PER_NODE)

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
    children: list["RuleNode"] = Field(..., min_items=MIN_CHILDREN_PER_NODE, max_items=MAX_CHILDREN_PER_NODE)

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
    """Use a template file to generate a prompt.

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


def validate_rule_structure(rule_data: dict, valid_market_ids: set[str]) -> tuple[bool, str]:
    """Validate rule structure before Pydantic parsing.
    
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
                return False, f"{rule_type} must have {MIN_CHILDREN_PER_NODE}-{MAX_CHILDREN_PER_NODE} children"
                
            # Recursively validate children
            for child in children:
                child_valid, child_error = validate_rule_structure({"rule": child}, valid_market_ids)
                if not child_valid:
                    return False, f"Invalid child in {rule_type}: {child_error}"
                    
        elif rule_type == "not":
            if "child" not in rule:
                return False, "Missing 'child' field in not"
            child_valid, child_error = validate_rule_structure({"rule": rule["child"]}, valid_market_ids)
            if not child_valid:
                return False, f"Invalid child in not: {child_error}"
        else:
            return False, f"Unknown rule type: {rule_type}"
            
        return True, ""
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def get_rules(
    prompt: str, 
    valid_market_ids: set[str],
    client: instructor.Instructor,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_retries: int = DEFAULT_MAX_RETRIES
) -> list[LogicalRule]:
    """Get rules using eligible markets with improved validation.

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
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt}
                ],
                response_model=list[LogicalRule],
                max_retries=2,  # Lower retries per attempt
                temperature=temperature,
                top_p=0.9,
                frequency_penalty=0.5,
                presence_penalty=0.3
            )
            
            # Validate each rule before returning
            validated_rules = []
            for rule in response:
                rule_dict = {"rule": rule.rule.model_dump()}
                is_valid, error = validate_rule_structure(rule_dict, valid_market_ids)
                if is_valid:
                    validated_rules.append(rule)
                else:
                    print(f"Warning: Skipping invalid rule - {error}")
                    
            if validated_rules:
                return validated_rules
            else:
                raise ValueError("No valid rules generated")
                
        except Exception as e:
            if attempt == max_retries:
                raise Exception(f"Failed to generate rules after {max_retries + 1} attempts: {e}") from e
            print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
            # Increase temperature slightly for retry
            temperature = min(1.0, temperature + 0.1)
            
    raise Exception("Unexpected error in rule generation")
