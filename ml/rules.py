"""
Rule generator for index.

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


# --- Data Model for Prompt Input ---
class PromptInformation(BaseModel):
    """Structured model for Disease information."""

    disease: str = Field(description="What disease we are interested in.")
    date: datetime = Field(description="The date we are interested in.")
    overall_index_question: str = Field(description="Overall index question.")
    num_of_rules: int = Field(description="Number of rules to be generated.")



class LiteralNode(BaseModel):
    type: Literal["literal"]
    name: str

    def flatten(self) -> list["LiteralNode"]:
        return [self]


class AndNode(BaseModel):
    type: Literal["and"]
    children: list["RuleNode"] = Field(..., min_items=2, max_items=3)

    def flatten(self) -> list["LiteralNode"]:
        return [lit for child in self.children for lit in child.flatten()]

    @model_validator(mode="after")
    def validate_and_node(self):
        def depth(node, level=1):
            if isinstance(node, LiteralNode):
                return level
            elif hasattr(node, "children") and node.children:
                if level >= 2:
                    return level
                return max(depth(c, level + 1) for c in node.children)
            return level

        max_depth = max(depth(c) for c in self.children)
        if max_depth > 2:
            raise ValueError("Nesting too deep; max depth is 2.")

        all_literals = {lit.name for lit in self.flatten()}
        if len(all_literals) == 0:
            raise ValueError("AndNode must contain at least one literal")
        if len(all_literals) > 3:
            raise ValueError(f"Too many unique literals in 'and' node: "
                             f"{len(all_literals)}. Max is 3.")
        return self


class OrNode(BaseModel):
    type: Literal["or"]
    children: list["RuleNode"] = Field(..., min_items=2, max_items=3)

    def flatten(self) -> list["LiteralNode"]:
        return [lit for child in self.children for lit in child.flatten()]

    @model_validator(mode="after")
    def validate_or_node(self):
        def depth(node, level=1):
            if isinstance(node, LiteralNode):
                return level
            elif hasattr(node, "children") and node.children:
                if level >= 2:
                    return level
                return max(depth(c, level + 1) for c in node.children)
            return level

        max_depth = max(depth(c) for c in self.children)
        if max_depth > 2:
            raise ValueError("Nesting too deep; max depth is 2.")

        all_literals = {lit.name for lit in self.flatten()}
        if len(all_literals) == 0:
            raise ValueError("OrNode must contain at least one literal")
        if len(all_literals) > 3:
            raise ValueError(f"Too many unique literals in 'or' node: {len(all_literals)}. Max is 3.")
        return self


class NotNode(BaseModel):
    type: Literal["not"]
    child: "RuleNode"

    def flatten(self) -> list["LiteralNode"]:
        return self.child.flatten()

    @model_validator(mode="after")
    def validate_not_node(self):
        if not self.child:
            raise ValueError("NotNode must have a single child")
        return self


# Now define RuleNode after all the types it depends on
RuleNode = Annotated[
    LiteralNode | AndNode | OrNode | NotNode,
    Field(discriminator="type")
]

# Top-level wrapper
class LogicalRule(BaseModel):
    """Logical rule with both a rule and reasoning"""

    reasoning: str = Field(description="Reasoning for rule creation.")
    rule: RuleNode

# Rebuild forward references to resolve strings
LiteralNode.model_rebuild()
AndNode.model_rebuild()
OrNode.model_rebuild()
NotNode.model_rebuild()
LogicalRule.model_rebuild()



def get_rules_prompt(prompt_template_file: str, prompt_data: PromptInformation) -> str:
    """
    Use a template file to generate a prompt.

    Args:
        prompt_template_file: template file to use.
        prompt_data: prompt_information about the event.

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
    return template.render(disease = prompt_data.disease,
                           date = prompt_data.date,
                           overall_index_question = prompt_data.overall_index_question,
                           num_of_rules = prompt_data.num_of_rules,)



def get_rules(prompt: str, market_text_representation: str,
                    client: instructor.Instructor) -> list[LogicalRule]:
    """
    Get rules using eligible markets.

    Args:
        prompt: The system-level instruction or prompt for ranking.
        market_text_representation: A text representation of the market.
        client: An Instructor-enhanced OpenAI client.

    Returns:
        A list of LogicalRules.

    """
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": market_text_representation}
        ],
        response_model=list[LogicalRule],
        max_retries=3,
        temperature=0.8,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.3
    )
    return response
