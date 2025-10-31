"""
Meta-prompt generation for dynamic task-specific prompts.

Authors:
- Erik Arnold <ewarnold@uvm.edu>
- JGY <jyoung22@uvm.edu>
"""

from datetime import UTC, datetime
from os import path

import instructor
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from models.markets import RelevancePrompt


class MetaPromptResponse(BaseModel):
    """Structured response from meta-prompt generation."""
    
    generated_prompt: str = Field(
        description="The complete relevance evaluation prompt for the given index question"
    )


def generate_meta_prompt(
    overall_index_question: str,
    current_date: datetime,
    client: instructor.Instructor,
    model: str = "gpt-4o",
    temperature: float = 0.0,
) -> str:
    """Generate a task-specific relevance evaluation prompt using meta-prompting."""
    base_dir = path.dirname(path.abspath(__file__))
    templates_dir = path.join(base_dir, "prompts")
    env = Environment(
        loader=FileSystemLoader(templates_dir),
        autoescape=True
    )
    template = env.get_template("meta_prompt.j2")
    meta_prompt = template.render(
        overall_index_question=overall_index_question,
        date=current_date
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": meta_prompt}],
        response_model=MetaPromptResponse,
        max_retries=3,
        temperature=temperature,
    )
    
    return response.generated_prompt


def get_or_create_relevance_prompt(
    session: Session,
    label_type_id: int,
    index_question: str,
    current_date: datetime,
    client: instructor.Instructor,
    force_regenerate: bool = False,
) -> RelevancePrompt:
    """Get existing relevance prompt or generate a new one if needed."""
    if not force_regenerate:
        existing = session.exec(
            select(RelevancePrompt)
            .where(
                RelevancePrompt.label_type_id == label_type_id,
                RelevancePrompt.index_question == index_question
            )
            .order_by(RelevancePrompt.created_at.desc())
        ).first()
        
        if existing:
            return existing
    
    generated_prompt = generate_meta_prompt(
        overall_index_question=index_question,
        current_date=current_date,
        client=client,
    )
    
    new_prompt = RelevancePrompt(
        prompt=generated_prompt,
        label_type_id=label_type_id,
        index_question=index_question,
        created_at=datetime.now(UTC),
    )
    
    session.add(new_prompt)
    session.flush()
    
    return new_prompt
