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

from models.markets import IndexQuestion, Prompt, PromptPurpose


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
    index_question_id: int,
    current_date: datetime,
    client: instructor.Instructor,
    force_regenerate: bool = False,
) -> Prompt:
    """Get existing relevance prompt or generate a new one if needed."""
    # Get the index question text
    index_question_obj = session.exec(
        select(IndexQuestion).where(IndexQuestion.id == index_question_id)
    ).first()
    
    if not index_question_obj:
        raise ValueError(f"IndexQuestion with id {index_question_id} not found")
    
    if not force_regenerate:
        existing = session.exec(
            select(Prompt)
            .where(
                Prompt.index_question_id == str(index_question_id),
                Prompt.purpose == PromptPurpose.RELEVANCE
            )
            .order_by(Prompt.created_at.desc())
        ).first()
        
        if existing:
            return existing
    
    generated_prompt = generate_meta_prompt(
        overall_index_question=index_question_obj.question,
        current_date=current_date,
        client=client,
    )
    
    new_prompt = Prompt(
        prompt=generated_prompt,
        index_question_id=str(index_question_id),
        purpose=PromptPurpose.RELEVANCE,
        created_at=datetime.now(UTC),
    )
    
    session.add(new_prompt)
    session.flush()
    
    return new_prompt
