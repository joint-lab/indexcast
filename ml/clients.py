"""
Client for LLM calls.

Authors:
- JGY <jyoung22@uvm.edu>
- Erik Arnold <ewarnold@uvm.edu>
"""
import os

import instructor
from openai import OpenAI


def get_client() -> instructor.Instructor:
    """
    Create an instructor client.

    Note: key will be loaded from environment as a variable (OPENAI_API_KEY).
    """
    try:
        api_key = os.environ["OPENAI_API_KEY"]
        return instructor.from_openai(OpenAI(api_key=api_key))
    except Exception as e:
        raise Exception(f"Failed to initialize OpenAI client: {e}") from e
