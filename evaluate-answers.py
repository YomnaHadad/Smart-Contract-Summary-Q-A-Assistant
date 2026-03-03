import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import json

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=groq_api_key,
    temperature=0
)

import re

def evaluate_answer(question: str, answer: str, sources: str) -> dict:

    template = """
You are a strict automated judge.

Evaluate the answer using ONLY the provided sources.

Question:
{question}

Answer:
{answer}

Sources:
{sources}

Respond EXACTLY in this format:

verdict: Correct or Incorrect
score: number between 0 and 100
explanation: short explanation
"""

    prompt = ChatPromptTemplate.from_template(template)

    messages = prompt.format_messages(
        question=question,
        answer=answer,
        sources=sources
    )

    try:
        response = llm.invoke(messages)
        content = response.content.strip()

        
        verdict_match = re.search(r"verdict:\s*(Correct|Incorrect)", content, re.IGNORECASE)
        score_match = re.search(r"score:\s*(\d+)", content)
        explanation_match = re.search(r"explanation:\s*(.*)", content, re.IGNORECASE)

        verdict = verdict_match.group(1) if verdict_match else "Error"
        score = int(score_match.group(1)) if score_match else 0
        explanation = explanation_match.group(1) if explanation_match else content

        return {
            "verdict": verdict,
            "score": score,
            "explanation": explanation
        }

    except Exception as e:
        return {
            "verdict": "Error",
            "score": 0,
            "explanation": str(e)
        }
