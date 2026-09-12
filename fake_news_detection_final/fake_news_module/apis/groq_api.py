"""
apis/groq_api.py — Groq API integration
Uses Groq's high-speed Llama-3.3 LLM for claim verification.
AI-Driven Digital Evidence Integrity Monitoring System
"""

import logging
import requests

from fake_news_module.config import GROQ_API_KEY, GROQ_API_URL, GROQ_MODEL

logger = logging.getLogger(__name__)


def call_groq_api(claim: str) -> str:
    """Query the Groq API to evaluate a claim and return a simple label.

    Sends the claim to Groq's OpenAI-compatible chat completions endpoint and returns one of:
    'Real', 'Fake', 'Uncertain', or 'Unknown'.
    """
    if not GROQ_API_KEY:
        logger.error("Groq API key not configured")
        return "Unknown"

    if not claim or not claim.strip():
        logger.warning("call_groq_api: empty claim, returning Unknown.")
        return "Unknown"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a fact-checking assistant. "
                    "When given a claim, respond with exactly one word: "
                    "'Real', 'Fake', or 'Uncertain'. No explanation."
                ),
            },
            {
                "role": "user",
                "content": f"Is this claim real or fake? Claim: {claim}",
            },
        ],
        "temperature": 0,
    }

    try:
        response = requests.post(
            GROQ_API_URL,
            headers=headers,
            json=payload,
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        text = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
            .lower()
        )
        logger.info("Groq API raw response: '%s'", text)

        if "real" in text or "true" in text:
            return "Real"
        if "fake" in text or "false" in text:
            return "Fake"
        if "uncertain" in text:
            return "Uncertain"
        return "Unknown"

    except Exception as exc:
        logger.error("Groq API exception: %s", exc)
        return "Unknown"
