"""
fake_news_module/explainability/__init__.py
===========================================
Phase 3: Explainability Engine sub-package.

Provides human-readable explanations for the pipeline's verdict by
analysing every signal's contribution: RoBERTa, FAISS similarity,
Google Fact Check, NewsAPI, Groq, heuristic analysis, and source credibility.

Public API
----------
    from fake_news_module.explainability.explanation_engine import generate_explanation
"""
