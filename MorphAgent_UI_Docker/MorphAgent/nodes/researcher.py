"""Researcher node - knowledge retrieval"""
from typing import Dict, Any
from state import AgentState


def researcher_node(state: AgentState) -> Dict[str, Any]:
    """Knowledge retrieval node (debug mode: skipped)"""
    user_query = state.get("user_query", "")
    research_summary = f"Debug mode: RAG not enabled. User query: {user_query}"
    
    return {
        "research_summary": research_summary,
        "expert_examples": [],
        "current_step": "research",
    }
