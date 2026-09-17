"""LangGraph main graph construction"""
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
import json
import re

from config import make_chat_llm
from state import AgentState
from nodes.researcher import researcher_node
from nodes.prompt_gen import prompt_gen_node
from nodes.execution import execution_node


def build_morph_agent_graph(temperature: float = 0.0):
    """Build the LangGraph graph for MorphAgent

    Args:
        temperature: The temperature parameter for the LLM when generating features (default 0.0; higher values make the output more random)
    """
    from config import settings
    kwargs = {"temperature": temperature}
    if settings.reproduce_mode:
        kwargs["seed"] = settings.reproduce_seed
    llm = make_chat_llm(**kwargs)
    
    def planning_node(state: AgentState):
        """Planning node: use the LLM to generate the feature plan"""
        feature_plan = state.get("feature_plan")
        if not feature_plan or not isinstance(feature_plan, dict):
            return {"current_step": "error", "feature_plan": None}
        
        filled_template = feature_plan.get("template_filled", "")
        if not filled_template:
            return {"current_step": "error", "feature_plan": feature_plan}
        
        print(f"\n[Planning] Calling LLM to generate the feature plan...")
        
        # Print the first 5 and last 5 lines of the input
        print(f"[Planning] Input content (first 5 lines):")
        input_lines = filled_template.split('\n')
        for i, line in enumerate(input_lines[:5], 1):
            print(f"  {i}: {line[:100]}..." if len(line) > 100 else f"  {i}: {line}")
        
        if len(input_lines) > 10:
            print(f"  ... (omitting {len(input_lines) - 10} lines) ...")
            print(f"[Planning] Input content (last 5 lines):")
            for i, line in enumerate(input_lines[-5:], len(input_lines) - 4):
                print(f"  {i}: {line[:100]}..." if len(line) > 100 else f"  {i}: {line}")
        else:
            print(f"[Planning] Input content (last 5 lines):")
            for i, line in enumerate(input_lines[-5:], max(1, len(input_lines) - 4)):
                print(f"  {i}: {line[:100]}..." if len(line) > 100 else f"  {i}: {line}")
        
        response = llm.invoke([HumanMessage(content=filled_template)])
        content = response.content
        
        # Print the first 5 and last 5 lines of the output
        print(f"\n[Planning] Output content (first 5 lines):")
        output_lines = content.split('\n')
        for i, line in enumerate(output_lines[:5], 1):
            print(f"  {i}: {line[:100]}..." if len(line) > 100 else f"  {i}: {line}")
        
        if len(output_lines) > 10:
            print(f"  ... (omitting {len(output_lines) - 10} lines) ...")
            print(f"[Planning] Output content (last 5 lines):")
            for i, line in enumerate(output_lines[-5:], len(output_lines) - 4):
                print(f"  {i}: {line[:100]}..." if len(line) > 100 else f"  {i}: {line}")
        else:
            print(f"[Planning] Output content (last 5 lines):")
            for i, line in enumerate(output_lines[-5:], max(1, len(output_lines) - 4)):
                print(f"  {i}: {line[:100]}..." if len(line) > 100 else f"  {i}: {line}")
        
        # Parse JSON
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            try:
                features = json.loads(json_match.group(0))
                feature_plan["features"] = features
                print(f"[Planning] Generated {len(features)} features")
            except json.JSONDecodeError:
                feature_plan["features"] = []
        else:
            feature_plan["features"] = []
        
        return {
            "feature_plan": feature_plan,
            "current_step": "planning",
        }
    
    def synthesis_node(state: AgentState):
        """Synthesis node: aggregate results"""
        analysis_results = state.get("analysis_results", {})
        return {
            "current_step": "synthesis",
            "analysis_results": analysis_results,
        }
    
    # Build the graph
    workflow = StateGraph(AgentState)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("prompt_gen", prompt_gen_node)
    workflow.add_node("planning", planning_node)
    workflow.add_node("execution", execution_node)
    workflow.add_node("synthesis", synthesis_node)
    
    workflow.add_edge(START, "researcher")
    workflow.add_edge("researcher", "prompt_gen")
    workflow.add_edge("prompt_gen", "planning")
    workflow.add_edge("planning", "execution")
    workflow.add_edge("execution", "synthesis")
    workflow.add_edge("synthesis", END)
    
    return workflow.compile()
