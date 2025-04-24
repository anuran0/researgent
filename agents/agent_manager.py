from typing import Dict, List, Any, TypedDict
import asyncio
import operator
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
from agents.research_agent import ResearchAgent
from agents.answer_agent import AnswerAgent
from utils.helpers import merge_research_results

class ResearchState(TypedDict):
    query: str
    research_plan: Dict[str, Any]
    research_results: Dict[str, Any]
    draft_answer: Dict[str, Any]
    final_answer: Dict[str, Any]
    style: str
    error: str

class AgentManager:
    """Manager for coordinating multiple agents in the research system."""

    def __init__(self):
        self.research_agent = ResearchAgent()
        self.answer_agent = AnswerAgent()

    async def process_query(self, query: str, style: str = "academic") -> Dict[str, Any]:
        research_results = await self.research_agent.execute_research(query)
        draft_answer = await self.answer_agent.draft_answer(query, research_results)
        final_answer = await self.answer_agent.refine_answer(draft_answer, style)

        return {
            "query": query,
            "answer": final_answer.get("refined_answer", final_answer.get("draft_answer", "")),
            "style": style,
            "sources": final_answer.get("sources", []),
            "metadata": {
                "research_plan": research_results.get("research_plan", {}),
                "key_findings": research_results.get("synthesis", {}).get("key_findings", []),
                "contradictions_gaps": research_results.get("synthesis", {}).get("contradictions_gaps", [])
            }
        }

    async def multi_agent_research(self, query: str, num_agents: int = 2) -> Dict[str, Any]:
        if num_agents < 1:
            num_agents = 1
        elif num_agents > 5:
            num_agents = 5

        plan = await self.research_agent.generate_research_plan(query)
        subtopics = plan.get("subtopics", [])

        tasks = [self.research_agent.execute_research(query)]

        for i in range(min(len(subtopics), num_agents - 1)):
            sub_query = f"{query} - {subtopics[i]}"
            tasks.append(self.research_agent.execute_research(sub_query))

        results = await asyncio.gather(*tasks)
        all_synthesis = [result.get("synthesis", {}) for result in results]
        merged_synthesis = merge_research_results(all_synthesis)

        return {
            "query": query,
            "research_plan": plan,
            "synthesis": merged_synthesis
        }

    def build_research_graph(self):
        graph = StateGraph(ResearchState)

        async def generate_plan(state: ResearchState) -> ResearchState:
            try:
                research_plan = await self.research_agent.generate_research_plan(state["query"])
                return {"research_plan": research_plan}
            except Exception as e:
                return {"error": f"Error in research plan generation: {str(e)}"}

        async def execute_research(state: ResearchState) -> ResearchState:
            try:
                query = state["query"]
                plan = state["research_plan"]
                if len(plan.get("subtopics", [])) > 0:
                    research_results = await self.multi_agent_research(query, num_agents=3)
                else:
                    research_results = await self.research_agent.execute_research(query)
                return {"research_results": research_results}
            except Exception as e:
                return {"error": f"Error in research execution: {str(e)}"}

        async def draft_answer_node(state: ResearchState) -> ResearchState:
            try:
                draft = await self.answer_agent.draft_answer(state["query"], state["research_results"])
                return {"draft_answer": draft}
            except Exception as e:
                return {"error": f"Error in answer drafting: {str(e)}"}

        async def refine_answer(state: ResearchState) -> ResearchState:
            try:
                refined = await self.answer_agent.refine_answer(state["draft_answer"], state["style"])
                return {"final_answer": refined}
            except Exception as e:
                return {"error": f"Error in answer refinement: {str(e)}"}

        graph.add_node("generate_plan", generate_plan)
        graph.add_node("execute_research", execute_research)
        graph.add_node("draft_answer_node", draft_answer_node)
        graph.add_node("refine_answer", refine_answer)

        graph.add_edge("generate_plan", "execute_research")
        graph.add_edge("execute_research", "draft_answer_node")
        graph.add_edge("draft_answer_node", "refine_answer")
        graph.add_edge("refine_answer", END)

        def should_end(state: ResearchState) -> bool:
            return "error" in state and bool(state["error"])

        graph.add_conditional_edges("generate_plan", should_end, {True: END, False: "execute_research"})
        graph.add_conditional_edges("execute_research", should_end, {True: END, False: "draft_answer_node"})
        graph.add_conditional_edges("draft_answer_node", should_end, {True: END, False: "refine_answer"})

        graph.set_entry_point("generate_plan")
        return graph

    async def run_langgraph_workflow(self, query: str, style: str = "academic") -> Dict[str, Any]:
        if not hasattr(self, "_graph"):
            self._graph = self.build_research_graph()

        workflow = self._graph.compile()

        initial_state = {
            "query": query,
            "research_plan": {},
            "research_results": {},
            "draft_answer": {},
            "final_answer": {},
            "style": style,
            "error": ""
        }

        final_state = await workflow.ainvoke(initial_state)

        if final_state.get("error"):
            return {
                "query": query,
                "error": final_state["error"],
                "answer": f"An error occurred during research: {final_state['error']}"
            }

        final_answer = final_state.get("final_answer", {})
        sources = final_state.get("draft_answer", {}).get("sources", [])

        # Combine final answer with sources
        final_answer["sources"] = sources
        return final_answer

