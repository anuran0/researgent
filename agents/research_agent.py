from typing import Dict, List, Any, Optional
import google.generativeai as genai
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from config import GEMINI_API_KEY, RESEARCH_AGENT_MODEL
from tools.tavily_tools import TavilySearchTool
from utils.helpers import extract_key_info, format_sources

genai.configure(api_key=GEMINI_API_KEY)

class ResearchAgent:
    """Agent responsible for researching information on a given topic."""

    def __init__(self):
        self.search_tool = TavilySearchTool()
        self.llm = ChatGoogleGenerativeAI(model=RESEARCH_AGENT_MODEL, google_api_key=GEMINI_API_KEY)

        self.system_prompt = """You are an expert research agent. Your task is to gather comprehensive information on topics by:
1. Breaking down complex queries into specific research questions
2. Identifying key subtopics to explore
3. Finding reliable and relevant sources
4. Extracting and summarizing the most important information
5. Organizing findings in a structured format

Be thorough, objective, and precise. Focus on factual information and cite all sources properly.
"""

    async def generate_research_plan(self, query: str) -> Dict[str, Any]:
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""I need to research: \"{query}\"

Please help me create a research plan by:
1. Breaking this topic into 3-5 specific research questions
2. Identifying 3-7 key subtopics to explore
3. Suggesting search terms that would yield the most relevant results

Format your response as a JSON with the following structure:
{{
\"research_questions\": [\"question1\", \"question2\", ...],
\"subtopics\": [\"subtopic1\", \"subtopic2\", ...],
\"search_terms\": [\"term1\", \"term2\", ...]
}}""")
        ]

        response = await self.llm.ainvoke(messages)
        response_text = response.content

        try:
            import json
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                return json.loads(response_text[start_idx:end_idx])
        except Exception:
            pass

        return {
            "research_questions": [query],
            "subtopics": [],
            "search_terms": [query]
        }

    async def execute_research(self, query: str) -> Dict[str, Any]:
        research_plan = await self.generate_research_plan(query)
        search_results = []

        main_results = await self.search_tool.search(query, search_depth=2)
        search_results.append(main_results)

        for question in research_plan.get("research_questions", []):
            if question != query:
                results = await self.search_tool.search(question)
                search_results.append(results)

        for subtopic in research_plan.get("subtopics", [])[:3]:
            sub_query = f"{query} {subtopic}"
            results = await self.search_tool.search(sub_query)
            search_results.append(results)

        synthesis = await self._synthesize_research(query, search_results, research_plan)

        return {
            "query": query,
            "research_plan": research_plan,
            "raw_search_results": search_results,
            "synthesis": synthesis
        }

    async def _synthesize_research(self, query: str, search_results: List[Dict[str, Any]], research_plan: Dict[str, Any]) -> Dict[str, Any]:
        all_sources = []
        all_answers = []

        for result in search_results:
            for source in result.get("sources", []):
                if source not in all_sources:
                    all_sources.append(source)
            if result.get("answer"):
                all_answers.append(result["answer"])

        source_excerpts = "\n\n".join([
            f"Source {i+1}: {source.get('title', 'Untitled')}\n{extract_key_info(source.get('content', ''))}"
            for i, source in enumerate(all_sources[:10])
        ])

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
I've researched the topic: \"{query}\"

Here are excerpts from the most relevant sources:

{source_excerpts}

Based on these sources, please provide:
1. A comprehensive summary of the key findings
2. The main points related to each of these subtopics: {', '.join(research_plan.get('subtopics', []))}
3. Any contradictions or knowledge gaps identified
4. A list of the most credible sources from the research

Format your response as a JSON with the following structure:
{{
  \"summary\": \"comprehensive summary here\",
  \"key_findings\": [\"finding1\", \"finding2\", ...],
  \"subtopic_analysis\": {{\"subtopic1\": \"analysis1\", ...}},
  \"contradictions_gaps\": [\"contradiction1\", \"gap1\", ...],
  \"top_sources\": [1, 4, 7]
}}
""")
        ]

        response = await self.llm.ainvoke(messages)
        response_text = response.content

        try:
            import json
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                synthesis_results = json.loads(response_text[start_idx:end_idx])
            else:
                synthesis_results = {
                    "summary": " ".join(all_answers),
                    "key_findings": [],
                    "subtopic_analysis": {},
                    "contradictions_gaps": [],
                    "top_sources": []
                }
        except Exception:
            synthesis_results = {
                "summary": " ".join(all_answers),
                "key_findings": [],
                "subtopic_analysis": {},
                "contradictions_gaps": [],
                "top_sources": []
            }

        synthesis_results["sources"] = all_sources
        return synthesis_results
