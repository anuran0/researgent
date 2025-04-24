from typing import Dict, List, Any
import google.generativeai as genai
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from config import GEMINI_API_KEY, ANSWER_AGENT_MODEL
from utils.helpers import format_sources

genai.configure(api_key=GEMINI_API_KEY)

class AnswerAgent:
    """Agent responsible for drafting comprehensive answers based on research."""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model=ANSWER_AGENT_MODEL, google_api_key=GEMINI_API_KEY)

        self.system_prompt = """You are an expert answer drafter specialized in turning research findings into comprehensive, accurate, and well-structured responses. Your task is to:

1. Analyze research findings thoroughly
2. Organize information in a logical and coherent structure
3. Ensure factual accuracy by citing relevant sources
4. Present balanced perspectives on any controversial topics
5. Identify limitations and areas for future research
6. Write in a clear, engaging, and authoritative style

Your answers should be tailored to the specific query while incorporating all relevant information from the research findings. Always cite sources appropriately.
"""

    async def draft_answer(self, query: str, research_results: Dict[str, Any]) -> Dict[str, Any]:
        synthesis = research_results.get("synthesis", {})
        summary = synthesis.get("summary", "")
        key_findings = synthesis.get("key_findings", [])
        subtopic_analysis = synthesis.get("subtopic_analysis", {})
        contradictions_gaps = synthesis.get("contradictions_gaps", [])
        sources = synthesis.get("sources", [])

        formatted_sources = format_sources(sources)
        key_findings_str = "\n".join([f"- {finding}" for finding in key_findings])

        subtopic_str = ""
        for subtopic, analysis in subtopic_analysis.items():
            subtopic_str += f"### {subtopic}\n{analysis}\n\n"

        contradictions_str = "\n".join([f"- {item}" for item in contradictions_gaps])

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
I need a comprehensive answer to the query: \"{query}\"

Here's the research that has been gathered:

SUMMARY:
{summary}

KEY FINDINGS:
{key_findings_str}

SUBTOPIC ANALYSIS:
{subtopic_str}

CONTRADICTIONS AND KNOWLEDGE GAPS:
{contradictions_str}

SOURCES:
{formatted_sources}

Please draft a comprehensive answer that:
1. Directly addresses the query
2. Incorporates all relevant information
3. Is well-structured with clear sections
4. Cites sources appropriately (using [1], [2], etc.)
5. Acknowledges any limitations or areas of uncertainty
6. Provides a balanced view if there are competing perspectives

Your answer should be authoritative, informative, and engaging.
""")
        ]

        response = await self.llm.ainvoke(messages)

        return {
            "query": query,
            "draft_answer": response.content,
            "sources": sources
        }

    async def refine_answer(self, draft_answer: Dict[str, Any], style: str = "academic") -> Dict[str, Any]:
        style_descriptions = {
            "academic": "formal, rigorous, with proper citations and methodology discussion",
            "business": "concise, practical, with actionable insights and executive summary",
            "educational": "clear, pedagogical, with examples and explanations of complex concepts",
            "journalistic": "balanced, engaging, with quotes and contemporary context"
        }

        style_desc = style_descriptions.get(style, style_descriptions["academic"])

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
Please refine the following draft answer to the query: \"{draft_answer.get('query', '')}\"

DRAFT ANSWER:
{draft_answer.get('draft_answer', '')}

I'd like the answer to be in a {style} style that is {style_desc}.

Please preserve all factual information and citations, but restructure and rewrite the content to match the requested style.
""")
        ]

        response = await self.llm.ainvoke(messages)

        refined_answer = draft_answer.copy()
        refined_answer["refined_answer"] = response.content
        refined_answer["style"] = style

        return refined_answer
