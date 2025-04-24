import asyncio
import argparse
import json
import os
from typing import Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai

from config import GEMINI_API_KEY, TAVILY_API_KEY
from agents.agent_manager import AgentManager

# Configure API keys
load_dotenv()
genai.configure(api_key=GEMINI_API_KEY)

async def main():
    """Main entry point for the Deep Research System."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Deep Research AI Agentic System")
    parser.add_argument("--query", "-q", type=str, help="The research query to process")
    parser.add_argument("--style", "-s", type=str, default="academic", 
                        choices=["academic", "business", "educational", "journalistic"],
                        help="Style for the final answer")
    parser.add_argument("--output", "-o", type=str, help="Output file path (optional)")
    parser.add_argument("--agents", "-a", type=int, default=2, help="Number of research agents to use")
    parser.add_argument("--workflow", "-w", action="store_true", help="Use LangGraph workflow")
    
    args = parser.parse_args()
    
    # Check API keys
    if not TAVILY_API_KEY:
        print("Error: TAVILY_API_KEY not found in environment variables or .env file")
        return
        
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not found in environment variables or .env file")
        return
    
    # Get query from arguments or prompt user
    query = args.query
    if not query:
        query = input("Enter your research query: ")
    
    style = args.style
    num_agents = args.agents
    
    # Initialize agent manager
    manager = AgentManager()
    
    print(f"Starting research on: {query}")
    print(f"Using {style} answer style")
    
    # Process with LangGraph workflow if requested
    if args.workflow:
        print("Using LangGraph workflow for research process...")
        final_response = await manager.run_langgraph_workflow(query, style)
    else:
        # Process with multiple agents if requested
        if num_agents > 1:
            print(f"Conducting multi-agent research with {num_agents} agents...")
            research_results = await manager.multi_agent_research(query, num_agents)
            
            print("Drafting final answer based on multi-agent research...")
            final_response = await manager.answer_agent.draft_answer(query, research_results)
            final_response = await manager.answer_agent.refine_answer(final_response, style)
        else:
            # Process with standard pipeline
            print("Processing query with standard pipeline...")
            final_response = await manager.process_query(query, style)
    
    # Display the answer
    print("\n" + "="*80)
    print(f"ANSWER TO: {query}")
    print("="*80 + "\n")
    
    answer_text = final_response.get("answer", final_response.get("refined_answer", ""))
    print(answer_text)
    
    # Print sources
    print("\n" + "="*80)
    print("SOURCES:")
    print("="*80)
    
    sources = final_response.get("sources", [])
    for i, source in enumerate(sources, 1):
        print(f"{i}. {source.get('title', 'Untitled')}")
        print(f"   URL: {source.get('url', 'No URL')}")
        print(f"   Published: {source.get('published_date', 'Unknown date')}")
        print()
    
    # Save output if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(final_response, f, indent=2)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    asyncio.run(main())