from typing import Dict, List, Any, Optional
from tavily import TavilyClient
from config import TAVILY_API_KEY, MAX_RESULTS

class TavilySearchTool:
    """Tool for searching the web using Tavily API."""
    
    def __init__(self):
        self.client = TavilyClient(api_key=TAVILY_API_KEY)
    
    async def search(self, query: str, search_depth: int = 1, max_results: int = MAX_RESULTS) -> Dict[str, Any]:
        """
        Perform a search on Tavily.
        
        Args:
            query: The search query
            search_depth: How deep to search (1-3)
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary containing search results and metadata
        """
        try:
            response = self.client.search(
                query=query,
                search_depth=search_depth,
                max_results=max_results,
                include_answer=True,
                include_images=False,
                include_raw_content=True
            )
            
            # Extract and structure the results
            results = {
                "query": query,
                "answer": response.get("answer", ""),
                "sources": response.get("results", []),
                "raw_search_results": response
            }
            
            return results
        
        except Exception as e:
            return {
                "query": query,
                "error": str(e),
                "sources": [],
                "answer": f"Error occurred during search: {str(e)}"
            }
    
    async def deep_search(self, query: str, subtopics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform a deeper search with multiple related queries.
        
        Args:
            query: The main search query
            subtopics: Optional list of subtopics to search
            
        Returns:
            Combined search results
        """
        all_results = []
        
        # Search the main query
        main_results = await self.search(query, search_depth=2)
        all_results.append(main_results)
        
        # If subtopics provided, search each one
        if subtopics:
            for subtopic in subtopics:
                sub_query = f"{query} {subtopic}"
                sub_results = await self.search(sub_query)
                all_results.append(sub_results)
        
        # Combine results
        combined_sources = []
        combined_answer = main_results.get("answer", "")
        
        for result in all_results:
            if "sources" in result:
                # Add only unique sources
                for source in result["sources"]:
                    if source not in combined_sources:
                        combined_sources.append(source)
            
            # Extend the answer if it adds new information
            if "answer" in result and result["answer"] and result["answer"] != combined_answer:
                combined_answer += "\n\nAdditional information: " + result["answer"]
        
        return {
            "query": query,
            "answer": combined_answer,
            "sources": combined_sources,
            "subtopics": subtopics or []
        }