import json
from typing import Dict, List, Any

def format_sources(sources: List[Dict[str, Any]]) -> str:
    """Format sources into a readable string with numbered references."""
    if not sources:
        return "No sources found."
    
    formatted = "Sources:\n"
    for i, source in enumerate(sources, 1):
        formatted += f"{i}. {source.get('title', 'Untitled')}\n"
        formatted += f"   URL: {source.get('url', 'No URL')}\n"
        formatted += f"   Published: {source.get('published_date', 'Unknown date')}\n\n"
    
    return formatted

def extract_key_info(text: str, max_length: int = 1000) -> str:
    """Extract key information from longer texts."""
    if len(text) <= max_length:
        return text
    
    
    return text[:max_length] + "..."

def merge_research_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple research results into a consolidated format."""
    merged = {
        "sources": [],
        "extracted_information": "",
        "key_findings": []
    }
    
    all_texts = []
    
    for result in results:
        
        for source in result.get("sources", []):
            if source not in merged["sources"]:
                merged["sources"].append(source)
        
        
        if "extracted_information" in result:
            all_texts.append(result["extracted_information"])
        
        
        if "key_findings" in result:
            merged["key_findings"].extend(result["key_findings"])
    
    
    merged["extracted_information"] = "\n\n".join(all_texts)
    
    return merged