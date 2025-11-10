"""
Tool Wrappers for LLM Agent

Wraps existing tools (HTTP fetcher, vector store, executor) with
structured interfaces for LLM tool calling.
"""

from typing import Dict, Any, List
from packages.agents.llm_agent import Tool
from packages.tools import http_fetcher, vector_store, executor
from packages.agents.reliability import log_step


def create_http_fetcher_tool() -> Tool:
    """Create a tool wrapper for HTTP fetcher"""
    def fetch_url(url: str) -> Dict[str, Any]:
        """Fetch content from a URL. Returns the fetched text and metadata."""
        result = http_fetcher.fetch(url)
        return {
            "url": result["url"],
            "status": result["status"],
            "text": result["text"][:5000],  # Limit text size
            "from_cache": result.get("from_cache", False),
            "text_length": len(result["text"])
        }
    
    return Tool(
        name="fetch_url",
        description="Fetch content from a URL. Useful for getting README files, documentation, or code from GitHub.",
        func=fetch_url,
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch (must be http:// or https://)"
                }
            },
            "required": ["url"]
        }
    )


# Global vector store instance (shared across tool calls)
_global_vs_instance = None

def get_vector_store():
    """Get or create global vector store instance"""
    global _global_vs_instance
    if _global_vs_instance is None:
        _global_vs_instance = vector_store.MockVectorStore()
    return _global_vs_instance

def create_vector_search_tool() -> Tool:
    """Create a tool wrapper for vector store search"""
    
    def search_vector_store(query: str, top_k: int = 3) -> Dict[str, Any]:
        """Search the vector store for relevant documentation or code context."""
        vs = get_vector_store()
        results = vs.search(query, top_k=top_k)
        return {
            "query": query,
            "results": [
                {
                    "doc_id": doc_id,
                    "score": float(score),
                    "text_preview": vs._index.get(doc_id, {}).get("text", "")[:500]
                }
                for doc_id, score in results
            ]
        }
    
    # Return search tool
    return Tool(
        name="search_documentation",
        description="Search previously indexed documentation or code for relevant context about the repository.",
        func=search_vector_store,
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to find relevant documentation"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 3)",
                    "default": 3
                }
            },
            "required": ["query"]
        }
    )


def create_executor_tool() -> Tool:
    """Create a tool wrapper for code executor/syntax checker"""
    def check_syntax(code: str) -> Dict[str, Any]:
        """Check if Python code has valid syntax. Does not execute the code."""
        result = executor.syntax_check(code)
        return {
            "ok": result.get("ok", False),
            "message": result.get("message", "Unknown result"),
            "code_length": len(code)
        }
    
    return Tool(
        name="check_code_syntax",
        description="Check if Python code has valid syntax. Use this to validate generated test code before proposing it.",
        func=check_syntax,
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to syntax-check"
                }
            },
            "required": ["code"]
        }
    )


def register_all_tools(agent) -> None:
    """Register all available tools with the LLM agent"""
    agent.register_tool(create_http_fetcher_tool())
    agent.register_tool(create_vector_search_tool())
    agent.register_tool(create_executor_tool())
    log_step("tool_wrappers", "Registered all tools with LLM agent")

