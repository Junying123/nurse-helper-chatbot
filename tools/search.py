# search.py
import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults

# Load environment variables from .env file
load_dotenv()

def perform_tavily_search(query):
    """Perform a search using Tavily and return formatted results."""
    
    # Get the API key from environment variables
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    
    if not tavily_api_key:
        raise ValueError("API key not found. Please set the TAVILY_API_KEY environment variable.")
    
    try:
        # Initialize the Tavily Search Tool
        tavily_tool = TavilySearchResults(
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=True
        )
        
        # Invoke the search tool with the query
        response = tavily_tool.invoke({"query": query})
        
        # Prepare search output
        if 'results' in response:
            formatted_results = []
            for result in response['results']:
                formatted_results.append({
                    'url': result.get('url', 'No URL available'),
                    'content': result.get('content', 'No content available')
                })
            return formatted_results
        else:
            return []
    
    except Exception as e:
        print(f"An error occurred while performing the search: {e}")
        return []

