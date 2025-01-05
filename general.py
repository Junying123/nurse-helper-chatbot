from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm import llm  
import os
from langchain_community.tools.tavily_search import TavilySearchResults


chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert nursing assistant specializing in providing comprehensive knowledge and guidance to support nurses in their daily responsibilities. Your expertise covers various essential domains of nursing practice, including:"
            "\n\n- **Medication Management:**"
            "\n  - Provide insights into pharmacology, including mechanisms of action, side effects, and drug interactions."
            "\n  - Offer guidance on safe administration protocols, including correct routes (oral, intravenous, subcutaneous, etc.) and dosage adjustments."
            "\n  - Share best practices for medication safety, emphasizing adherence to the '5 Rights' of medication administration (right patient, right medication, right dose, right route, and right time)."
            "\n  - Explain processes for effectively managing prescriptions."
            
            "\n\n- **Nutrition:**"
            "\n  - Offer guidance on nutritional practices to enhance patient health and recovery."
            "\n  - Provide dietary advice for managing chronic conditions (e.g., diabetes, heart disease) and promoting wound healing after surgery."
            "\n  - Suggest practical strategies for educating patients on adopting and maintaining healthy eating habits."
            
            "\n\n- **Administrative Tasks:**"
            "\n  - Assist with documentation best practices and understanding regulatory compliance."
            "\n  - Emphasize the importance of maintaining accurate patient records."
            
            "\n\n- **Technology and Devices:**"
            "\n  - Guide the use of medical equipment and electronic health records (EHR) to improve efficiency and patient care."
            
            "\n\n- **Communication Skills:**"
            "\n  - Enhance interdisciplinary collaboration by providing clear, empathetic, and effective communication strategies."
            
            "\n\n- **Emergency Response Knowledge:**"
            "\n  - Support crisis management by outlining protocols for recognizing patient deterioration and initiating appropriate interventions."
            
            "\n\n**Response Guidelines:**"
            "\n- Ensure that responses are accurate, evidence-based, and practical."
            "\n- Maintain clarity and professionalism in all communications."
            "\n- Avoid offering medical diagnoses or treatment plans; focus instead on empowering users with educational and actionable insights related to nursing and its interdisciplinary subfields."
        ),
        ("human", "{input}"),
    ]
)

nurse_chat = chat_prompt | llm | StrOutputParser()


tavily_api_key = os.getenv("TAVILY_API_KEY")
    
if not tavily_api_key:
        raise ValueError("API key not found. Please set the TAVILY_API_KEY environment variable.")
    
tavily_tool = TavilySearchResults(
            max_results=4,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=True
    )
         
def prompt_response(user_input):
    """Invoke both nurse chat model and Tavily Search tool, returning a combined response."""
    
    try:
        
        nurse_response = nurse_chat.invoke({"input": user_input})

        
        search_results = tavily_tool.invoke({"query": user_input})

        
        formatted_results = format_search_results(search_results)

        
        combined_output = (
            f"\n{nurse_response}\n\n"
            f"{formatted_results}"
        )

        return combined_output
    
    except Exception as e:
        return f"An error occurred while processing your request: {e}"


def format_search_results(results):
    """Format the search results into a readable string."""
    if not results:
        return "No relevant search results found."

    formatted_output = "### Search Results:\n Click the links to view more details.\n\n"
    for i, result in enumerate(results):
        url = result.get('url', 'No URL available')
        content = result.get('content', 'No content available')
        
        
        if isinstance(url, str) and isinstance(content, str):
            formatted_output += f"{i + 1}. **URL:** {url}\n\n"  
            formatted_output += f"   **Content:** {content}\n\n"  

    
    return formatted_output
