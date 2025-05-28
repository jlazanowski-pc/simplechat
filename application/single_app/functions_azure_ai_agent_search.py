# azure_ai_agent_search.py

import os
import time
from dotenv import load_dotenv
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import MessageRole
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError

load_dotenv()

# --- Configuration ---
PROJECT_CONNECTION_STRING = os.getenv("AZURE_AI_FOUNDRY_PROJECT_ENDPOINT")
AGENT_ID = os.getenv("AZURE_AI_FOUNDRY_AGENT_ID") # The ID like "asst_jbsIiKRK5MmHeJ0hwjOQWtpg"

if not all([PROJECT_CONNECTION_STRING, AGENT_ID]):
    raise ValueError("Please set AZURE_AI_FOUNDRY_PROJECT_ENDPOINT and AZURE_AI_FOUNDRY_AGENT_ID environment variables.")

# --- Helper Function to Get Last Assistant Message ---
def get_last_assistant_message(messages):
    """
    Extract the assistant's text response from Azure AI messages.
    Handles both SDK objects and raw dict responses.
    Returns the text string or None.
    """
    print("--- Inside get_last_assistant_message ---")
    
    # Handle raw dict response (what you're seeing in logs)
    if isinstance(messages, dict):
        print("Processing raw dict response")
        data = messages.get("data", [])
        
        # Look for assistant messages (newest first)
        for msg in data:
            role = str(msg.get("role", "")).lower()
            print(f"Processing message: {msg.get('id')}, Role: '{role}'")
            
            if role == "assistant":
                content_items = msg.get("content", [])
                print(f"Found assistant message with {len(content_items)} content items")
                
                for content_item in content_items:
                    if content_item.get("type") == "text":
                        text_obj = content_item.get("text", {})
                        value = text_obj.get("value")
                        
                        if value:
                            print(f"Found text value: {value[:100]}...")
                            return value
                        else:
                            print(f"Text object found but no value: {text_obj}")
                
                print("Assistant message found but no usable text content")
                return None
        
        print("No assistant message found in data")
        return None
    
    # Handle SDK object response
    if hasattr(messages, "data"):
        print("Processing SDK object with .data attribute")
        for msg in messages.data:
            role = str(getattr(msg, "role", "")).lower()
            print(f"Processing message: {getattr(msg, 'id', 'unknown')}, Role: '{role}'")
            
            if role == "assistant":
                content_items = getattr(msg, "content", [])
                print(f"Found assistant message with {len(content_items)} content items")
                
                for content_item in content_items:
                    if getattr(content_item, "type", "") == "text":
                        text_obj = getattr(content_item, "text", None)
                        if text_obj and hasattr(text_obj, "value"):
                            value = getattr(text_obj, "value")
                            if value:
                                print(f"Found text value: {value[:100]}...")
                                return value
                
                print("Assistant message found but no usable text content")
                return None
        
        print("No assistant message found in SDK data")
        return None
    
    # Handle SDK object with text_messages attribute
    if hasattr(messages, "text_messages"):
        print("Processing SDK object with .text_messages attribute")
        for msg in messages.text_messages:
            role = str(getattr(msg, "role", "")).lower()
            
            if role == "assistant":
                content_items = getattr(msg, "content", [])
                for content_item in content_items:
                    if getattr(content_item, "type", "") == "text":
                        text_obj = getattr(content_item, "text", None)
                        if text_obj and hasattr(text_obj, "value"):
                            return getattr(text_obj, "value")
        
        return None
    
    print("Unknown message format")
    return None

# --- Main Search Function ---
def run_search(query: str) -> str:
    """
    Calls the Azure AI Foundry Agent using the SDK to perform a grounded search.

    Args:
        query: The search query.

    Returns:
        The agent's response as a string, or an error message.
    """
    try:
        print("Authenticating and creating AIProjectClient...")
        # Use DefaultAzureCredential - ensure you are logged in (az login)
        # or have appropriate environment variables set for authentication.
        project_client = AIProjectClient.from_connection_string(
            credential=DefaultAzureCredential(),
            conn_str=PROJECT_CONNECTION_STRING,
        )
        print("AIProjectClient created.")

        # Optional: Verify agent exists (can be removed in production)
        try:
            agent = project_client.agents.get_agent(AGENT_ID)
            print(f"Successfully retrieved agent: {agent.name} ({agent.id})")
        except HttpResponseError as e:
            print(f"Error retrieving agent '{AGENT_ID}': {e}")
            return f"Error: Could not retrieve agent. Check AGENT_ID. Details: {e}"

        print("Creating a new thread...")
        thread = project_client.agents.create_thread()
        print(f"Thread created: {thread.id}")

        print(f"Adding user message: '{query}'...")
        project_client.agents.create_message(
            thread_id=thread.id,
            role=MessageRole.USER,
            content=query
        )
        print("Message added.")

        print("Creating and processing run...")
        run = project_client.agents.create_and_process_run(
            thread_id=thread.id,
            agent_id=AGENT_ID
        )
        print(f"Run started with ID: {run.id}, Status: {run.status}")

        # --- Poll for completion ---
        start_time = time.time()
        while run.status in ["queued", "in_progress", "requires_action"]:
            if time.time() - start_time > 120: # 2-minute timeout
                return "Error: Search timed out."

            time.sleep(3) # Wait before polling again
            run = project_client.agents.get_run(thread_id=thread.id, run_id=run.id)
            print(f"Polling Run Status: {run.status}")

            # Handle 'requires_action' if you add tools that need client-side execution.
            # For Bing Search (grounding), it should usually run server-side.
            if run.status == "requires_action":
                 print("Warning: Run requires action - this is unexpected for Bing Grounding alone.")
                 # If you had functions, you'd handle them here and submit tool outputs.
                 # For now, we'll assume it will eventually complete or fail.
                 pass

        if run.status == "completed":
            print("Run completed. Fetching messages...")
            messages = project_client.agents.list_messages(thread_id=thread.id)
            
            # Debug: Print the raw messages structure
            print(f"Messages type: {type(messages)}")
            if hasattr(messages, 'model_dump'):
                try:
                    messages_dict = messages.model_dump()
                    print(f"Messages as dict: {messages_dict}")
                except:
                    print("Could not convert messages to dict")
            
            assistant_response = get_last_assistant_message(messages)
            print(f"Assistant Response: {assistant_response}")
            
            if assistant_response:
                return assistant_response
            else:
                return "Error: Run completed, but no assistant text response found."
        else:
            error_details = run.last_error.message if run.last_error else "Unknown reason."
            return f"Error: Run ended with status: {run.status}. Details: {error_details}"

    except ClientAuthenticationError as e:
        return (f"Error: Authentication failed. Details: {e}. "
                "Ensure you are logged in via 'az login' or have the correct Azure environment variables set up.")
    except HttpResponseError as e:
        print(f"API Request Error: {e}")
        return f"Error: An API error occurred during agent interaction. {e}"
    except Exception as e:
        import traceback
        print(f"--- CAUGHT UNEXPECTED EXCEPTION ---")
        traceback.print_exc()
        print(f"--- END OF TRACEBACK ---")
        return f"An unexpected error occurred: {e}"

# --- Example Usage (Optional) ---
if __name__ == "__main__":
    # Make sure to set AZURE_AI_FOUNDRY_PROJECT_ENDPOINT and AZURE_AI_FOUNDRY_AGENT_ID
    # and run 'az login' first.
    search_query = "What is the latest news about Azure AI?"
    result = run_search(search_query)
    print(f"\nQuery: {search_query}")
    print(f"Result: {result}")