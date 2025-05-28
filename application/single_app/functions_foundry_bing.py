"""
functions_foundry_bing.py
------------------------------------------------------------------
Key points
• Uses azure-ai-projects 1.0.0b11 (same as MS sample).
• Auth:  AzureKeyCredential if AZURE_AI_FOUNDRY_API_KEY is set,
         otherwise DefaultAzureCredential().
• Mirrors the sample code but hides the SDK details behind
  run_search(query, top_n) so existing SimpleChat calls stay unchanged.
"""

from __future__ import annotations
from typing import List, Dict
import os, time, functools

from azure.ai.projects import AIProjectClient
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential

# -----------------------------------------------------------------
# Build one client + cache it per worker
# -----------------------------------------------------------------
@functools.lru_cache
def _client_and_agent():
    conn_str = os.environ["AZURE_AI_FOUNDRY_CONNECTION_STRING"]
    agent_id = os.environ["AZURE_AI_FOUNDRY_BING_AGENT_ID"]
    api_key  = os.getenv("AZURE_AI_FOUNDRY_API_KEY")

    credential = AzureKeyCredential(api_key) if api_key else DefaultAzureCredential()
    client     = AIProjectClient.from_connection_string(
        credential=credential,
        conn_str=conn_str
    )

    agent = client.agents.get_agent(agent_id)

    # Quick guard: make sure Grounding-with-Bing is in Knowledge
    if not any(k.provider == "grounding-bing" for k in (agent.knowledge or [])):
        raise RuntimeError(
            f"Assistant {agent_id} has no 'Grounding with Bing Search' knowledge source."
        )

    return client, agent_id


# -----------------------------------------------------------------
# Helper: synchronous run (poll until completed)
# -----------------------------------------------------------------
def _sync_run(client: AIProjectClient, thread_id: str, agent_id: str, *, timeout=30):
    run = client.agents.create_and_process_run(thread_id=thread_id, agent_id=agent_id)
    t0  = time.time()
    while True:
        run = client.agents.get_run(thread_id=thread_id, run_id=run.id)
        if run.status == "completed":
            return
        if run.status in {"failed", "cancelled"}:
            raise RuntimeError(f"Assistant run {run.status}: {run.error}")
        if time.time() - t0 > timeout:
            raise TimeoutError("Assistant run timed out")
        time.sleep(2)


# -----------------------------------------------------------------
# Public API for SimpleChat
# -----------------------------------------------------------------
def run_search(query: str, top_n: int = 10) -> List[Dict]:
    try:
        client, agent_id = _client_and_agent()

        # 1. create isolated thread
        thread = client.agents.create_thread()

        # 2. user message
        client.agents.create_message(
            thread_id=thread.id,
            role="user",
            content=query
        )

        # 3. run assistant synchronously
        _sync_run(client, thread.id, agent_id)

        # 4. fetch assistant’s reply (last assistant message)
        msgs = client.agents.list_messages(thread_id=thread.id)
        answer = [m for m in msgs.text_messages if m.role == "assistant"][-1].content

    except Exception as ex:
        print(f"[FoundryBing] search error: {ex}")
        return []

    return [{
        "name": query,
        "url":  "",
        "snippet": answer
    }][:top_n]
