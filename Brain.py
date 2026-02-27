import time
from langgraph.graph import StateGraph, END
from memory import save_to_memory

def progressive_logic(state: dict):
    # Path 1: Research, Path 2: Verification
    print(f"--- 1-to-2 Logic Split for: {state.get('question')} ---")
    return {"question": state.get("question"), "paths": ["Deep_Research", "Cross_Verify"]}

def autonomous_crawl_and_digest(state: dict):
    subject = state.get("question", "General AI Trends")
    print(f"--- Autonomous Crawl Started for {subject} ---")
    
    # In a full setup, this triggers Crawl4AI. Here we simulate the result:
    absorbed_info = [
        f"Neural Link optimization discovered for {subject}.",
        f"Verified: 2026 Cloud latency reduced by 15%."
    ]
    save_to_memory(absorbed_info)
    return {"data_absorbed": absorbed_info}

workflow = StateGraph(dict)
workflow.add_node("logic_split", progressive_logic)
workflow.add_node("crawler_digest", autonomous_crawl_and_digest)
workflow.set_entry_point("logic_split")
workflow.add_edge("logic_split", "crawler_digest")
workflow.add_edge("crawler_digest", END)
cortana_system = workflow.compile()

if __name__ == "__main__":
    print("Cortana Cloud Brain Active.")
    while True:
        cortana_system.invoke({"question": "Cloud Intelligence Nodes"})
        time.sleep(21600) # Sleep 6 hours
