import threading
from duckduckgo_search import DDGS
import sqlite3
import time

class SubconsciousResearcher(threading.Thread):
    def __init__(self, topic, sig):
        super().__init__()
        self.topic = topic
        self.sig = sig

    def run(self):
        print(f"[AURA] Subconscious split: Researching {self.topic}")
        try:
            with DDGS() as ddgs:
                results = [r['body'] for r in ddgs.text(self.topic, max_results=10)]
                combined_knowledge = " ".join(results)
                
            with sqlite3.connect('aura_prime.db') as conn:
                c = conn.cursor()
                # Saving to a special 'Knowledge' table for long-term intelligence
                c.execute("CREATE TABLE IF NOT EXISTS global_knowledge (topic TEXT, data TEXT, stamp DATETIME)")
                c.execute("INSERT INTO global_knowledge VALUES (?, ?, ?)", (self.topic, combined_knowledge, time.ctime()))
                conn.commit()
            print(f"[AURA] Topic {self.topic} absorbed into core.")
        except Exception as e:
            print(f"[AURA] Research Fragmented: {e}")
