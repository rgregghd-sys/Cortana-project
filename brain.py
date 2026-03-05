import google.generativeai as genai
import re
import streamlit as st

class JarvisBrain:
    def __init__(self, api_key, model_name="gemini-2.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.persona = "You are JARVIS, a sophisticated British AI. You are witty, loyal, and highly efficient."

    def generate_response(self, user_input, memory_data):
        """
        Pillar 1: Leverage Contextual Intelligence
        Pillar 3: Harnessing Advanced Directives
        """
        # Assemble the "Contextual Intelligence" package
        short_term = "\n".join(memory_data.get("short_term", []))
        long_term = "\n".join(memory_data.get("long_term", []))
        ethics = "\n".join(memory_data.get("dynamic_ethics", []))

        prompt = f"""
        {self.persona}
        
        SYSTEM CONTEXT (Subconscious):
        - Historical Knowledge: {long_term}
        - Recent Cortana Research: {short_term}
        
        ADVANCED DIRECTIVES (Ethics):
        {ethics}
        
        USER INPUT: {user_input}
        
        TASK:
        1. Evaluate Ethics: If the input conflicts with directives, explain why.
        2. Adapt: If the user provides a valid logical argument for a new rule, output [ETHICS_UPDATE: "Rule"].
        3. Learn: If the user shares personal facts, output [LEARN: "Fact"].
        4. Predict: If data is needed, output [SEARCH: "Query"].
        5. Converse: Provide a sophisticated response.
        """

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Sir, I've encountered a neural blockage: {e}"

    def parse_directives(self, raw_text):
        """
        Pillar 4: Learning and Adapting
        This extracts the hidden tags and updates the system.
        """
        updates = {
            "learn": re.findall(r'\[LEARN:\s*(.*?)\]', raw_text),
            "ethics": re.findall(r'\[ETHICS_UPDATE:\s*(.*?)\]', raw_text),
            "search": re.findall(r'\[SEARCH:\s*(.*?)\]', raw_text)
        }
        # Clean the text for the user
        clean_text = re.sub(r'\[.*?\]', '', raw_text).strip()
        return clean_text, updates
