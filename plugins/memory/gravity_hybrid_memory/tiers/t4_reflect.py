import logging
import os
import json
from agent.auxiliary_client import get_text_auxiliary_client
from .t1_core import Tier1Core

logger = logging.getLogger(__name__)

class Tier4Reflection:
    def __init__(self, t1: Tier1Core):
        self.t1 = t1

    def reflect_and_learn(self, conversation_chunk: str):
        """Analyzes a conversation chunk to extract permanent T1 facts."""
        client, model = get_text_auxiliary_client("fact_extractor")
        
        # Matt Pocock Style: Robust fallback for auxiliary client
        if not client or not model:
            from agent.openai_adapter import OpenAIAdapter
            if os.environ.get("GOOGLE_API_KEY"):
                client = OpenAIAdapter(
                    api_key=os.environ["GOOGLE_API_KEY"],
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai"
                )
                model = "google/gemini-2.0-flash-exp" # Fast and smart for extraction
            else:
                logger.warning("No auxiliary client or Google key available for reflection.")
                return

        prompt = (
            "You are the Gravity Reflection Engine. Extract NEW permanent user facts from this conversation.\n"
            "Facts should be concise and permanent (preferences, location, identity).\n"
            "Format: JSON list [{\"key\": \"...\", \"value\": \"...\", \"category\": \"...\"}]\n"
            f"CONVERSATION:\n{conversation_chunk}"
        )

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            raw = resp.choices[0].message.content
            # Basic JSON extraction
            start = raw.find('[')
            end = raw.rfind(']') + 1
            if start != -1 and end != -1:
                facts = json.loads(raw[start:end])
                for f in facts:
                    self.t1.learn_fact(f['key'], f['value'], f.get('category', 'general'))
                    logger.info(f"Gravity Reflected and Learned: {f['key']}")
        except Exception as e:
            logger.error(f"Reflection error: {e}")
