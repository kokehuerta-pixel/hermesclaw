import os
import json
import time
import unittest
from unittest.mock import MagicMock, patch
import sys
sys.path.insert(0, os.getcwd())
from plugins.memory.gravity import GravityMemoryProvider

class TestGravityEfficiency(unittest.TestCase):
    def setUp(self):
        # Mock dependencies
        self.provider = GravityMemoryProvider()
        self.provider._initialized = True
        self.provider._supabase = MagicMock()
        self.provider._pc_index = MagicMock()
        self.provider._session_id = "test-session"
        self.provider._user_id = "test-user"

    @patch("plugins.memory.gravity.get_text_auxiliary_client")
    def test_noise_filtering(self, mock_get_client):
        # 1. Test short message (should skip extraction)
        self.provider.sync_turn("Hola", "Hola, ¿en qué puedo ayudarte?")
        
        # Wait a bit for the thread (though we'll check call_count)
        time.sleep(0.5)
        self.assertEqual(mock_get_client.call_count, 0, "Should skip extraction for short message")

    @patch("plugins.memory.gravity.get_text_auxiliary_client")
    def test_intent_extraction(self, mock_get_client):
        # Mock client response
        mock_client = MagicMock()
        mock_model = "test-model"
        mock_get_client.return_value = (mock_client, mock_model)
        
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = '[]'
        mock_client.chat.completions.create.return_value = mock_resp

        # 2. Test significant message with intent (should trigger extraction)
        self.provider.sync_turn(
            "Mi lenguaje de programación favorito es Python y estoy trabajando en un bot.", 
            "Entendido, guardaré esa información."
        )
        
        time.sleep(1) # Wait for thread
        self.assertGreaterEqual(mock_get_client.call_count, 1, "Should trigger extraction for message with intent")

    @patch("plugins.memory.gravity.get_text_auxiliary_client")
    def test_consolidation_efficiency(self, mock_get_client):
        # Mock client and response
        mock_client = MagicMock()
        mock_model = "test-model"
        mock_get_client.return_value = (mock_client, mock_model)
        
        mock_resp = MagicMock()
        # Single response with a fact
        mock_resp.choices[0].message.content = '[{"fact": "Usa Python", "category": "technical_stack", "action": "add"}]'
        mock_client.chat.completions.create.return_value = mock_resp

        # Mock prefetch to return empty context
        with patch.object(self.provider, 'prefetch', return_value=""):
            self.provider.sync_turn(
                "Uso Python para mis scripts de automatización.", 
                "Lo tendré en cuenta."
            )
            
            time.sleep(1)
            
            # Should only have called the auxiliary client ONCE for extraction + reconciliation
            # In the old version, it would be 1 for extraction + 1 for reconciliation per fact.
            self.assertEqual(mock_client.chat.completions.create.call_count, 1, 
                             "Consolidation should result in exactly 1 LLM call per turn")

if __name__ == "__main__":
    unittest.main()
