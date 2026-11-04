import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rag.core.llm.text_generation import GroqTextGenerator


class TestGroqTextGenerator(unittest.TestCase):
    def test_uses_groq_openai_compatible_api_and_requested_model(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"graph_worthy": true}'))]
        )
        with patch("groq.Groq") as groq_client:
            groq_client.return_value.chat.completions.create.return_value = response
            generator = GroqTextGenerator(
                model="llama-3.1-8b-instant",
                api_key="test-key",
            )

            result = generator.generate(
                "classify this",
                max_tokens=32,
                timeout_seconds=10.0,
            )

        groq_client.assert_called_once_with(
            api_key="test-key",
        )
        groq_client.return_value.chat.completions.create.assert_called_once_with(
            model="llama-3.1-8b-instant",
            temperature=0,
            max_tokens=32,
            timeout=10.0,
            messages=[{"role": "user", "content": "classify this"}],
        )
        self.assertEqual(result, '{"graph_worthy": true}')

    def test_requires_api_key(self):
        with self.assertRaisesRegex(ValueError, "GROQ_API_KEY"):
            GroqTextGenerator(model="llama-3.1-8b-instant", api_key="")


if __name__ == "__main__":
    unittest.main()
