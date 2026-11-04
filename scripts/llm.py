import os

from rag.retrieval.LLM.text_generation import GeminiTextGenerator 


api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Set the GEMINI_API_KEY environment variable before running this script.")

gemini_generator = GeminiTextGenerator(
    model="gemini-3.5-flash",
    api_key=api_key,
    timeout_seconds=25.0,
)

response = gemini_generator.generate(
    prompt="Write a short story about a robot learning to dance.",
    max_tokens=100,
    timeout_seconds=25.0,
)   

print(response)