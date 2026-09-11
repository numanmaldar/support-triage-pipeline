from llm_client import call_llm

if __name__ == "__main__":
    print("Sending a tiny test prompt to Gemini...")
    result = call_llm("Reply with exactly one word: pong")
    print(f"Response: {result!r}")