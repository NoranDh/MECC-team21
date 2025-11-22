import os

key = os.environ.get("OPENAI_API_KEY")
print("Len:", len(key) if key else None)
print("Starts with:", key[:10] if key else None)
print("Raw repr:", repr(key))