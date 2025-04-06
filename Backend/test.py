import httpx

headers = {
    "Authorization": "Bearer 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",  # replace if key is updated
    "Content-Type": "application/json"
}

payload = {
    "model": "mixtral-8x7b-32768",
    "messages": [{"role": "user", "content": "Hi there!"}],
    "max_tokens": 50
}

response = httpx.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)

print("Status Code:", response.status_code)
print("Response:", response.text)
