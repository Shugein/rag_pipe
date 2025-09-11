from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="test")

resp = client.chat.completions.create(
    model="unsloth/Qwen3-8B-unsloth-bnb-4bit",
    # messages=[
    #     {"role": "system", "content": "Ты движок RAG. Отвечай строго по предоставленному контенту."},
    #     {"role": "user", "content": "Вопрос: ...\nКонтекст:\n<doc>...</doc>"}
    # ],
    # messages=[{"role": "user", "content": "2+2=?"}],
        messages=[
        {"role": "system", "content": "Ты движок RAG. Отвечай строго по предоставленному контенту."},
        {"role": "user", "content": "Вопрос: 2+2=?"}
    ],
    # Для быстрого ответа без рассуждений:
    extra_body={"enable_thinking": False},
    temperature=0.3, top_p=0.9, max_tokens=800
)
print(resp.choices[0].message.content)
print('==========================================')
print(resp)
