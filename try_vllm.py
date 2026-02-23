import json

from ddgs import DDGS
from openai import OpenAI

client = OpenAI(base_url="http://rack-gamir-g11.cs.tau.ac.il:8000/v1", api_key="dummy")
MODEL = "Qwen/Qwen3-4B"

tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information on a topic",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The search query"}},
                "required": ["query"],
            },
        },
    }
]


def web_search(query: str, max_results: int = 5) -> str:
    print(f"\n[searching: {query}]")
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    if not results:
        return "No results found."
    formatted = ""
    for i, r in enumerate(results, 1):
        formatted += f"{i}. {r['title']}\n{r['href']}\n{r['body']}\n\n"
    return formatted.strip()


def run(user_message: str):
    print(f"\nUser: {user_message}\n")
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )

        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # No tool call — we have a final answer
        if finish_reason == "stop" or not message.tool_calls:
            print(f"\nAssistant: {message.content}")
            return message.content

        # Handle tool calls
        messages.append(message)
        for tool_call in message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            result = web_search(args["query"])
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})


if __name__ == "__main__":
    import sys

    query = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "What are the latest developments in AI this week?"
    )
    run(query)
