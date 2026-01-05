from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI


load_dotenv()


"""
* The current implementation does not produce structured output *
Recommended patterns for structured output 
Two-phase (recommended):
Phase 1: llm.bind_tools(tools) loop to gather observations.
Phase 2: llm.with_structured_output(MyModel) to produce the final, schema-validated result using the messages (including tool outputs).
Single-phase only if no tools:
If the task never needs tools, then with_structured_output(...) directly is great.
"""

@tool
def get_text_length(text: str) -> int:
    """
    Returns the length of a text by charactercs
    """
    text = text.strip("'\n").strip('"')
    return len(text)


def main():
    print("Hello from react-search-agent!")
    tools = [get_text_length]
    tools_by_name = {t.name: t for t in tools}

    llm = ChatOpenAI(temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant. "
                "You may call tools when it helps you answer. "
                "After you have the result, respond with the final answer."
            )
        ),
        HumanMessage(content="What is the text length of 'DOG' in characters?"),
    ]

    curr_iter, max_iters = 0, 5
    while curr_iter < max_iters:
        curr_iter += 1
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        tool_calls = getattr(ai_msg, "tool_calls", None) or []
        if not tool_calls:
            print(ai_msg.content)
            return

        for tc in tool_calls:
            # LangChain represents tool calls as dicts with: {id, name, args}
            tool_name = tc.get("name")
            tool_call_id = tc.get("id")
            tool_args = tc.get("args") or {}

            tool_to_use = tools_by_name.get(tool_name)
            if tool_to_use is None:
                raise ValueError(f"Tool with name {tool_name} not found.")

            observation = tool_to_use.invoke(tool_args)
            print(f"Observation ({tool_name}): {observation}")
            messages.append(
                ToolMessage(content=str(observation), tool_call_id=str(tool_call_id))
            )

    raise RuntimeError("Reached max_iters without producing a final answer.")

    
    

if __name__ == "__main__":
    main()
