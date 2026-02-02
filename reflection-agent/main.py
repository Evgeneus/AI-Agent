
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph, MessagesState

from chains import generation_chain, reflection_chain

REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: MessagesState):
    response = generation_chain.invoke({"messages": state["messages"]})

    return {"messages": [response]}

def reflection_node(state: MessagesState):
    response = reflection_chain.invoke({"messages": state["messages"]})

    return {"messages": [HumanMessage(content=response.content)]}

def should_continue(state: MessagesState):
    if len(state["messages"]) > 6:
        return END
    else:
        return REFLECT

def build_graph():
    flow = StateGraph(MessagesState)
    flow.add_node(GENERATE, generation_node)
    flow.set_entry_point(GENERATE)
    flow.add_node(REFLECT, reflection_node)
    flow.add_conditional_edges(GENERATE, should_continue, [REFLECT, END])
    flow.add_edge(REFLECT, GENERATE)
    graph = flow.compile()

    graph.get_graph().draw_mermaid_png(output_file_path="flow.png")

    return graph


def main():
    graph = build_graph()
    inputs = {
        "messages": [
            HumanMessage(
                content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """
            )
        ]
    }
    response = graph.invoke(inputs)
    print(response)


if __name__ == "__main__":
    main()
