from pydoc import doc
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState

load_dotenv()
web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> dict[str, object]:
    question = state["question"]
    if "documents" in state:
        documents = state["documents"]
    else:
        documents = None

    tavily_results = web_search_tool.invoke(
        {"query": question}
    )["results"]
    joined_results = "\n".join([result["content"] for result in tavily_results])
    web_results = Document(page_content=joined_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    
    return {"documents": documents, "question": question}


if __name__ == "__main__":
    web_search(state={"question": "agent memory", "documents": None})