from typing import Any

from graph.state import GraphState
from graph.chains.router import question_router

def route_query(state: GraphState) -> dict[str, Any]:
    res = question_router.invoke(
        {"question": state["question"]}
    )

    return {"data_source": res.data_source}
    