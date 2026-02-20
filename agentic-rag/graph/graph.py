from dotenv import load_dotenv
from langgraph.graph import START, END, StateGraph

from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH, ROUTER
from graph.nodes import generate, grade_documents, retrieve, web_search, route_query
from graph.state import GraphState
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.answer_grader import answer_grader

load_dotenv()


def route_query_edge(state: GraphState):
    data_source = state["data_source"]
    if data_source == "websearch":
        print("--- ROUTE TO WEBSEARCH ---")
        return "websearch"
    else:
        print("--- ROUTE TO VECTORSTORE ---")
        return "vectorstore"


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    if score.binary_score:
         print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
         print("---GRADE GENERATION vs QUESTION---")
         score = answer_grader.invoke({"question": question, "generation": generation})
         if score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
         else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "hallucination detected"

def decide_to_generate(state):
    if state["web_search"]:
        return WEBSEARCH
    else:
        return GENERATE

workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GENERATE, generate)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(WEBSEARCH, web_search)
workflow.add_node(ROUTER, route_query)
workflow.set_entry_point(ROUTER)

workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    ROUTER,
    route_query_edge,
    {
        "websearch": WEBSEARCH,
        "vectorstore": RETRIEVE
    }
)
workflow.add_conditional_edges(GRADE_DOCUMENTS, decide_to_generate, [WEBSEARCH, GENERATE])
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "hallucination detected": GENERATE,
        "useful": END,
        "not useful": WEBSEARCH
    }
)

workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")