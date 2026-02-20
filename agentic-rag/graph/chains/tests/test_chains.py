from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document
from graph.state import GraphState
from graph.chains.retrieval_grader import retrieval_grader, GradeDocuments
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.generation import generation_chain
from graph.chains.router import question_router
from graph.nodes.grade_documents import grade_documents
from ingestion import retriever


def test_retriever_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "yes"


def test_retriever_grader_anser_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "pizza", "document": doc_txt}
    )

    assert res.binary_score == "no"

def test_grade_documents_node_web_search_false() -> None:
    state = GraphState(
        question="tell me about pizza",
        generation="",
        web_search=False,
        documents=[Document(
            page_content=
            """La pizza Margherita è l'iconica pizza napoletana condita con pomodoro,
             mozzarella (fior di latte o bufala), basilico fresco, sale e olio 
             extravergine di oliva, creata nel 1889 per la Regina Margherita di Savoia 
             con i colori della bandiera italiana."""
             )
             ]
    )
    res = grade_documents(state)
    assert res["web_search"] == False


def test_grade_documents_node_web_search_true() -> None:
    state = GraphState(
        question="tell me about pizza",
        generation="",
        web_search=False,
        documents=[Document(
            page_content=
            """Define your graph state as a TypedDict (or Pydantic model/dataclass) 
            that represents the shared data structure passed between nodes."""
             )
             ]
    )
    res = grade_documents(state)
    assert res["web_search"] == True

def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke(
        {
            "context": docs,
            "question": question
        }
        )
    print(generation)


def test_hallucination_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    res = hallucination_grader.invoke(
        {
            "documents": docs,
            "generation": "In order to make pizza we need to first start with the dough"
        }
    )
    assert not res.binary_score


def test_router_verctorstore() -> None:
    question = "agent mode"
    res = question_router.invoke(
        {"question": question}
    )
    assert res.data_source == "vectorstore"


def test_router_websearch() -> None:
    question = "pizza margarita"
    res = question_router.invoke(
        {"question": question}
    )
    assert res.data_source == "websearch"
