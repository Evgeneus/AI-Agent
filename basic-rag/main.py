import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pydantic import BaseModel, Field

load_dotenv()

class RagAnswer(BaseModel):
    answer: str = Field(description="Final answer")

def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)
    

def main():
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    llm = ChatOpenAI(model="gpt-5.2", temperature=0).with_structured_output(RagAnswer)

    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt_template = ChatPromptTemplate.from_template(
        """Answer the question based only on the following context:

        {context}

        Question: {question}

        Provide a detailed answer:"""
    )
    
    # Define Langchain Retrieval Chain
    retrieval_chain = (
        # RunnablePassthrough means: “pass the input through unchanged”.
        # .assign(context=...) means: “while passing it through, compute a new key called context and add it to the dict.”
        RunnablePassthrough.assign(
            context=(lambda x: x["question"])
            | retriever
            | format_docs
        )
        | prompt_template
        | llm
    )
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL - Better Approach")
    print("=" * 70)
    print("Why LCEL is better:")
    print("- More concise and declarative")
    print("- Built-in streaming: chain.stream()")
    print("- Built-in async: chain.ainvoke()")
    print("- Easy to compose with other chains")
    print("- Better for production use")
    print("=" * 70)

    query = "what is Pinecone in machine learning?"
    result = retrieval_chain.invoke({"question": query})
    print("\nAnswer:")
    print(result)


if __name__ == "__main__":
    main()
