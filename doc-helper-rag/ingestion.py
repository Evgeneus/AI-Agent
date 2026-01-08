import asyncio

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


async def main():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", chunk_size=50, retry_min_seconds=10
    )
    vectorstore = PineconeVectorStore(
        index_name="langchain-docs-2026", embedding=embeddings
    )

    # Crawl text from a website
    tavily_crawl = TavilyCrawl()
    res = tavily_crawl.invoke(
        {
            "url": "https://python.langchain.com",
            "max_depth": 5,
            "extract_depth": "advanced",
        }
    )
    all_docs = [
        Document(page_content=result["raw_content"], metadata={"source": result["url"]})
        for result in res["results"]
    ]

    # Split documents into chancks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200,
    )
    splitted_docs = text_splitter.split_documents(all_docs)

    # Populate Vector DB with document embeddings
    # Currently aadd_documents does not support Semaphore, it is possible to implement a wrapper
    await vectorstore.aadd_documents(splitted_docs, batch_size=500)
    print("Pipeline Completed")


if __name__ == "__main__":
    asyncio.run(main())
