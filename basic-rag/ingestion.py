import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()


if __name__ == "__main__":
    print("Ingesting..")
    loader = TextLoader(os.path.join(os.path.dirname(__file__), "mediumblog1.txt"))
    document = loader.load()

    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    
    embedder = OpenAIEmbeddings(
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        model="text-embedding-ada-002"
        )

    print("Ingesting...")
    PineconeVectorStore.from_documents(texts, embedder, index_name=os.environ["INDEX_NAME"])
    print("Finished")