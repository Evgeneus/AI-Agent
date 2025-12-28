from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain_openai import ChatOpenAI
import tavily
from tavily import TavilyClient
from langchain_tavily import TavilySearch


class Source(BaseModel):
    """
    Schema for a source used by the agent
    """
    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """
    Schema for the agent response
    """
    answer: str = Field(description="The agent's response")
    source: list[Source] = Field(default_factory=list, description="List of sources used to generate response")


# tavily = TavilyClient()

# @tool
# def search(query: str) -> str:
#     """
#     Tool that seaches over the internet
#     """
#     print(f"Searching for {query}")
#     return tavily.search(query=query)

llm = ChatOpenAI(temperature=0., model="gpt-5")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    result = agent.invoke(
        {
            "messages": HumanMessage(content="What is the weather in Trento today? Use one website and one iteration.")
        }
    )
    print(result)
    


if __name__ == "__main__":
    main()
