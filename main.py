from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
import os
from tools import search_tool, wiki_tool, save_tool

## Insering apikey through .env file 
load_dotenv("D:\Programming_workspaces\Python\AI_Agent\.env")
apik = os.getenv("OPENAI_API_KEY")


## Creating State of the LangChain
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

## Initializing LLM Model
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=apik)
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("What can I help you research? ")
raw_response = agent_executor.invoke({"query": query})
# print(raw_response)

try:
    structured_ouput = parser.parse(raw_response.get("output")[0]["text"])
    print(structured_ouput)
except Exception as e:
    print("Error Parsing response", e, "Raw Response", raw_response)

## Invoking LLM Model
# response = llm.invoke("what is gaurd?")
# print(response)
