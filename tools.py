from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime

## Creating a custom Tool // Custom calling tool
def save_to_file(data:str, filename:str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formated_text = f"--- Research Output ---\nTImestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, 'a', encoding='utf-8') as f:
        f.write(formated_text)

    return f"Data successfully saved to {filename}"

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_file,
    description="Saves research data to file"
)

## Creating a DuckDuckGo tool
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name = "search",
    func = search.run,
    description = "Use for finding current information or ambiguous terms"
)

## Creating Wikipedia Tool
# api_wrapper = WikipediaAPIWrapper(top_k_results = 1, doc_content_chars_max =300)
# wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=500))
wiki_tool = Tool(
    name="wikipedia",
    func=wiki.run,
    description="Use for historical facts and well-documented topics"
)