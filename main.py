import streamlit as st
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import Tool

from tools import save_to_file, search_tool, wiki_tool

# Page configuration
st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔍",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .subheader {
        font-size: 1.2rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .results-area {
        background-color: #f7f7f7;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("<h1 class='main-header'>AI Research Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Let AI help you research any topic!</p>", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Load environment variables
def load_environment():
    # Try to load from .env file
    load_dotenv()
    
    # Check if API keys are available
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    return openai_key, anthropic_key

# Define the Pydantic model for structured output
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    
    # Model selection
    model_option = st.selectbox(
        "Choose LLM Provider",
        ["OpenAI", "Anthropic"]
    )
    
    if model_option == "OpenAI":
        model_name = st.selectbox(
            "Select Model",
            ["gpt-3.5-turbo", "gpt-4"]
        )
    else:
        model_name = st.selectbox(
            "Select Model",
            ["claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"]
        )
    
    # API Key inputs
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    anthropic_api_key = st.text_input("Anthropic API Key", type="password")
    
    # Load API keys from environment if not provided
    if not openai_api_key or not anthropic_api_key:
        env_openai, env_anthropic = load_environment()
        if not openai_api_key and env_openai:
            openai_api_key = env_openai
            st.success("Loaded OpenAI API key from environment")
        if not anthropic_api_key and env_anthropic:
            anthropic_api_key = env_anthropic
            st.success("Loaded Anthropic API key from environment")
    
    # Output file settings
    output_filename = st.text_input("Output Filename", "research_output.txt")
    
    # About section
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This application uses LangChain with various LLMs to help you research any topic.
    It combines information from Wikipedia and web searches to provide comprehensive results.
    """)

# Main research area
st.header("Research Query")
query = st.text_area("What would you like to research?", height=100)

# Process button
if st.button("Start Research", type="primary"):
    if not query:
        st.error("Please enter a research query")
    elif model_option == "OpenAI" and not openai_api_key:
        st.error("Please provide an OpenAI API key")
    elif model_option == "Anthropic" and not anthropic_api_key:
        st.error("Please provide an Anthropic API key")
    else:
        # Show progress
        with st.status("Researching your topic...", expanded=True) as status:
            try:
                # Initialize the LLM
                if model_option == "OpenAI":
                    llm = ChatOpenAI(model=model_name, api_key=openai_api_key)
                    st.write("🤖 Using OpenAI model:", model_name)
                else:
                    llm = ChatAnthropic(model=model_name, api_key=anthropic_api_key)
                    st.write("🤖 Using Anthropic model:", model_name)
                
                # Setup parser and prompt
                parser = PydanticOutputParser(pydantic_object=ResearchResponse)
                
                prompt = ChatPromptTemplate.from_messages([
                    (
                        "system",
                        """
                        You are a research assistant that will help generate a research paper.
                        For every query, use both the Wikipedia and search tools to gather information, then save the combined summary to a file.
                        
                        Wrap the output in this format and provide no other text
                        {format_instructions}
                        """,
                    ),
                    ("placeholder", "{chat_history}"),
                    ("human", "{query}"),
                    ("placeholder", "{agent_scratchpad}"),
                ]).partial(format_instructions=parser.get_format_instructions())
                
                # Create agent and executor
                st.write("🔧 Setting up research tools...")
                
                # Update save_tool to use custom filename
                custom_save_tool = Tool(
                    name="save_text_to_file",
                    func=lambda x: save_to_file(x, output_filename),
                    description="Saves research data to file"
                )
                
                tools = [search_tool, wiki_tool, custom_save_tool]
                
                agent = create_tool_calling_agent(
                    llm=llm,
                    prompt=prompt,
                    tools=tools
                )
                
                agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
                
                # Run the query
                st.write("🔍 Researching...")
                raw_response = agent_executor.invoke({"query": query})
                
                # Parse the response
                st.write("📊 Processing results...")
                structured_output = parser.parse(raw_response.get("output"))
                
                # Update status
                status.update(label="Research completed!", state="complete")
                
                # Add to history
                st.session_state.history.append({
                    "query": query,
                    "response": structured_output
                })
                
                # Display results
                st.markdown("## Research Results")
                results_container = st.container(border=True)
                with results_container:
                    st.markdown(f"### {structured_output.topic}")
                    
                    st.markdown("#### Summary")
                    st.write(structured_output.summary)
                    
                    st.markdown("#### Sources")
                    for source in structured_output.sources:
                        st.markdown(f"- {source}")
                    
                    st.markdown("#### Tools Used")
                    st.write(", ".join(structured_output.tools_used))
                    
                    st.success(f"Research saved to {output_filename}")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.code(f"Raw response: {raw_response}" if 'raw_response' in locals() else "No response received")
                status.update(label="Research failed", state="error")

# History section
if st.session_state.history:
    st.markdown("---")
    st.header("Previous Research")
    
    for i, item in enumerate(reversed(st.session_state.history)):
        with st.expander(f"Query: {item['query'][:50]}..."):
            st.markdown(f"### {item['response'].topic}")
            st.markdown("#### Summary")
            st.write(item['response'].summary)
            st.markdown("#### Sources")
            for source in item['response'].sources:
                st.markdown(f"- {source}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using LangChain, Streamlit, and AI")