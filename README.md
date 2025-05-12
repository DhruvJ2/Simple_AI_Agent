# 🔍 AI Research Assistant

A Streamlit-based web application that leverages powerful LLMs via LangChain, combined with Wikipedia and web search tools, to assist users in researching any topic with ease.


## 🚀 Features

- Supports **OpenAI** (`gpt-3.5-turbo`, `gpt-4`) and **Anthropic** (`claude-3` family) models  
- Gathers data from **Wikipedia** and **DuckDuckGo Search**  
- Saves structured research summaries to a local file  
- Interactive **Streamlit** UI with a history panel  
- Customizable output filename and model selection  


## 🛠️ Requirements

- Python 3.10+
- API keys for:
  - [OpenAI](https://platform.openai.com/account/api-keys)
  - [Anthropic](https://console.anthropic.com/settings/keys)


## 📦 Installation

```bash
git clone https://github.com/yourusername/ai-research-assistant.git
cd ai-research-assistant
pip install -r requirements.txt
```

## 🔐 Environment Setup

Create a .env file in the root directory and add your API keys:
```bash
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```
Alternatively, input the keys via the app sidebar if not set in the environment.

## ▶️ Running the App
Launch the Streamlit application:
```bash
streamlit run main.py
```

## 🧩 Tools Used
1. wikipedia: Queries structured data from Wikipedia
2. search: Uses DuckDuckGo to find current or general web info
3. save_text_to_file: Saves research output with timestamp to a local .txt file

## 📚 Resources
- [LangChain Documentation](https://python.langchain.com/docs/introduction/)

- [Streamlit Documentation](https://docs.streamlit.io/)

- [OpenAI Platform](https://platform.openai.com/docs/overview)

- [Anthropic API](https://docs.anthropic.com/en/docs/welcome)
