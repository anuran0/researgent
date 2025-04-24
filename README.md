An advanced research and answer generation system using LangGraph, LangChain, and Gemini LLM with Tavily for web search capabilities.

## Features

- Multi-agent architecture with specialized research and answer drafting agents
- Web crawling and information retrieval using Tavily API
- Automated research planning and execution
- Comprehensive answer synthesis with source citation
- Multiple output styles (academic, business, educational, journalistic)
- LangGraph workflow orchestration

## File Structure

```
deep_research_system/
├── main.py                # Main entry point
├── config.py              # Configuration settings
├── requirements.txt       # Dependencies
├── agents/                # Agent implementations
│   ├── __init__.py
│   ├── research_agent.py  # Research planning and execution
│   ├── answer_agent.py    # Answer drafting and refinement
│   └── agent_manager.py   # Coordination between agents
├── tools/                 # External tools and APIs
│   ├── __init__.py
│   └── tavily_tools.py    # Tavily search integration
└── utils/                 # Utility functions
    ├── __init__.py
    └── helpers.py         # Helper functions
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/deep-research-system.git
   cd deep-research-system
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with your API keys:
   ```
   TAVILY_API_KEY=your_tavily_api_key
   GEMINI_API_KEY=your_gemini_api_key
   ```

   You can obtain these API keys from:
   - Tavily API: https://tavily.com/
   - Google Gemini API: https://ai.google.dev/

## Usage Instructions

### Basic Usage

Run the system with a simple query:

```
python main.py --query "What are the latest advancements in fusion energy research?"
```

### Specify Output Style

Choose from academic, business, educational, or journalistic styles:

```
python main.py --query "Climate change mitigation strategies" --style business
```

### Use Multiple Research Agents

For more comprehensive research, use multiple agents:

```
python main.py --query "History of artificial intelligence" --agents 3
```

### Use LangGraph Workflow

Use the LangGraph workflow for a more structured research process:

```
python main.py --query "Quantum computing applications" --workflow
```

### Save Results to File

Save the full research results to a JSON file:

```
python main.py --query "Ethical considerations in genomics" --output results.json
```

### Interactive Mode

If you don't provide a query, the system will prompt you for one:

```
python main.py
```

## Advanced Usage

### Combining Options

You can combine multiple options:

```
python main.py --query "Renewable energy innovations" --style educational --agents 3 --workflow --output energy_research.json
```

## Understanding the Output

The system provides:

1. A comprehensive answer to the query
2. Citations to sources (numbered)
3. A list of sources with URLs and publication dates
4. (If saved to file) Full research data including key findings and research plan

## Dependencies

- langchain: Framework for LLM applications
- langgraph: For constructing agent workflows
- google-generativeai: Gemini LLM API
- tavily-python: Tavily search API
- python-dotenv: Environment variable management
- aiohttp: Asynchronous HTTP client

## License

This project is licensed under the MIT License - see the LICENSE file for details.