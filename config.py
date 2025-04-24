# config.py

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Agent Configuration
RESEARCH_AGENT_MODEL = "models/gemini-2.0-flash"
ANSWER_AGENT_MODEL = "models/gemini-2.0-flash"

# Search Configuration
MAX_RESULTS = 5
SEARCH_DEPTH = 2
MAX_CONCURRENT_REQUESTS = 3
