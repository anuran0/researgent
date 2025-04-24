

import os
from dotenv import load_dotenv
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

RESEARCH_AGENT_MODEL = "models/gemini-2.0-flash"
ANSWER_AGENT_MODEL = "models/gemini-2.0-flash"
MAX_RESULTS = 5
SEARCH_DEPTH = 2
MAX_CONCURRENT_REQUESTS = 3
