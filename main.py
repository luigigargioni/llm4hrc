import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path, override=True)

#if os.getenv("OPENAI_API_KEY") is not None: print("OPENAI_API_KEY loaded successfully")
#if os.getenv("LANGCHAIN_API_KEY") is not None: print("LANGCHAIN_API_KEY loaded successfully")

from graph.graph_init import graph_init
from utils.logs import create_log_file


create_log_file()

invoked_graph = graph_init()
