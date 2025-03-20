from dotenv import load_dotenv
from graph.graph_init import graph_init
from utils.logs import create_log_file

load_dotenv(override=True)

create_log_file()

invoked_graph = graph_init()
