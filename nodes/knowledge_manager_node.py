from datetime import datetime
import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
import chromadb
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_core.messages import SystemMessage, HumanMessage

from graph.graph_init import GraphState
from utils.models import OllamaModel, define_model
from utils.logs import log_graph_state, log_message


INITIAL_PROMPT = """
# CONTEXT #
You are a robot assistant in a care home. You must help a patient to complete his/her activities.

# TASK #
From the conversation history and the execution_notes of each activity, extract key information to use them as new knowledge about the patient.
This new knowledge will be used to better assist the patient in the next interactions.

# OUTPUT #
Be precise and concise. The key information should be a few sentences long. Include the actual datetime in the summary.
"""

NEW_REQUEST_PROMPT = """
- Conversation history:
{conversation_history}

- Task:
{task}

- Datetime:
{datetime}
"""


def add_knowledge_to_vector_store(vector_store: Chroma):
    documents = []
    data_path = "./mock_data"

    for filename in os.listdir(data_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_path, filename)

            loader = TextLoader(file_path)
            doc = loader.load()[0]

            patient_id = os.path.splitext(filename)[0]

            document = Document(
                page_content=doc.page_content, metadata={"patient_id": patient_id}
            )

            documents.append(document)

    vector_store.add_documents(documents=documents)


def knowledge_manager_node(state: GraphState) -> GraphState:
    log_graph_state(state, "knowledge_manager_node")

    embeddings_model = os.getenv("EMBEDDINGS_MODEL") or OllamaModel.LLAMA_3_1.value
    embeddings = OllamaEmbeddings(model=embeddings_model)
    patient_knowledge_collection_name = "patient_knowledge"

    # If the vector store is None, create a new one
    if state["vector_store"] is None:
        persistent_client = chromadb.PersistentClient()
        collection = persistent_client.get_or_create_collection(
            patient_knowledge_collection_name
        )

        vector_store = Chroma(
            client=persistent_client,
            collection_name=patient_knowledge_collection_name,
            embedding_function=embeddings,
        )

        # If the patient_knowledge collection is empty, add knowledge to the vector store
        if collection.count() == 0:
            add_knowledge_to_vector_store(vector_store)

        return {**state, "vector_store": vector_store}

    # If the vector is already created, add new knowledge to it through the conversation history
    vector_store = state["vector_store"]

    # REMOVE: This is a temporary solution to NOT add knowledge to the vector store
    return {**state, "vector_store": vector_store, "end": True}

    formatted_datetime = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    initial_prompt_template = PromptTemplate.from_template(INITIAL_PROMPT)
    initial_prompt = initial_prompt_template.invoke(
        {
            "datetime": formatted_datetime,
            "conversation_history": state["conversation_history"],
            "task": state["task"],
        }
    )

    new_request_prompt_template = PromptTemplate.from_template(NEW_REQUEST_PROMPT)
    new_request_prompt = new_request_prompt_template.invoke(
        {
            "datetime": formatted_datetime,
            "task": state["task"],
            "conversation_history": state["conversation_history"],
        }
    )

    model = define_model()

    messages = []
    messages.append(SystemMessage(content=initial_prompt.to_string()))
    messages.append(HumanMessage(content=new_request_prompt.to_string()))

    result = model.invoke(messages).to_json()

    conversation_summary = result["kwargs"]["content"]
    log_message("\nKNOWLEDGE_MANAGER_NODE - result")
    log_message(conversation_summary)

    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=1000, chunk_overlap=200, add_start_index=True
    # )
    # chunks = text_splitter.split_text(conversation_summary)
    # documents = [Document(page_content=chunk) for chunk in chunks]
    # vector_store.add_documents(documents=documents)

    return {**state, "vector_store": vector_store, "end": True}
