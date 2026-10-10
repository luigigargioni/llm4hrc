from datetime import datetime
import os

from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
import chromadb
from colorama import Fore, Style

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

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

            loader = TextLoader(
                file_path,
                encoding="utf-8"
            )

            loaded_docs = loader.load()

            if len(loaded_docs) == 0:
                continue

            patient_id = os.path.splitext(filename)[0]

            raw_text = loaded_docs[0].page_content

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )

            chunks = splitter.split_text(raw_text)

            for chunk in chunks:

                document = Document(
                    page_content=chunk,
                    metadata={
                        "patient_id": patient_id
                    }
                )

                documents.append(document)
    #print(f"KNOWLEDGE_MANAGER_NODE - documents: {documents} from {data_path}")
    if len(documents) > 0:

        vector_store.add_documents(documents)

        log_message(
            f"KNOWLEDGE_MANAGER_NODE - added {len(documents)} chunks to vector store"
        )


def knowledge_manager_node(state: GraphState) -> GraphState:
    print(Fore.GREEN + "[position: KNOWLEDGE MANAGER NODE]" + Style.RESET_ALL)

    log_graph_state(state, "knowledge_manager_node")

    embeddings_model = (
        os.getenv("EMBEDDINGS_MODEL")
        or OllamaModel.LLAMA_3_1.value
    )

    embeddings = OllamaEmbeddings(
        model=embeddings_model
    )

    patient_knowledge_collection_name = "patient_knowledge"

    # ---------------------------------------------------
    # CREATE VECTOR STORE IF NEEDED
    # ---------------------------------------------------

    if state["vector_store"] is None:

        persistent_client = chromadb.PersistentClient()

        vector_store = Chroma(
            client=persistent_client,
            collection_name=patient_knowledge_collection_name,
            embedding_function=embeddings,
        )

        state["vector_store"] = vector_store

    vector_store = state["vector_store"]

    # ---------------------------------------------------
    # GET COLLECTION
    # ---------------------------------------------------

    persistent_client = chromadb.PersistentClient()

    collection = persistent_client.get_or_create_collection(
        patient_knowledge_collection_name
    )

    # ---------------------------------------------------
    # ALWAYS REBUILD VECTOR STORE
    # ---------------------------------------------------

    try:

        existing = collection.get()

        existing_ids = existing["ids"]

        if len(existing_ids) > 0:

            collection.delete(ids=existing_ids)

            log_message(
                "KNOWLEDGE_MANAGER_NODE - cleared old vector store"
            )

    except Exception as e:

        print("Vector store cleanup warning:")
        print(e)

    # ---------------------------------------------------
    # RELOAD KNOWLEDGE FROM FILES
    # ---------------------------------------------------

    add_knowledge_to_vector_store(vector_store)

    log_message(
        "KNOWLEDGE_MANAGER_NODE - vector store rebuilt from mock_data"
    )

    # ---------------------------------------------------
    # GENERATE NEW MEMORY FROM CONVERSATION
    # ---------------------------------------------------

    formatted_datetime = datetime.now().strftime(
        "%d-%m-%Y %H:%M:%S"
    )

    initial_prompt = PromptTemplate.from_template(
        INITIAL_PROMPT
    ).invoke(
        {
            "datetime": formatted_datetime,
            "conversation_history": state["conversation_history"],
            "task": state["task"],
        }
    )

    request_prompt = PromptTemplate.from_template(
        NEW_REQUEST_PROMPT
    ).invoke(
        {
            "datetime": formatted_datetime,
            "task": state["task"],
            "conversation_history": state["conversation_history"],
        }
    )

    model = define_model()

    messages = [
        SystemMessage(content=initial_prompt.to_string()),
        HumanMessage(content=request_prompt.to_string()),
    ]

    result = model.invoke(messages)

    conversation_summary = result.content

    # ---------------------------------------------------
    # NORMALIZE MODEL OUTPUT
    # ---------------------------------------------------

    if not isinstance(conversation_summary, str):
        conversation_summary = str(conversation_summary)

    log_message("\nKNOWLEDGE_MANAGER_NODE - result")

    log_message(conversation_summary)

    # ---------------------------------------------------
    # ADD CONVERSATION SUMMARY TO VECTOR STORE
    # ---------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.split_text(conversation_summary)

    documents = [
        Document(page_content=chunk)
        for chunk in chunks
    ]

    if len(documents) > 0:

        vector_store.add_documents(documents)

    # ---------------------------------------------------
    # RETURN UPDATED STATE
    # ---------------------------------------------------

    return {
        **state,
        "vector_store": vector_store,
    }