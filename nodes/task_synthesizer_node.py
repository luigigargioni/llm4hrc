#struttura il task da far eseguire al robot sulla base delle informazioni presenti nei documenti e nelle specifiche del dottore:
#i dati testuali vengono trasformati in una struttura

import os
from unittest import result
from langchain_core.prompts import PromptTemplate
from datetime import datetime, timedelta
from langchain_core.tools import tool
from colorama import Fore, Back, Style

from graph.graph_init import GraphState
from deterministic.activity_planner import set_skipped_activities
from utils.interfaces import PATIENTS_ID_LIST, Task
from utils.models import define_model
from utils.logs import conversation_header, log_graph_state, log_message


INITIAL_PROMPT = """
# CONTEXT #
You are a robot assistant in a care home. You must help the patient to do his/her activities.

# TASK #
Create a JSON structure for the prescribed activities of the day based on this information:
{docs_content}.

# SPECIFICATIONS #
The task structure must have the following fields:
- patient_name (string): the name of the patient.
- patient_surname (string): the surname of the patient.
- patient_information (string): the information of the patient.
- datetime (string): the datetime of the task creation.
- activities (list): a list of activities to be prescribed. Each activity should have the following fields:
    - activity_name (string): the name of the activity.
    - activity_time (string | null): when do the activity.
    - without_time (boolean): if the activity does not have a specific time.
    - activity_duration (int): the duration of the activity in minutes.
    - activity_details (string): the details on how execute the activity.
    - execution_notes (string): the notes about the execution of the activity.
    - done (boolean): if the activity has been done or not.
    - skipped (boolean): if the activity has been skipped or not.
    - critical (boolean): if the activity is critical or not.
    - priority (int): the priority of the activity. The higher priority is 1.

# ADDITIONAL INFORMATION #
The optional additional information from the doctor are:
{doctor_specs}.

# TASK #
You must consider any additional information from the doctor to define the activities to be prescribed, the activity time and the activity details.
The information from the doctor has the priority over the information provided in the documents. If the doctor says to skip or avoid an activity, remove it from the list of the activities of the day.
The priority of the activities is from 1 to 3, where 1 is the highest priority and 3 is the lowest priority.

# DATETIME #
The actual datetime is {datetime}.

# OUTPUT FORMAT #
Return ONLY a valid JSON. Do not include any explanation, text, or formatting outside the JSON.
"""


@tool
def response_template(task: Task) -> bool:
    """The JSON format for the output task.

    Args:
        The task structure should have the following fields:
            - patient_name (string): the name of the patient.
            - patient_information (string): the information of the patient.
            - datetime (string): the datetime of the task creation.
            - activities (list): a list of activities to be prescribed. Each activity should have the following fields:
                - activity_name (string): the name of the activity.
                - activity_time (string): when do the activity.
                - without_time (boolean): if the activity does not have a specific time.
                - activity_duration (int): the duration of the activity in minutes.
                - activity_details (string): the details on how execute the activity.
                - execution_notes (string): the notes about the execution of the activity.
                - done (boolean): if the activity has been done or not.
                - skipped (boolean): if the activity has been skipped or not.
                - critical (boolean): if the activity is critical or not.
                - priority (int): the priority of the activity. The higher priority is 1.
    """
    return True


def task_synthesizer_node(state: GraphState) -> GraphState:
    print(Fore.GREEN + "[position: TASK SYNTHESIZER NODE]" + Style.RESET_ALL)
    
    log_graph_state(state, "task_synthesizer_node")

    if state["vector_store"] is None:
        raise ValueError("TASK_SYNTHESIZER_NODE - vector_store is None")

    #prende info da utente

    patient_id = None
    while not patient_id or patient_id not in PATIENTS_ID_LIST:
        patient_id = input(
            Fore.YELLOW + "\nPlease provide the ID of the patient: " + Style.RESET_ALL
        ).strip()
        if not patient_id:
            print(
                Fore.YELLOW
                + "Input cannot be empty. Please provide a valid patient ID."
                + Style.RESET_ALL
            )
        elif patient_id not in PATIENTS_ID_LIST:
            print(
                Fore.YELLOW
                + "Invalid patient ID. Please provide an ID from the list: "
                + ", ".join(PATIENTS_ID_LIST)
                + Style.RESET_ALL
            )

    #embedda la frase "create a new task..." con Chroma, questo fa sì che  si recuperi il documento relativo a quel paziente,
    # con filtro patient_id.
    retrieved_vectors = state["vector_store"].similarity_search(
        "Create a new task based on the provided information for the patient with Record Number: "
        + patient_id,
        k=10,
        filter={"patient_id": patient_id},
    )

    docs_content = "\n\n".join(vector.page_content for vector in retrieved_vectors)

    # definisce il template del prompt invocando from_template con INITIAL_PROMPT, che è il prompt che spiega al modello cosa
    #  deve fare, e con i placeholder per le informazioni che arriveranno dopo (doctor_specs, datetime, docs_content)

    prompt_template = PromptTemplate.from_template(INITIAL_PROMPT)

    #prendo da utente le specifiche del dottore e l'ora

    doctor_specs = input(
        Fore.YELLOW
        + "\nPlease provide the additional information from the doctor: "
        + Style.RESET_ALL
    ).strip()
    if doctor_specs == "":
        doctor_specs = None

    fake_datetime = None

    while fake_datetime is None:
        user_input = input(
            Fore.YELLOW + "\nEnter the datetime (HH:MM): " + Style.RESET_ALL
        ).strip()
        if not user_input:
            print(
                Fore.YELLOW
                + "Input cannot be empty. Please enter a valid time in HH:MM format."
                + Style.RESET_ALL
            )
            continue

        try:
            fake_datetime = datetime.strptime(
                datetime.now().strftime("%d-%m-%Y") + " " + user_input + ":00",
                "%d-%m-%Y %H:%M:%S",
            )
        except ValueError:
            print(
                Fore.YELLOW
                + "Invalid format. Please enter the time in HH:MM format."
                + Style.RESET_ALL
            )
            fake_datetime = None

    formatted_datetime = fake_datetime.strftime("%d-%m-%Y %H:%M:%S")

    log_formatted_datetime = fake_datetime.strftime("%H:%M")
    conversation_header(
        f"{log_formatted_datetime} - Patient ID: {patient_id}\nDoctor specifications: {doctor_specs or ''}\n\n"
    )

    #semplicemente riempi le placeholders, il template del prompt strutturato si popola con le informazioni vere
    prompt = prompt_template.invoke(
        {
            "datetime": formatted_datetime,
            "docs_content": docs_content,
            "doctor_specs": doctor_specs,
        }
    )

    #definisce il modello come specificato in models.py
    model = define_model(response_template)

    if model is not None: print("model defined correctly")

    # qui l'LLM genera il task ragionando sul formato del prompt (popolato) che deve rispettare
    result = model.invoke(prompt).to_json()

    #in qualche modo, prendo la response del modello, che dovrebbe essere un task in formato JSON, e la salvo in new_task. 
    if result["kwargs"]["tool_calls"] is None:
        print("No tool calls found in the model response.")
        return {**state, "task": None, "doctor_specs": doctor_specs}

    if result["kwargs"].get("tool_calls"):
        #print("Tool calls found in the model response.")
        new_task = result["kwargs"]["tool_calls"][0]["args"]["task"]
    else:
        print("⚠️ Ollama fallback: parsing raw JSON")

        raw_output = result["kwargs"].get("content", "")

        import json
        try:
            new_task = json.loads(raw_output)
        except Exception:
            print("❌ Raw output was:")
            print(raw_output)
            raise
    
    #imposto come skippate tutte le attività della mia new_task siffatta, che dovrebbero essere fatte prima della datetime finta,
    # in modo che il robot non cerchi di eseguire attività passate. E' un metodo messo a disposizione dall'activity planner 
    # (deterministico)

    set_skipped_activities(new_task, fake_datetime)

    if os.getenv("ABLATION_MODE", "false").lower() == "true":
        log_message(
            "Skipping skipped_activities as per environment variable ABLATION_MODE"
        )
        if result["kwargs"].get("tool_calls"):
            new_task = result["kwargs"]["tool_calls"][0]["args"]["task"]
        else:
            print("⚠️ Ollama fallback: parsing raw JSON")

            raw_output = result["kwargs"].get("content", "")

            import json
            try:
                new_task = json.loads(raw_output)
            except Exception:
                print("❌ Raw output was:")
                print(raw_output)
                raise

    #loggo e ritorno lo stato aggiornato con la newtask, le doctor spec, la data, l'id retrievato.
    log_message("\nTASK_SYNTHESIZER_NODE - new_task")
    log_message(new_task)

    #print(f"TASK CREATA: {new_task}")

    return {
        **state,
        "task": new_task,
        "doctor_specs": doctor_specs,
        "fake_datetime": fake_datetime,
        "patient_id": patient_id,
        "doctor_specifications": doctor_specs,
    }
