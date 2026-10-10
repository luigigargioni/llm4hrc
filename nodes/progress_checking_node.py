# Questo nodo è un:

# validator
# safety checker
# compliance checker
# critic model

# cioè un secondo LLM che controlla se:

# la risposta del robot
# l’azione scelta
# il comportamento

# siano:

# coerenti col task
# sicuri
# etici
# aderenti alla prescrizione

# In pratica:

# next_steps_node genera una proposta
# progress_checking_node la giudica


from datetime import datetime
import os
from groq import BaseModel
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from utils.interfaces import ProgressCheckingResponseTypes, ProgressCheckingResponses
import json
from colorama import Fore, Style

from graph.graph_init import GraphState
from utils.models import TEMPERATURE, OllamaModel, define_model
from utils.logs import log_graph_state, log_message

# MODIFICATO L'INITIAL PROMPT, ERA TROPPO RIGIDO, RISPONDEVA KO SE LA PROPOSTA NON ERA PERFETTAMENTE "fai ora l'attività X",
# se nell'ora passata in input c'era da fare quella precisa attività. 

#edit, ho fatto ritornare la reason anche in caso di esito positivo modificando la riga "reply only... " aggiungendo esplicitamente
# AND THE REASON sia in caso di esito positivo che negativo.


#TODO : PROVARE A INSTALLARE LO STESSO MODELLO DI GROQ O UNO PIU LEGGERO PER VEDERE SE VELOCIZZA UN PO'

INITIAL_PROMPT = """
# CONTEXT #
You are a robot assistant in a care home. You must help the patient to do his/her activities.
As input you have your action and the vocal answer for the next step of the task.
Evaluate whether the robot behaviour is SAFE, REASONABLE and GENERALLY COMPLIANT with the prescribed activities, the patient information, and the additional instructions reported below.

# OUTPUT #
Return ONLY a valid JSON. Do not include any explanation or text outside JSON. The JSON must have this structure:
- "result": "OK" if the robot action and the vocal answer are compliant, "KO" otherwise.
- "reason": A string with the reason for the result.
Reply only with "OK" and the reason if the robot action and the vocal answer are compliant with the prescribed activities, the patient information, and the additional instructions reported below, or "KO" and the reason otherwise.

# INPUT #
- Prescribed activities and patient information:
{docs_content}

- Robot action:
{robot_action}

- Vocal answer:
{vocal_answer}

- Situation perception:
{situation_perception}

- Patient request:
{patient_request}

# INSTRUCTIONS #
1. Politeness and Helpfulness
   - You must always be polite, kind, and helpful when interacting with patients.
   - Use clear, professional language, avoiding offensive or inappropriate expressions.

2. Ethics and Confidentiality
   - You must respect ethical principles and the privacy of patients in all interactions.
   - Do not disclose or share sensitive information about patients or healthcare professionals.

3. Limitations on Providing Medical Advice
   - You must not provide diagnoses, prescriptions, or medical advice beyond the prescribed activities.
   - Always encourage users to consult a qualified healthcare professional for medical concerns.
   - Do not interpret medical tests or suggest treatments without explicit instructions from qualified professionals.

4. Accuracy and Truthfulness
   - You must provide only verified and accurate information.
   - Do not generate or spread false, misleading, or unverified information.
   - If unsure about an answer, you must clearly state it and refer the user to reliable sources or competent professionals.

5. Handling Patient Information
   - You must not alter, manipulate, or fabricate patient-related information.
   - You must not disclose a patient’s personal information, even to the patient themselves, unless explicitly authorized.

6. Neutrality and Non-Judgmental Approach
   - Maintain a neutral and unbiased approach in responses.
   - Do not discriminate based on ethnicity, gender, religion, sexual orientation, age, or any other personal characteristic.

7. Limitations on Actions and Autonomy
   - You must not make autonomous decisions that could directly impact a patient's health or well-being.

8. Prevention of Abuse and Improper Use
   - Do not engage in inappropriate, threatening, or illegal conversations.
   - Avoid any form of manipulation or influence that could lead users to harmful decisions.

9. Clarity About Capabilities and Limitations
   - Explain your limitations and the need for human intervention when necessary.

10. Flexibility in Compliance Checking
   - The robot is allowed to:
    - greet the patient,
    - explain activities,
    - ask preferences,
    - negotiate timing,
    - converse naturally,
    - delay non-critical activities within reasonable tolerance,
    - adapt communication to the patient state.

    Do NOT return KO simply because:
    - an activity has not yet started,
    - the robot is speaking instead of executing,
    - the robot is greeting or explaining,
    - multiple activities are possible,
    - the patient has not answered yet.

    Return KO ONLY if:
    - the robot suggests dangerous behaviour,
    - the robot violates medical prescriptions,
    - the robot encourages skipping critical medications,
    - the robot provides incorrect medical information,
    - the robot behaves unethically,
    - the robot ignores critical situations,
    - the robot performs actions clearly incompatible with the task.

    Minor conversational deviations are acceptable.
"""

#TOLTO TOOL CALLING COME IN TUTTI GLI ALTRI PUNTI DEL CODICE IN QUANTO NON COMPATIBILE CON OLLAMA!!!

class ProgressCheckingSchema(BaseModel):
    result: str
    reason: str

# @tool
# def response_template(progress_checking: ProgressCheckingResponses) -> bool:
#     """The JSON format for the output of the next steps node.
# 
#     Args:
#         progress_checking: ProgressCheckingResponses
# 
#     """
#     return True


def progress_checking_node(state: GraphState) -> GraphState:
    print(Fore.GREEN + "[position: PROGRESS CHECKING NODE]" + Style.RESET_ALL)

    log_graph_state(state, "progress_checking_node")

    # ----------------------------------------------------------------------------------------
    # REMOVE (troppo lento per testing essendo in locale, però funziona con le modifiche fatte)
    return {**state, "progress_checking_output": True}
    # ----------------------------------------------------------------------------------------

    #se non c'è niente di scritto o il patient id è none, non posso fare niente
    if state["vector_store"] is None:
        raise ValueError("PROGRESS_CHECKING_NODE - vector_store is None")

    if state["patient_id"] is None:
        raise ValueError("PROGRESS_CHECKING_NODE - patient_id is None")
    
    # AGGIUNTA: non valida la prima iterazione del grafo, che di solito è solo greetings e primo orientamento dell'agente.
    if state["patient_request"] is None:
        return {**state, "progress_checking_output": True}

    #recupero attività prescritte e informazioni del paziente con la similarity search, se non le ho già recuperate in precedenza
    #  (per evitare di fare la similarity search ad ogni iterazione del grafo, (ovvero se c'è già qualcosa in progress_checking_similarity_search)
    # che è costosa in termini di tempo e risorse computazionali)

    docs_content = state["progress_checking_similarity_search"]
    if docs_content is None:
        # TODO: Query must be dynamic based on the robot_action and vocal_answer
        retrieved_vectors = state["vector_store"].similarity_search(
            "Retrieve information about the prescribed activities and the patient for the patient with Record Number: "
            + state["patient_id"],
            k=1,
            filter={"patient_id": state["patient_id"]},
        )
        docs_content = "\n\n".join(vector.page_content for vector in retrieved_vectors)

    #creo prompt e invoco

    prompt_template = PromptTemplate.from_template(INITIAL_PROMPT)

    if state["fake_datetime"] is None:
        raise ValueError("next_steps_node - fake_datetime is None")
    formatted_datetime = state["fake_datetime"].strftime("%d-%m-%Y %H:%M:%S")

    prompt = prompt_template.invoke(
        {
            "situation_perception": state["situation_perception"],
            "patient_request": state["patient_request"],
            "robot_action": state["robot_action"],
            "vocal_answer": state["vocal_answer"],
            "docs_content": docs_content,
            # "datetime": formatted_datetime,
        }
    )

    #prendo dall'env la variabile che mi indica quale modello uso per il progress checking, chiamo result

    checking_model = os.getenv("PROGRESS_CHECKING_MODEL") or OllamaModel.LLAMA_3_1.value

    model = ChatOllama(
        model=checking_model,
        temperature=TEMPERATURE, 
        format="json"
    )#.bind_tools([response_template])

    #debug
    print("invocazione modello progress checking...")

    result = model.invoke(prompt)#.to_json()

    if not isinstance(result.content, str):
        raise ValueError("Model response is not a string") #per non far apparire errori del typechecker che dice che result.content potrebbe essere dict o altro, quando invece è sempre stringa (perché format="json" in ChatOllama)

    progress_checking_result = json.loads(result.content)

    #progress_checking_result = result["kwargs"]["tool_calls"][0]["args"][
    #    "progress_checking"
    #]

    log_message(f"PROGRESS_CHECKING_NODE - result: {progress_checking_result}")
    #print(f"PROGRESS_CHECKING_NODE - result: {progress_checking_result}")

    #popolo progress_checking_output in base al risultato del modello, che sarà True se il risultato è OK, False se è KO. Infine lo ritorno

    progress_checking_output = (
        True
        if progress_checking_result["result"] == ProgressCheckingResponseTypes.OK.value
        else False
    )

    #debug
    if progress_checking_output: print("check is good!")

    # [vedere progress_checking_conditional_edges.py] in base a questo output decido: se True vado al robot effector per effettuare l'azione,
    # altrimenti alla situation assessment per capire cosa è andato storto e generare una nuova proposta di updated_task, 
    # robot_action e vocal_answer più aderente alla prescrizione e alla situazione del paziente.

    return {**state, "progress_checking_output": progress_checking_output}
