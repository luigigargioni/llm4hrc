# LLM4HRC: Hybrid Architecture for Human-Robot Collaboration in Care Homes

## Overview
LLM4HRC is a Python-based project designed to facilitate human-robot collaboration in care homes. It leverages large language models (LLMs) to assist patients with their daily activities, ensuring personalized and context-aware interactions. The system combines an LLM-driven state graph workflow with a deterministic activity planner (expert system with forward-chaining rules) and a RAG-based knowledge manager to provide a comprehensive solution for patient care.

## Features
- **Task Synthesis**: Dynamically generates structured daily task plans for patients from a vector knowledge base, with optional doctor overrides.
- **Deterministic Activity Planner**: Schedules activities using forward-chaining rules, resolving conflicts based on priority, criticality, and time constraints.
- **Progress Checking**: Validates task compliance after each robot–patient interaction using a dedicated LLM.
- **Knowledge Management**: Maintains a persistent Chroma vector store of patient information for personalized retrieval.
- **Graph Workflow**: Implements a LangGraph state graph (`StateGraph`) to orchestrate all nodes and transitions.
- **Patient Interaction**: Supports both proactive and reactive robot behaviors configurable via environment variables.
- **Multi-provider LLM Support**: Compatible with Ollama (local), OpenAI, and Groq as LLM backends.
- **Conversation & Log Persistence**: Saves structured conversation logs and detailed graph-state logs for each session.

## Project Structure
```
llm4hrc/
├── conversation/              # Saved per-session conversation logs
│   └── case_study/            # Case study conversation records
├── deterministic/             # Deterministic components
│   ├── activity_planner.py    # Forward-chaining activity scheduler (integrated)
│   └── backward_chaining.py   # Experimental backward-chaining planner (deprecated)
├── edges/                     # Conditional edge logic for graph transitions
│   ├── knowledge_manager_conditional_edges.py
│   ├── progress_checking_conditional_edges.py
│   ├── robot_effector_conditional_edges.py
│   └── situation_assessment_conditional_edges.py
├── graph/                     # Graph initialization and workflow representation
│   ├── graph_init.py          # StateGraph definition, compilation, and invocation
│   └── representation/        # Saved Mermaid/Graphviz workflow diagrams
├── mock_data/                 # Sample patient records for testing (123, 456, 789, 0789)
├── nodes/                     # Node implementations for the graph workflow
│   ├── knowledge_manager_node.py
│   ├── next_steps_node.py
│   ├── progress_checking_node.py
│   ├── robot_effector_node.py
│   ├── robot_perception_node.py
│   ├── situation_assessment_node.py
│   ├── task_progress_node.py
│   └── task_synthesizer_node.py
├── utils/                     # Shared utilities
│   ├── interfaces.py          # TypedDicts, enums, graph state, constants
│   ├── logs.py                # Log and conversation file helpers
│   └── models.py              # LLM provider factory (Ollama, OpenAI, Groq)
├── .env.example               # Example environment variables configuration
├── .flake8                    # Linting configuration
├── .gitignore                 # Git ignore rules
├── environment.yml            # Conda environment definition
├── pip-requirements.txt       # Additional pip dependencies
├── main.py                    # Entry point for the application
└── README.md                  # Project documentation
```

## Getting Started

### Prerequisites
- **Conda** (Miniconda or Anaconda): [conda.io](https://docs.conda.io/en/latest/miniconda.html)
- At least one LLM backend configured (see [Configuration](#configuration))

### Environment Setup

1. **Create the Conda environment**:
   ```bash
   conda env create -f environment.yml
   ```

2. **Activate the environment**:
   ```bash
   conda activate my_env
   ```

3. **Install additional pip packages**:
   ```bash
   pip install -r pip-requirements.txt
   ```

4. **Configure environment variables**:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` with your API keys and settings (see [Configuration](#configuration)).

### Running the Application
```bash
python main.py
```

At startup, the application prompts for:
- **Patient ID** — one of the available IDs (`123`, `456`, `789`, `0789`)
- **Doctor specifications** — optional free-text overrides for the day's activities
- **Simulation time** — a start time in `HH:MM` format used as the fake datetime for scheduling

## Graph Workflow

The application uses a LangGraph `StateGraph` with the following nodes:

| Node | Description |
|---|---|
| `knowledge_manager_node` | **Entry point.** Initializes/retrieves the Chroma vector store with patient knowledge. |
| `task_synthesizer_node` | Generates a structured daily task plan via LLM, applying doctor specs and activity scheduling. |
| `situation_assessment_node` | Evaluates the current context to decide the next robot action. |
| `robot_perception_node` | Captures patient input/observations from the environment. |
| `next_steps_node` | Determines the subsequent activity based on task progress. |
| `task_progress_node` | Advances task state (marks activities done/skipped). |
| `progress_checking_node` | Validates whether the patient correctly completed an activity. |
| `robot_effector_node` | Executes the robot's response and interacts with the patient. |

Conditional edges govern transitions between nodes based on the `GraphState`.

## Configuration

All settings are managed via the `.env` file (copy from `.env.example`):

| Variable | Description | Default |
|---|---|---|
| `SYSTEM_LANGUAGE` | Language for robot responses (`italian` or `english`) | `italian` |
| `LLM_PROVIDER` | LLM backend: `ollama`, `openai`, or `groq` | `ollama` |
| `OLLAMA_MODEL` | Ollama model name (e.g. `llama3.3`, `llama3.1`, `llama4`) | `llama3.3` |
| `OPENAI_MODEL` | OpenAI model name (e.g. `gpt-4.1-mini`, `gpt-4o`) | `gpt-4.1-mini` |
| `GROQ_MODEL` | Groq model name (e.g. `llama-3.3-70b-versatile`) | `llama-3.3-70b-versatile` |
| `PROGRESS_CHECKING_MODEL` | Ollama model for progress checking (e.g. `bespoke-minicheck`) | `llama3.1` |
| `EMBEDDINGS_MODEL` | Ollama model for embeddings | `llama3.1` |
| `OPENAI_API_KEY` | OpenAI API key (required if using OpenAI) | — |
| `GROQ_API_KEY` | Groq API key (required if using Groq) | — |
| `LANGCHAIN_TRACING_V2` | Enable LangSmith tracing | `false` |
| `LANGCHAIN_API_KEY` | LangSmith API key | — |
| `ROBOT_BEHAVIOUR` | Robot interaction mode: `proactive` or `reactive` | `proactive` |
| `ABLATION_MODE` | Disable deterministic activity scheduling (ablation study) | `true` |
| `AUDIO_ENABLED` | Enable audio output | `false` |
| `SAVE_GRAPH_WORKFLOW_IMAGES` | Save Mermaid/Graphviz workflow diagrams to `graph/representation/` | `true` |

## Notes on Experimental Components
The repository contains a `backward_chaining.py` module developed during an early exploratory phase of the project. This component is a proof of concept for goal-driven task generation and is **not integrated into the main workflow**.
This file is retained in the repository for transparency but should be considered deprecated. A more robust planning component is identified as a direction for future work.

## Future Works
- Add timeout handling for idle states.
- Support activities without specific time constraints.
- Enhance progress checking with rule-based and retrieval-augmented generation (RAG) techniques.
- Improve task prioritization and scheduling logic.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For any questions or inquiries, feel free to contact me at [luigi.gargioni@unibs.it](mailto:luigi.gargioni@unibs.it).
