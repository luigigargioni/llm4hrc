# LLM4HRC: Hybrid Architecture for Human-Robot Collaboration in Care Homes
# NEW PROJECT README

## Overview
LLM4HRC is a Python-based project designed to facilitate human-robot collaboration in care homes. It leverages advanced language models to assist patients with their daily activities, ensuring personalized and ethical interactions. The system integrates task planning, progress monitoring, and knowledge management to provide a comprehensive solution for patient care. It also integrates new modules for goal decomposition, world state updating, plan validation and preconditions/effect action mapping.

## Features introduced by the new version
- **Action canonicalizer**: maps every natural language action into a canonical action, taking as source "planning/canonical_actions.py (now really  "canonical_actions_test.py")
- **World state updater**: updates the state of the world according to the effects of the executed actions
- **plan validator**: Validates the actions, setting if they're executable or not, based on their preconditions and time windows.
- **robot perception conditional edge**: system edge to determine whether a new goal should be reached and decomposed into new activities or not
- **Goal planner**: decomposes the new request into activities to execute, based on patient request

## New Project Structure
```
llm4hrc/
├── conversation/              # Saved conversations
├── edges/                     # Conditional edge logic for graph transitions
├── graph/                     # Graph initialization and workflow representation
├── mock_data/                 # Sample patient data for testing
│   └── variants/              # Alternative care plans for replay runs (not loaded into the knowledge base)
├── nodes/                     # Node implementations for graph workflow
├── planning/                  # Canonical actions to be used for mapping
├── tests/                     # Offline test scripts (run from the repo root: python tests/test_<name>.py)
├── utils/                     # Utility functions and interfaces
├── .env                       # Environment variables configuration
├── .flake8                    # Linting configuration
├── .gitignore                 # Git ignore rules
├── main.py                    # Entry point for the application
├── README.md                  # Project documentation
```

## Getting Started

To set up the same environment on a new system, follow these steps:

1. **Install Conda** (if not already installed):  
   Download and install Miniconda or Anaconda from [conda.io](https://docs.conda.io/en/latest/miniconda.html).

2. **Create a new Conda environment** using the exported Conda packages:  
   ```bash
    conda env create -f environment.yml

3. **Activate the environment**:
   ```bash
    conda activate my_env

4. **Install additional pip packages**:
   ```bash
    pip install -r requirements.txt


### Running the Application
Start the application by running:
```bash
python main.py
```

### Workflow
The application uses a state graph to manage the workflow:
1. **Task Synthesizer Node**: Generates tasks based on patient data.
2. **Situation Assessment Node**: Evaluates the current state.
3. **Action canonicalizer Node**: canonicalizes actions
4. **World state updater Node**: updates the world state
5. **Plan validator Node**: validates actions
6. **Next Steps**: Decides the next step
7. **Progress Checking Node**: Validates task compliance.
8. **Robot Effector Node**: Executes actions and interacts with the patient.
9. **Knowledge Manager Node**: Updates the knowledge base with new insights.
10. **Robot perception Node**: Percepts a new request
11. **Goal planner Node**: Plans new actions (eventually)

## Configuration
- **Environment Variables**: Configure API keys, model settings, and robot behavior in the `.env` file.
- **Graph Workflow Images**: Enable saving workflow diagrams by setting `SAVE_GRAPH_WORKFLOW_IMAGES` to `true`.

## Future Works
- Add timeout handling for idle states.
- Support activities without specific time constraints.
- Enhance progress checking with rule-based and retrieval-augmented generation (RAG) techniques.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For any questions or inquiries, feel free to contact me at [luigi.gargioni@unibs.it](mailto:luigi.gargioni@unibs.it).
Contributor: [a.maffezzoli001@studenti.unibs.it](mailto:a.maffezzoli001@studenti.unibs.it).
