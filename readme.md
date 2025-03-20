# LLM4HRC: Large Language Model for Human-Robot Collaboration in Care Homes

## Overview
LLM4HRC is a Python-based project designed to facilitate human-robot collaboration in care homes. It leverages advanced language models to assist patients with their daily activities, ensuring personalized and ethical interactions. The system integrates task planning, progress monitoring, and knowledge management to provide a comprehensive solution for patient care.

## Features
- **Task Management**: Dynamically generates and updates patient tasks based on their needs and doctor specifications.
- **Progress Checking**: Ensures compliance with prescribed activities and provides feedback for corrections.
- **Knowledge Management**: Maintains a vector store of patient information for personalized assistance.
- **Graph Workflow**: Implements a state graph to manage the flow of tasks and decisions.
- **Patient Interaction**: Supports proactive and reactive robot behaviors for patient engagement.
- **Activity Scheduling**: Automatically schedules activities while resolving conflicts based on priority and criticality.

## Project Structure
```
llm4hrc/
├── conversation/              # Saved conversations
├── edges/                     # Conditional edge logic for graph transitions
├── graph/                     # Graph initialization and workflow representation
├── mock_data/                 # Sample patient data for testing
├── nodes/                     # Node implementations for graph workflow
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
    pip install -r pip-requirements.txt


### Running the Application
Start the application by running:
```bash
python main.py
```

### Workflow
The application uses a state graph to manage the workflow:
1. **Task Synthesizer Node**: Generates tasks based on patient data.
2. **Situation Assessment Node**: Evaluates the current state.
3. **Next Steps**: Decides the next step
4. **Progress Checking Node**: Validates task compliance.
5. **Robot Effector Node**: Executes actions and interacts with the patient.
6. **Knowledge Manager Node**: Updates the knowledge base with new insights.

## Configuration
- **Environment Variables**: Configure API keys, model settings, and robot behavior in the `.env` file.
- **Graph Workflow Images**: Enable saving workflow diagrams by setting `SAVE_GRAPH_WORKFLOW_IMAGES` to `true`.

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
