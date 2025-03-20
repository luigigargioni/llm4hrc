import os
from langgraph.graph import StateGraph

from utils.interfaces import GRAPH_CONFIG, INITIAL_GRAPH_STATE, GraphNodes, GraphState
from nodes.robot_effector_node import robot_effector_node
from nodes.situation_assessment_node import situation_assessment_node
from nodes.knowledge_manager_node import knowledge_manager_node
from nodes.progress_checking_node import progress_checking_node
from nodes.robot_perception_node import robot_perception_node
from nodes.task_synthesizer_node import task_synthesizer_node
from nodes.task_progress_node import task_progress_node
from nodes.next_steps_node import next_steps_node
from edges.situation_assessment_conditional_edges import (
    situation_assessment_conditional_edges,
)
from edges.progress_checking_conditional_edges import (
    progress_checking_conditional_edges,
)
from edges.knowledge_manager_conditional_edges import (
    knowledge_manager_conditional_edges,
)
from edges.robot_effector_conditional_edges import (
    robot_effector_conditional_edges,
)


def graph_init():
    # Create the graph workflow
    graph_builder = StateGraph(GraphState)

    # Define the nodes
    graph_builder.add_node(GraphNodes.ROBOT_EFFECTOR.value, robot_effector_node)
    graph_builder.add_node(
        GraphNodes.SITUATION_ASSESSMENT.value, situation_assessment_node
    )
    graph_builder.add_node(GraphNodes.KNOWLEDGE_MANAGER.value, knowledge_manager_node)
    graph_builder.add_node(GraphNodes.PROGRESS_CHECKING.value, progress_checking_node)
    graph_builder.add_node(GraphNodes.ROBOT_PERCEPTION.value, robot_perception_node)
    graph_builder.add_node(GraphNodes.TASK_SYNTHESIZER.value, task_synthesizer_node)
    graph_builder.add_node(GraphNodes.TASK_PROGRESS.value, task_progress_node)
    graph_builder.add_node(GraphNodes.NEXT_STEPS.value, next_steps_node)

    # Define the edges
    graph_builder.set_entry_point(GraphNodes.KNOWLEDGE_MANAGER.value)
    graph_builder.add_edge(
        GraphNodes.TASK_SYNTHESIZER.value,
        GraphNodes.SITUATION_ASSESSMENT.value,
    )
    graph_builder.add_edge(
        GraphNodes.ROBOT_PERCEPTION.value,
        GraphNodes.SITUATION_ASSESSMENT.value,
    )
    graph_builder.add_edge(
        GraphNodes.TASK_PROGRESS.value,
        GraphNodes.NEXT_STEPS.value,
    )
    graph_builder.add_edge(
        GraphNodes.NEXT_STEPS.value,
        GraphNodes.PROGRESS_CHECKING.value,
    )
    graph_builder.add_conditional_edges(
        GraphNodes.SITUATION_ASSESSMENT.value,
        situation_assessment_conditional_edges,
    )
    graph_builder.add_conditional_edges(
        GraphNodes.PROGRESS_CHECKING.value,
        progress_checking_conditional_edges,
    )
    graph_builder.add_conditional_edges(
        GraphNodes.KNOWLEDGE_MANAGER.value,
        knowledge_manager_conditional_edges,
    )
    graph_builder.add_conditional_edges(
        GraphNodes.ROBOT_EFFECTOR.value,
        robot_effector_conditional_edges,
    )

    # Compile and run the workflow
    compiled_graph = graph_builder.compile()

    if os.getenv("SAVE_GRAPH_WORKFLOW_IMAGES") == "true":
        # Save the graph workflow as a Mermaid PNG
        with open("graph/representation/graph_workflow_mermaid_image.png", "wb") as f:
            diagram_mermaid_image = compiled_graph.get_graph().draw_mermaid_png()
            f.write(diagram_mermaid_image)

        # Save the graph workflow as a Mermaid file
        with open("graph/representation/graph_workflow_mermaid_diagram.txt", "w") as f:
            diagram_mermaid_data = compiled_graph.get_graph().draw_mermaid()
            f.write(diagram_mermaid_data)

        # Save the graph workflow as a Graphviz PNG
        with open("graph/representation/graph_workflow_graphviz_image.png", "wb") as f:
            diagram_graphviz_image = compiled_graph.get_graph().draw_png()  # type: ignore
            f.write(diagram_graphviz_image)

    # Graph workflow execution
    inputs = INITIAL_GRAPH_STATE
    invoked_graph = compiled_graph.invoke(
        inputs,
        GRAPH_CONFIG,
    )

    return invoked_graph
