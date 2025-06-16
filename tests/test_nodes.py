from promptstrap.llm import MixtralLLM, OpenAILLM

from langgraph.graph import StateGraph

from langgraph.graph import StateGraph

from promptstrap.state import PromptstrapState
from promptstrap.graph import node_analyze_prompt, node_create_repo, node_files

from pprint import pprint


llm = OpenAILLM()


def test_nodes():
    graph = StateGraph(state_schema=PromptstrapState)
    graph.add_node("AnalyzePrompt", node_analyze_prompt)
    graph.add_node("CreateRepo", node_create_repo)
    graph.add_node("CreateFiles", node_files)

    graph.add_edge("AnalyzePrompt", "CreateRepo")
    graph.add_edge("CreateRepo", "CreateFiles")

    graph.set_entry_point("AnalyzePrompt")

    compiled_graph = graph.compile()

    input_dict = {
        "input": (
            """Design a web application that allows users to interact with an llm by inputting text prompts and receiving messages.
            Use a color palette based on Bright Blue, Dark Teal, Aqua Orange.
            The application sohuld be mobile-friendly and responsive.
            The header will display a futuristic image of a city skyline at night with neon lights.
            The footer will contain links to the terms of service, privacy policy, and contact information.
            """
        )
    }

    result = compiled_graph.invoke(input_dict)

    pprint(result)
