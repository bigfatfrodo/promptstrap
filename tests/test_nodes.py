from promptstrap.llm import MixtralLLM, OpenAILLM

from langgraph.graph import StateGraph

from promptstrap.state import PromptstrapState
from promptstrap.graph import node_analyze_prompt


llm = OpenAILLM()

graph = StateGraph(state_schema=PromptstrapState)
graph.add_node("AnalyzePrompt", node_analyze_prompt)

graph.set_entry_point("AnalyzePrompt")
compiled_graph = graph.compile()

input_dict = {
    "input": (
        """Create a web application that allows users to interact with an llm by inputting text prompts and reciving messages.
        Use a color palette based on Bright Blue, Dark Teal, Aqua Orange.
        The application sohuld be mobile-friendly and responsive.
        """
    )
}

result = compiled_graph.invoke(input_dict)

print("Generated repository name:", result)