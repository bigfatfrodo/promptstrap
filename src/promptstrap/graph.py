from langchain.prompts import PromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from promptstrap.state import PromptstrapState, Component, Style, Status
from promptstrap.llm import OpenAILLM, MixtralLLM


llm = OpenAILLM(
    system_prompt="""
                You are a helpful UI coding assistant, specialized in creating web applications with React, Tailwind CSS, and shadc/ui.
                """
)


def node_analyze_prompt(state: PromptstrapState) -> PromptstrapState:

    json_parser = JsonOutputParser(pydantic_object=PromptstrapState)

    prompt = PromptTemplate.from_template(
        """
        Analyze the following prompt: {input}.
        Return a JSON object with a plan on how to implement the prompt as a web application.
        The plan should include:
        - the project and repository names - the project name must be given, but don't worry about the url right now, just set it to empty string
        - a list of components - e.g. a button, a form, a table, etc, where each component has:
           - a unique name
           - functionality description
           - this may be a static component (such as a text or image) in which case assign a file name to the source field; these files will be created later
           - the parent component, if any
        - a style object with:
            - a theme (e.g. light, dark, etc)
            - a font (e.g. sans-serif, serif, etc)
            - a color palette (e.g. ['#ffffff', '#000000', '#ff0000'])
        - a list of behaviors for the overall application (e.g. responsive, mobile-friendly, etc)
        - a list of dependencies that should be installed when setting up the project (e.g. react, tailwindcss, etc)

        Here are the schema instructions for the JSON object. You should strictly follow these instructions:
        {format_instructions}
        """,
        partial_variables={
            "format_instructions": json_parser.get_format_instructions()
        },
    )

    chain = prompt | llm | json_parser

    result = chain.invoke({"input": state.input})

    return result
