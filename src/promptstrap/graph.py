from typing import Optional
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from promptstrap.state import PromptstrapState
from promptstrap.llm import OpenAILLM, MixtralLLM
from promptstrap.state import FileState, Status
import os
import shutil
from pydantic import BaseModel
import tqdm
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
        - a complete and exhaustive list of all files with their paths, types and prompt for a generative model to generate each file. This should be a typical file organization for a React web application for this purpose.
            - the list of files should include source files, assets files, css files, html files, js files, tsx, jpeg, json and other files, as needed
            - if you need to add an image in a component you should include that jpeg image file in the file list, together with its prompt
        - a style object with:
            - a theme (e.g. light, dark, etc)
            - a font (e.g. sans-serif, serif, etc)
            - a color palette (e.g. ['#ffffff', '#000000', '#ff0000'])
        - a list of behaviors for the overall application (e.g. responsive, mobile-friendly, etc)

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


def node_create_repo(state: PromptstrapState) -> PromptstrapState:
    # TODO: Make this actuall create a repo. For now use an ouput folder
    if state.repo_name is None:
        state.repo_name = "default-repo-name"
    folder_name = os.path.join(state.output_folder, state.repo_name)
    folder_name = state.output_folder + "/" + state.repo_name
    try:
        # os.rmdir only removes empty directories. To remove a non-empty directory, use shutil.rmtree.
        shutil.rmtree(folder_name)
    except FileNotFoundError:
        pass
    os.makedirs(folder_name)

    state = state.model_copy(update={"repo_state": FileState.GENERATED}, deep=True)
    return state


class CreateFileResponse(BaseModel):
    file_content: str
    status: Status = Status.SUCCESS
    error: Optional[str] = None


def node_create_files(state: PromptstrapState) -> PromptstrapState:
    if state.files is None:
        return state

    if state.repo_state != FileState.GENERATED:
        raise ValueError("Repository must be created before generating files.")

    if state.repo_name is None:
        raise ValueError("Repository name must be set before creating files.")

    json_parser = JsonOutputParser(pydantic_object=CreateFileResponse)
    prompt = PromptTemplate.from_template(
        """
        Your current task is to create this file: {input},
        The file syntax should be correct and follow the conventions of the specified file type.
        {fromat_instructions}
        If the file is in a format that you cannot generate, you should set the status accordingly and return an error message
        in the corresponding json field.
        This file is part of a web application project. Below is the context of the project and how it is built:
        {state_structure}
        """,
        partial_variables={
            "fromat_instructions": json_parser.get_format_instructions()
        },
    )

    chain = prompt | llm | json_parser
    total_files = len([f for f in state.files if f.state == FileState.PLANNED])
    with tqdm.tqdm(total=total_files, desc="Creating files") as pbar:
        for file in state.files:
            pbar.update(1)
            if file.state != FileState.PLANNED:
                continue

            folder_path = os.path.join(state.output_folder, state.repo_name)
            file_path = os.path.join(folder_path, file.path)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, "w") as f:
                # just write the prompt to the file for now
                try:
                    result_json = chain.invoke(
                        {
                            "input": file.prompt,
                            "state_structure": state.model_dump_json(),
                        }
                    )
                except Exception as e:
                    print(f"Error creating file {file.path}: {e}")
                    f.write(f"Error: {e}\n")
                    file.state = FileState.ERROR
                    continue

                if result_json.status == Status.SUCCESS:
                    f.write(result_json.file_content)
                    file.state = FileState.GENERATED
                else:
                    f.write(f"Error: {result_json.error}\n")
                    print(f"Error creating file {file.path}: {result_json.error}")
                    file.state = FileState.ERROR

    return state.model_copy(deep=True)
