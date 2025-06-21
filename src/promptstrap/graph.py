import base64
import os
import re
import shutil
import subprocess
import threading
import time
from enum import Enum
from pprint import pprint
from typing import List, Optional

import tqdm
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers.json import JsonOutputParser
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from promptstrap.llm import MixtralLLM, OpenAILLM
from promptstrap.state import FileState, FileType, PromptstrapState, Status
from promptstrap.tools import util_create_image

llm = OpenAILLM(
    system_prompt="""
                You are an experienced frontend software engineer, specialized in creating web applications with React, Tailwind CSS, and shadcn/ui.
                Coding rules you must follow:
                - In JS/React files, when you need to reference other files, avoid using paths directly. You should import them and reference the imported names.
                    For example, like this:
                      import horseImage from '../assets/horse.jpeg;
                     ... and later, to use it:
                           <img src={{horseImage}} ... /> 
                - Instead of <a> links, use the Link component from react-router-dom
                - You should ensure that the color of the text in input is not close to the background color of the input field.
                - You use vite as the build tool for the React application.
                """
)


def node_analyze_prompt(state: PromptstrapState) -> PromptstrapState:
    print("ANALYZE_PROMPT Enter")

    json_parser = JsonOutputParser(pydantic_object=PromptstrapState)
    prompt = PromptTemplate.from_template(
        """
        Analyze the following prompt: {input}.
        Return a JSON object with a plan on how to implement the prompt as a web application.
        The plan should include:
        - the project and repository names
        - a complete and exhaustive list of all files with their paths (including package.json), types and prompt for a generative model to generate each file. This should be a typical file organization for a React web application for this purpose.
            - the list of files should include source files, assets files, css files, html files, js files, tsx, jpeg, json and other files, as needed
            - if you need to add an image in a component you should include that jpeg image file in the file list, together with its prompt
            - create the file list in a bottom up order, so that when you will generate the files later, the content of lower level files will be available to the higher level files
            - add tailwind.config.js and postcss.config.js to the files to be generated.
            - leave package.json last, so it will be generated last to include all the dependencies
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
    print("REPO Enter")
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


def more_files(state: PromptstrapState) -> bool:
    if state.files is None:
        return False
    return len([f for f in state.files if f.state != FileState.GENERATED]) > 0


def node_files(state: PromptstrapState) -> PromptstrapState:
    print("FILES Enter")
    if state.files is None:
        return state

    if state.repo_state != FileState.GENERATED:
        raise ValueError("Repository must be created before generating files.")

    if state.repo_name is None:
        raise ValueError("Repository name must be set before creating files.")

    json_parser = JsonOutputParser(pydantic_object=CreateFileResponse)
    prompt = PromptTemplate.from_template(
        """
        Your current task is to create the file found at {path}: {input},
        The file syntax should be correct and follow the conventions of the specified file type. You should also respect your own coding rules.
        You should output the content of the file in the JSON object detailed below. Do not include any other text in the output.
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

    todo_files = [f for f in state.files if f.state != FileState.GENERATED]
    if len(todo_files) != 0:
        file = todo_files[0]
        state.functional_tests_cycles = (
            # if we update a file and there are no functional test cycles left, set it to 1
            # to run them once again
            1
            if state.functional_tests_cycles == 0
            else state.functional_tests_cycles
        )
    else:
        return clear_error_status(state)

    print(f"{file.path}, {len(todo_files)} / {len(state.files)} files remaining.")
    folder_path = os.path.join(state.output_folder, state.repo_name)
    file_path = os.path.join(folder_path, file.path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if file.state == FileState.NEEDS_SYNC:
        with open(file_path, "w") as f:
            f.write(file.content)
        file.state = FileState.GENERATED
        return clear_error_status(state)

    if file.type in [
        FileType.JPEG_WIDE,
        FileType.JPEG_TALL,
        FileType.JPEG_SQUARE,
    ]:
        # if the file is an image, create it using the util_create_image function
        util_create_image(file.type, file.prompt, file_path)
        file.state = FileState.GENERATED
        return clear_error_status(state)

    with open(file_path, "w") as f:
        # just write the prompt to the file for now
        try:
            result_json = chain.invoke(
                {
                    "input": file.prompt,
                    "path": file.path,
                    "state_structure": state.model_dump_json(),
                }
            )
        except Exception as e:
            print(f"Error creating file {file.path}: {e}")
            f.write(f"Error: {e}\n")
            file.state = FileState.ERROR
            return state

        if result_json.get("status", Status.SUCCESS) == Status.SUCCESS:
            try:
                f.write(result_json["file_content"])
                file.state = FileState.GENERATED
                file.content = result_json["file_content"]
            except Exception as e:
                print(f"Error writing file {file.path}: {e}")
                pprint(result_json)
                f.write(f"Error: {e}\n")
                file.state = FileState.ERROR
        else:
            f.write(f"Error: {result_json['error']}\n")
            print(f"Error creating file {file.path}: {result_json['error']}")
            file.state = FileState.ERROR

    return clear_error_status(state)


def node_install_dep(state: PromptstrapState) -> PromptstrapState:
    print("INSTALL_DEP Enter")
    # call npm install in the repo folder
    # if this fails, set status to error and capture the error message

    # all files should be in the generated state
    error_message = ""

    if state.files is None:
        return state

    if state.status != Status.SUCCESS:
        # if we enter here with an error, set all files to be updated and try again
        for file in state.files:
            file.state = FileState.NEEDS_UPDATE

    for file in state.files:
        if file.state != FileState.GENERATED:
            error_message += f"File {file.path} is not generated.\n"
            file.state = FileState.NEEDS_UPDATE
            state.status = Status.ERROR
            state.dep_result = error_message

    # if after the build we have an error, it is possible that
    # either some files need to be added
    # or some files need to be updated

    # run npm install in the repo folder
    folder_name = os.path.join(state.output_folder, state.repo_name)

    result = subprocess.run(
        ["npm", "install"],
        cwd=folder_name,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        error_message += f"Error installing dependencies: \nSTDOUT:\n{result.stdout}\n STDERR:\n{result.stderr}\n"
        state.status = Status.ERROR
        state.dep_result = error_message
        print("Npm install failed with errors:\n", error_message)

        return state

    else:
        print("npm install completed successfully.")

    return clear_error_status(state)


def node_build(state: PromptstrapState) -> PromptstrapState:
    print("BUILD Enter")

    if state.status != Status.SUCCESS:
        raise ValueError("Cannot build project with errors.")

    folder_name = os.path.join(state.output_folder, state.repo_name)

    result = subprocess.run(
        ["npm", "run", "build"],
        cwd=folder_name,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        error_message = f"Error building project: \nSTDOUT:\n{result.stdout}\n STDERR:\n{result.stderr}\n"
        state.status = Status.ERROR
        state.build_result = error_message
        print("Build failed with errors:\n", error_message)

        return state
    else:
        print("Build completed successfully.")
        return clear_error_status(state)


def start_vite(folder_name):
    print("Starting Vite development server...")
    process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=folder_name,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    print("Popen done")
    for line in process.stdout:
        print("[vite] ", line.strip())
        match = re.search(r"(http:\/\/\S+)\/", line)
        if match:
            vite_url = match.group(1)
            break
    print("Vite URL found:", vite_url)
    return process, vite_url


def start_selenium(vite_url, folder_name):

    import io
    from io import BytesIO

    from PIL import Image
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    print(f"Starting Selenium... {vite_url} {folder_name}")
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1920,1080")
    driver = webdriver.Chrome(options=options)
    print(f"getting |{vite_url}|")
    time.sleep(5)
    driver.get(vite_url.strip())
    time.sleep(5)
    png_bytes = driver.get_screenshot_as_png()
    driver.quit()
    screenshot_path = os.path.join(folder_name, "screenshot.jpg")
    image = Image.open(BytesIO(png_bytes)).convert("RGB")
    image.save(screenshot_path, "JPEG", quality=30)

    print("Selenium screenshot taken and saved.")


def screenshot_project(state: PromptstrapState) -> str:
    folder_name = os.path.join(state.output_folder, state.repo_name)

    process, vite_url = start_vite(folder_name)
    start_selenium(vite_url, folder_name)
    process.terminate()
    process.wait()
    print("Vite development server stopped.")

    return os.path.join(folder_name, "screenshot.jpg")


def node_test_project(state: PromptstrapState) -> PromptstrapState:
    print("TEST Enter")

    if state.status != Status.SUCCESS:
        raise ValueError("Cannot test project with errors.")

    state.functional_tests_cycles -= 1

    class UpdateActionList(BaseModel):
        current_state: str
        instructions: List[str] = []

    path = screenshot_project(
        PromptstrapState(input=state.input, repo_name=state.repo_name)
    )

    with open(path, "rb") as f:
        base64_img = base64.b64encode(f.read()).decode("utf-8")

    parser = JsonOutputParser(pydantic_object=UpdateActionList)
    messages = [
        SystemMessage(content="You are a UI/UX design assistant."),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"""Analyze the following project screenshot and describe its layout, components, color palette.
                    You will receive a base64-encoded image of the project and a prompt describing the project.
                    Evaluate the current state (the image) and provide instructions to add what is missing from the desired state (the prompt).
                    If the current state is an empty page, add just a single instruction to investigate this bug.
                    Don't give general advice, focus on the specific elements that need to be added or modified. If there are no changes needed, return an empty list.

                    Desired state prompt:
                    {state.input}
                    Use a json format for your response, with the following structure:
                    {parser.get_format_instructions()}""",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_img}"},
                },
            ]
        ),
    ]
    result = llm.invoke(messages)
    result = parser.parse(result.content)

    if len(result.instructions) > 0:
        state.test_results = result.instructions
        state.status = Status.ERROR

    return state


def clear_error_status(state: PromptstrapState) -> PromptstrapState:
    state.status = Status.SUCCESS
    state.dep_result = state.build_result = state.test_results = None
    return state


def node_apply_observations(state: PromptstrapState) -> PromptstrapState:
    print("APPLY_OBSERVATIONS Enter")

    parser = JsonOutputParser(pydantic_object=PromptstrapState)
    apply_common_prompt = f"""
            In the JSON object below you will find the list of files currently in the project. 
            Please analyze the issues and do the necessary updates in the file content to fix the errors and set the state of those files:
               - those that need to be update to {FileState.NEEDS_SYNC}.
               - If you need to add a new file, then feel free to add it to the list and also set the state to {FileState.NEEDS_SYNC}.
               - if no change is needed for a file, just leave it as is in the list.
            {{state_structure}}

            The format to be returned:
            {{format_instructions}}
    """
    install_dep_prompt = """
            The npm install command failed with the following error message:
            {error_message}
            """
    build_failed_prompt = """
            The build command failed with the following error message:
            {error_message}

    """

    test_results_prompt = """
            Functional tests failed with the following instructions:
            {error_message}
    """
    state.status = Status.SUCCESS

    if state.dep_result is not None:
        prompt_starter = install_dep_prompt + apply_common_prompt
        error_message = state.dep_result
    elif state.build_result is not None:
        prompt_starter = build_failed_prompt + apply_common_prompt
        error_message = state.build_result
    elif state.test_results is not None:
        prompt_starter = test_results_prompt + apply_common_prompt
        error_message = "\n".join(state.test_results)
    else:
        print("No errors to apply observations for.")
        return clear_error_status(state)

    prompt_template = PromptTemplate.from_template(
        prompt_starter,
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
        },
    )
    chain = prompt_template | llm | parser
    chain_result = chain.invoke(
        {
            "error_message": error_message,
            "state_structure": state.model_dump_json(),
        }
    )
    state.files = chain_result["files"]
    return clear_error_status(state)


ANALYZE_PROMPT = "AnalyzePrompt"
CREATE_REPO = "CreateRepo"
CREATE_FILES = "CreateFiles"
INSTALL_DEP = "InstallDep"
BUILD_PROJECT = "BuildProject"
TEST_PROJECT = "TestProject"
APPLY_OBSERVATIONS = "ApplyObservationsToStateFiles"


def NewGraph():

    graph = StateGraph(state_schema=PromptstrapState)
    graph.add_node(ANALYZE_PROMPT, node_analyze_prompt)
    graph.add_node(CREATE_REPO, node_create_repo)
    graph.add_node(CREATE_FILES, node_files)
    graph.add_node(INSTALL_DEP, node_install_dep)
    graph.add_node(BUILD_PROJECT, node_build)
    graph.add_node(TEST_PROJECT, node_test_project)
    graph.add_node(APPLY_OBSERVATIONS, node_apply_observations)

    graph.add_edge(ANALYZE_PROMPT, CREATE_REPO)
    graph.add_edge(CREATE_REPO, CREATE_FILES)
    graph.add_edge(APPLY_OBSERVATIONS, CREATE_FILES)

    graph.add_conditional_edges(
        CREATE_FILES,
        more_files,
        {
            True: CREATE_FILES,
            False: INSTALL_DEP,
        },
    )

    graph.add_conditional_edges(
        INSTALL_DEP,
        lambda state: state.status,
        {
            Status.SUCCESS: BUILD_PROJECT,
            Status.ERROR: APPLY_OBSERVATIONS,
        },
    )

    graph.add_conditional_edges(
        BUILD_PROJECT,
        lambda state: state.status,
        {Status.SUCCESS: TEST_PROJECT, Status.ERROR: APPLY_OBSERVATIONS},
    )

    graph.add_conditional_edges(
        TEST_PROJECT,
        lambda state: state.status,
        {
            Status.SUCCESS: END,
            Status.ERROR: APPLY_OBSERVATIONS,
        },
    )

    graph.set_entry_point(ANALYZE_PROMPT)

    compiled_graph = graph.compile()

    return compiled_graph
