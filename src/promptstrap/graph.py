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

import shortuuid
import sqlalchemy
import tqdm
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from promptstrap.llm import MixtralLLM, OpenAILLM
from promptstrap.state import (
    FileState,
    FileType,
    PromptstrapState,
    Status,
    get_workspace_folder,
    save_state,
)
from promptstrap.tools import create_tool_belt

system_prompt = """
            You are an experienced frontend software engineer, specialized in creating web applications with React, Tailwind CSS.
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


def persist_state_node(func):
    def wrapper(*args, **kwargs):
        state = func(*args, **kwargs)
        state.last_node = func.__name__
        save_state(state)
        return state

    return wrapper


@persist_state_node
def node_analyze_prompt(state: PromptstrapState) -> PromptstrapState:
    print("ANALYZE_PROMPT Enter")

    if len(state.files) > 0:
        # nothing to do, state already analyzed.
        return state

    llm = OpenAILLM(system_prompt=system_prompt)
    json_parser = JsonOutputParser(pydantic_object=PromptstrapState)
    prompt = PromptTemplate.from_template(
        """
        Analyze the following prompt: {input}.
        Return a JSON object with a plan on how to implement the prompt as a web application.
        The plan should include:
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

        Here are the schema instructions for the JSON object. You should strictly follow these instructions:
        {format_instructions}
        """,
        partial_variables={
            "format_instructions": json_parser.get_format_instructions()
        },
    )

    chain = prompt | llm | json_parser

    result = chain.invoke({"input": state.input})
    state.files = result.get("files", [])
    return state


@persist_state_node
def node_create_repo(state: PromptstrapState) -> PromptstrapState:
    print("REPO Enter")
    # TODO: Make this actuall create a repo. For now use an output folder
    folder_name = get_workspace_folder(state)

    os.makedirs(folder_name, exist_ok=True)
    print(f"Created folder (if it did not already exist): {folder_name}")
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


@persist_state_node
def node_files(state: PromptstrapState) -> PromptstrapState:
    print("FILES Enter")
    if state.files is None:
        return state

    if state.repo_state != FileState.GENERATED:
        raise ValueError("Repository must be created before generating files.")

    if state.repo_name is None:
        raise ValueError("Repository name must be set before creating files.")

    print(f"===== dep_result =====\n{state.dep_result}")
    print(f"===== build_result =====\n{state.build_result}")
    print(f"===== test_results =====\n{state.test_results}")

    llm = OpenAILLM(
        system_prompt=system_prompt, tools=create_tool_belt(state), temperature=0.6
    )
    prompt = PromptTemplate.from_template(
        f"""
        Look at the state structure below and decide which tool you need to invoke to generate the files or solve the issues.
        You can invoke 10 tools at a time in a single call.
        The file syntax should be correct and follow the conventions of the specified file type. You should also respect your own coding rules.

        Issues to be addressed: 
        Files that are not in the {FileState.GENERATED} state need to be updated or created.
        Address the following issues (if any): 
        - start of issues to be addressed -
        {state.dep_result}
        {state.build_result}
        {state.test_results}
        - end of issues to be addressed -
        If the file is in a format that you cannot generate, you should set the status accordingly and return an error message
        in the corresponding json field.
        This file is part of a web application project. Below is the context of the project and how it is built:
        {{state_structure}}
        """,
    )

    chain = prompt | llm

    result = chain.invoke(
        {
            "state_structure": state.model_dump_json(),
        }
    )

    return clear_error_status(state)


@persist_state_node
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
    folder_name = get_workspace_folder(state)

    result = subprocess.run(
        ["npm", "install"],
        cwd=folder_name,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        error_message += f"Error installing dependencies: \nSTDOUT:\n{result.stdout}\n STDERR:\n{result.stderr}\n"
        state.status = Status.ERROR
        state.dep_result = error_message.replace("{", "{{").replace("}", "}}")
        print("Npm install failed with errors:\n", error_message)

        return state

    else:
        print("npm install completed successfully.")

    return clear_error_status(state)


@persist_state_node
def node_build(state: PromptstrapState) -> PromptstrapState:
    print("BUILD Enter")

    if state.status != Status.SUCCESS:
        raise ValueError("Cannot build project with errors.")

    folder_name = get_workspace_folder(state)

    result = subprocess.run(
        ["npm", "run", "build"],
        cwd=folder_name,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        error_message = f"Error building project: \nSTDOUT:\n{result.stdout}\n STDERR:\n{result.stderr}\n"
        state.status = Status.ERROR
        state.build_result = error_message.replace("{", "{{").replace("}", "}}")
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
    options.set_capability("goog:loggingPrefs", {"browser": "ALL"})

    driver = webdriver.Chrome(options=options)
    print(f"getting |{vite_url}|")
    time.sleep(5)
    driver.get(vite_url.strip())
    time.sleep(5)
    png_bytes = driver.get_screenshot_as_png()
    browser_logs = driver.get_log("browser")
    driver.quit()
    screenshot_path = os.path.join(folder_name, "screenshot.jpg")
    image = Image.open(BytesIO(png_bytes)).convert("RGB")
    image.save(screenshot_path, "JPEG", quality=30)

    print("Selenium screenshot taken and saved.")

    return browser_logs


def screenshot_project(state: PromptstrapState) -> tuple[str, str, str, str]:
    folder_name = get_workspace_folder(state)

    process, vite_url = start_vite(folder_name)
    browser_logs = start_selenium(vite_url, folder_name)
    process.terminate()
    process.wait()

    stdout, stderr = process.communicate()

    print("Vite development server stopped.")

    return os.path.join(folder_name, "screenshot.jpg"), stdout, stderr, browser_logs


@persist_state_node
def node_test_project(state: PromptstrapState) -> PromptstrapState:
    print("TEST Enter")

    if state.status != Status.SUCCESS:
        raise ValueError("Cannot test project with errors.")
    llm = ChatOpenAI(model="gpt-4.1", temperature=0.0)
    state.functional_tests_cycles -= 1

    if state.functional_tests_cycles < 0:
        print("Skipping functional tests, no more cycles left.")
        return clear_error_status(state)

    class UpdateActionList(BaseModel):
        current_state: str
        instructions: List[str] = []

    path, stdout, stderr, browser_logs = screenshot_project(state)

    print(
        f"vite output: \nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}\n\nBrowser logs:\n{browser_logs}\n"
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
                    Don't give general advice, focus on the specific elements that need to be added or modified. If there are no major changes needed, return an empty list.
                    For instance, if there's an empty page, add instructions to fix that considering the logs below. If there are elements from the prompt that are missing from the page, instruct to add those.
                    Don't add instructions to check things, just focus on what is there and what should be there and the errors that need fixing.   
                    Desired state prompt:
                    {state.input}
                    ========
                    Here is the output from vite, maybe this helps understand the error:
                    STDOUT:
                    {stdout}

                    STDERR:
                    {stderr}

                    ========
                    Summarize instructions about fixing sever errors in the following browser logs:
                    {browser_logs}
                    ========
                    
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

    if len(result["instructions"]) > 0:
        pprint(result)
        state.test_results = result["instructions"]
        state.status = Status.ERROR
        return state

    return clear_error_status(state)


def clear_error_status(state: PromptstrapState) -> PromptstrapState:
    state.status = Status.SUCCESS
    state.dep_result = state.build_result = state.test_results = None
    return state


@persist_state_node
def node_apply_observations(state: PromptstrapState) -> PromptstrapState:
    print("APPLY_OBSERVATIONS Enter")

    llm = OpenAILLM(system_prompt=system_prompt)
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
# APPLY_OBSERVATIONS = "ApplyObservationsToStateFiles"


def NewGraph():

    graph = StateGraph(state_schema=PromptstrapState)
    graph.add_node(ANALYZE_PROMPT, node_analyze_prompt)
    graph.add_node(CREATE_REPO, node_create_repo)
    graph.add_node(CREATE_FILES, node_files)
    graph.add_node(INSTALL_DEP, node_install_dep)
    graph.add_node(BUILD_PROJECT, node_build)
    graph.add_node(TEST_PROJECT, node_test_project)
    #    graph.add_node(APPLY_OBSERVATIONS, node_apply_observations)

    graph.add_edge(ANALYZE_PROMPT, CREATE_REPO)
    graph.add_edge(CREATE_REPO, CREATE_FILES)
    #    graph.add_edge(APPLY_OBSERVATIONS, CREATE_FILES)

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
            Status.ERROR: CREATE_FILES,
        },
    )

    graph.add_conditional_edges(
        BUILD_PROJECT,
        lambda state: state.status,
        {Status.SUCCESS: TEST_PROJECT, Status.ERROR: CREATE_FILES},
    )

    graph.add_conditional_edges(
        TEST_PROJECT,
        lambda state: state.status,
        {
            Status.SUCCESS: END,
            Status.ERROR: CREATE_FILES,
        },
    )

    graph.set_entry_point(ANALYZE_PROMPT)

    compiled_graph = graph.compile()

    return compiled_graph
