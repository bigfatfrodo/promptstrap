import base64
import os
from typing import Optional
from langchain.tools import tool
import openai
import requests
from github import Github
from pydantic import BaseModel

from promptstrap.state import File, FileType, PromptstrapState, FileState



class RepoCreationResult(BaseModel):
    status: str
    repo_name: Optional[str]
    ssh_url: Optional[str]
    description: str
    error: Optional[str] = None


class RepoCreationInput(BaseModel):
    repo_name: str
    description: str


def create_private_repo(input: RepoCreationInput) -> RepoCreationResult:

    token = os.getenv("GITHUB_TOKEN")

    try:
        g = Github(token)
        user = g.get_user()
        repo = user.create_repo(
            name=input.repo_name,
            description=input.description,
            private=True,
        )
        return RepoCreationResult(
            status="success",
            repo_name=repo.name,
            ssh_url=repo.ssh_url,
            description=repo.description or "",
            error=None,
        )

    except Exception as e:
        return RepoCreationResult(
            status="error",
            repo_name=input.repo_name,
            ssh_url=None,
            description=input.description,
            error=str(e),
        )


def get_user_repos() -> list[str]:
    token = os.getenv("GITHUB_TOKEN")
    g = Github(token)
    user = g.get_user()
    return [repo.name for repo in user.get_repos()]


def create_create_image_file(state: PromptstrapState):

    @tool
    def util_create_image(type: FileType, prompt: str, file_path: str):
        """Create an image file at a relative path string from the top level of the project, using DALL-E 3 with the given prompt."""
        print("[tool] Creating image file:", file_path)
        full_path = os.path.join(state.output_folder, state.repo_name, file_path)
        if not check_path_within_project(file_path, state):
            return (
                f"Invalid file path: {file_path} is outside the repository directory."
            )
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        client = openai.Client()
        if type == FileType.JPEG_WIDE:
            size = "1792x1024"
        elif type == FileType.JPEG_TALL:
            size = "1024x1792"
        elif type == FileType.JPEG_SQUARE:
            size = "1024x1024"

        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
        )

        url = response.data[0].url
        image_response = requests.get(url)
        if image_response.status_code == 200:
            with open(full_path, "wb") as f:
                f.write(image_response.content)
        for file in state.files:
            if file.path == file_path:
                file.state = FileState.GENERATED
                return f"Generated {file_path}"

    return util_create_image


def check_path_within_project(path: str, state: PromptstrapState) -> bool:
    """Check if the given path is within the project directory."""
    full_path = os.path.join(state.output_folder, state.repo_name, path)
    return os.path.abspath(full_path).startswith(
        os.path.abspath(os.path.join(state.output_folder, state.repo_name))
    )


def create_create_file_tool(state: PromptstrapState):
    """Create a tool to create a file in the project directory."""

    @tool
    def create_or_update_file(file_path: str, content: str):
        """Create or update a text file at a relative path string from the top level of the project, with the given content."""
        print("[tool] Creating file:", file_path)
        full_path = os.path.join(state.output_folder, state.repo_name, file_path)
        if not check_path_within_project(file_path, state):
            return (
                f"Invalid file path: {full_path} is outside the repository directory."
            )

        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)

        for file in state.files:
            if file.path == file_path:
                file.content = content
                file.state = FileState.GENERATED
                return f"Wrote {file_path}."
        return f"File created at {file_path}"

    return create_or_update_file


def create_move_file_tool(state: PromptstrapState):
    """Create a tool to move a file in the project directory."""

    @tool
    def move_file(source_path: str, destination_path: str):
        """Move a file from source_path to destination_path."""
        print("[tool] Moving file from", source_path, "to", destination_path)
        full_source_path = os.path.join(
            state.output_folder, state.repo_name, source_path
        )
        full_destination_path = os.path.join(
            state.output_folder, state.repo_name, destination_path
        )
        if not check_path_within_project(full_source_path, state):
            return f"Invalid file path: {full_source_path} is outside the repository directory."
        if not check_path_within_project(full_destination_path, state):
            return f"Invalid file path: {full_destination_path} is outside the repository directory."

        os.makedirs(os.path.dirname(full_destination_path), exist_ok=True)
        os.rename(full_source_path, full_destination_path)

        for file in state.files:
            if file.path == source_path:
                file.path = destination_path
                break
        return f"File moved from {source_path} to {destination_path}"

    return move_file


def create_delete_file_tool(state: PromptstrapState):
    """Create a tool to delete a file in the project directory."""

    @tool
    def delete_file(file_path: str):
        """Delete a file at a relative path string from the top level of the project."""
        print("[tool] Deleting file:", file_path)
        full_path = os.path.join(state.repo_name, file_path)
        if not check_path_within_project(file_path, state):
            return (
                f"Invalid file path: {file_path} is outside the repository directory."
            )
        if os.path.exists(full_path):
            os.remove(full_path)
            state.files = [file for file in state.files if file.path != file_path]
            return f"File deleted at {full_path}"
        else:
            return f"File not found at {file_path}"

    return delete_file


def create_tool_belt(state: PromptstrapState):
    """Create a tool belt with all tools needed for the project."""

    return [
        create_create_image_file(state),
        create_create_file_tool(state),
        create_move_file_tool(state),
        create_delete_file_tool(state),
    ]
