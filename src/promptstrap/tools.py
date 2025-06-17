import base64
import os
from typing import Optional

import dotenv
import openai
import requests
from github import Github
from pydantic import BaseModel

from promptstrap.state import FileType

dotenv.load_dotenv()


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


def util_create_image(type: FileType, prompt: str, file_path: str):
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
        with open(file_path, "wb") as f:
            f.write(image_response.content)
