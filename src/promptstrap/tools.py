import os
from typing import Optional

import dotenv
from github import Github
from pydantic import BaseModel

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


class AnalyzePromptResult(BaseModel):
    repo_name: str
