from enum import Enum

from pydantic import BaseModel
from typing import Optional


class Component(BaseModel):
    name: str
    functionality: list[str]
    parent_component: str | None
    source: str | None


class Style(BaseModel):
    theme: str
    font: str
    colorPalette: list[str]


class Status(str, Enum):
    SUCCESS = "success"
    ERROR = "error"


class PromptstrapState(BaseModel):
    input: str
    project_name: Optional[str] = None
    project_description: Optional[str] = None
    repo_name: Optional[str] = None
    repo_ssh_url: Optional[str] = None
    components: Optional[list[Component]] = None
    style: Optional[Style] = None
    behaviors: Optional[list[str]] = None
    dependencies: Optional[list[str]] = None
    status: Optional[Status] = Status.SUCCESS
