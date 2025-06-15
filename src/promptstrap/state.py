from enum import Enum

from pydantic import BaseModel
from typing import Optional


class FileType(str, Enum):
    JS = "js"
    TS = "ts"
    JSX = "jsx"
    TSX = "tsx"
    CSS = "css"
    HTML = "html"
    JSON = "json"
    MD = "md"
    TXT = "txt"
    JPEG = "jpeg"


class FileState(str, Enum):
    PLANNED = "planned"
    GENERATED = "generated"
    ERROR = "error"


class File(BaseModel):
    path: str
    type: FileType
    prompt: str
    state: FileState = FileState.PLANNED


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
    repo_state: FileState = FileState.PLANNED
    files: Optional[list[File]] = None
    style: Optional[Style] = None
    behaviors: Optional[list[str]] = None
    status: Optional[Status] = Status.SUCCESS
    output_folder: str = "agent_output"
