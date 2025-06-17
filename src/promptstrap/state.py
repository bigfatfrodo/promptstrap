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
    JPEG_WIDE = "jpeg_landscape"
    JPEG_TALL = "jpeg_portrait  "
    JPEG_SQUARE = "jpeg_square"


class FileState(str, Enum):
    PLANNED = "planned"
    NEEDS_UPDATE = "needs_update"
    NEEDS_SYNC = "needs_sync"
    GENERATED = "generated"
    ERROR = "error"


class File(BaseModel):
    path: str
    type: FileType
    prompt: str
    state: FileState = FileState.PLANNED
    content: str = ""


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
    repo_state: FileState = FileState.PLANNED
    files: Optional[list[File]] = []
    style: Optional[Style] = None
    behaviors: Optional[list[str]] = None
    status: Optional[Status] = Status.SUCCESS
    error_message: Optional[str] = None
    output_folder: str = "agent_output"
