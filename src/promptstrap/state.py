import base64
import json
import os
from enum import Enum
from typing import List, Optional

import dotenv
from pydantic import BaseModel
from sqlalchemy import Column, DateTime, Integer, String, create_engine
from sqlalchemy.dialects.mysql import MEDIUMTEXT
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func


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
    JPEG_TALL = "jpeg_portrait"
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


DEFAULT_TEST_RUNS = 4


class PromptstrapState(BaseModel):
    input: str
    session_id: str
    project_name: str

    last_node: str = ""

    repo_name: Optional[str] = None
    repo_state: FileState = FileState.PLANNED
    files: Optional[list[File]] = []
    style: Optional[Style] = None
    status: Optional[Status] = Status.SUCCESS
    error_message: Optional[str] = None
    output_folder: str = "agent_output"
    dep_result: Optional[str] = None
    build_result: Optional[str] = None
    test_results: Optional[List[str]] = None
    functional_tests_cycles: int = DEFAULT_TEST_RUNS

    def __init__(self, **data):
        super().__init__(**data)
        self.repo_name = f"{self.project_name}_{self.session_id}"


dotenv.load_dotenv()
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "root")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "promptstrap")


Base = declarative_base()


class PrompstrapStateRow(Base):
    __tablename__ = "promptstrap_state"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(60), index=True)
    timestamp = Column(DateTime, default=func.now(), index=True)
    last_node = Column(String(60), index=True)
    state = Column(MEDIUMTEXT, nullable=False)

    def to_state(self):
        state_str = base64.b64decode(str(self.state)).decode()

        return PromptstrapState(**json.loads(state_str))

    @staticmethod
    def from_state(state: PromptstrapState):

        state_str = state.model_dump_json()
        # encode state_str to a base64 string
        state_b64 = base64.b64encode(state_str.encode()).decode()
        return PrompstrapStateRow(
            session_id=state.session_id, state=state_b64, last_node=state.last_node
        )


engine = create_engine(f"mysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)


def save_state(state: PromptstrapState):
    db = Session()
    db.add(PrompstrapStateRow.from_state(state))
    db.commit()
    db.close()


def load_latest_state(session_id: str) -> PromptstrapState | None:
    db = Session()

    row = (
        db.query(PrompstrapStateRow)
        .filter_by(session_id=session_id)
        .order_by(PrompstrapStateRow.timestamp.desc())
        .first()
    )
    db.close()

    return row.to_state() if row else None


def get_workspace_folder(state: PromptstrapState) -> str:
    return os.path.join(state.output_folder, state.repo_name)


def sync_state(session_id: str, project_name: str, input: str) -> PromptstrapState:
    state = load_latest_state(session_id)

    # there is a stored state for
    if state is None:
        print(f"{session_id}: No stored state found")
        state = PromptstrapState(
            session_id=session_id,
            project_name=project_name,
            input=input,
        )
        folder_name = get_workspace_folder(state)
        if os.path.exists(folder_name):
            # no previous state, if a file/folder with the same name
            # exists at the expected location, remove it now
            if os.path.isdir(folder_name):
                import shutil

                shutil.rmtree(folder_name)
            else:
                os.remove(folder_name)

        return state
    else:
        print(f"{session_id}: Previous stored state exits")
        folder_name = get_workspace_folder(state)
        state.functional_tests_cycles = DEFAULT_TEST_RUNS
    # else there is a stored state, synchronize with folder content

    if not os.path.exists(folder_name):
        # no folder exists, just set all files as planned
        print(f"{session_id}: No folder exists, setting all files to planned")
        for file in state.files:
            file.state = FileState.PLANNED
        return state

    # the folder exists, check through the files in the state and
    # if they exist in the folder, then update the state with their contents
    # if they do not, get contents from state.
    # special case, if the file is an image
    # - if it exits, set the state to generated
    # - if it does not, set the state to planned

    for file in state.files:
        file_path = os.path.join(folder_name, file.path)

        # if the file was not previously generated, go on
        # it will be generated in CreateFiles
        if file.state != FileState.GENERATED:
            print(f"{session_id}: {file.path} was not generated, skipping")
            continue

        if not os.path.exists(file_path):
            if file.type in [
                FileType.JPEG_SQUARE,
                FileType.JPEG_WIDE,
                FileType.JPEG_TALL,
            ]:
                print(
                    f"{session_id}: {file.path} is an image and does not exist, setting to {FileState.PLANNED}"
                )
                file.state = FileState.PLANNED
                continue

            # if not an image file, take contents from state
            with open(file_path, "w") as f:
                print(
                    f"{session_id}: {file.path} does not exist, writing contents from state"
                )
                f.write(file.content)
        else:
            # if the file exists, read the contents, if it's not an image
            if file.type in [
                FileType.JPEG_SQUARE,
                FileType.JPEG_WIDE,
                FileType.JPEG_TALL,
            ]:
                print(f"{session_id}: {file.path} is an image, using content on disk")
                continue

            with open(file_path, "r") as f:
                print(f"{session_id}: {file.path} exists, reading contents from disk")
                file.content = f.read()

    return state
