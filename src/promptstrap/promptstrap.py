from pprint import pprint
from promptstrap.graph import NewGraph
from promptstrap.state import PromptstrapState, sync_state
import argparse
import dotenv
import shortuuid

dotenv.load_dotenv()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a prompt file path.")
    parser.add_argument(
        "--prompt-path", type=str, required=True, help="Path to the input prompt file"
    )
    parser.add_argument(
        "--session-id",
        type=str,
        required=False,
        help="Continue an existing session",
        default="",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        required=True,
        help="Name of the project",
    )

    args = parser.parse_args()

    if args.prompt_path:
        with open(args.prompt_path, "r") as prompt_file:
            input_prompt = prompt_file.read()
    print("Prompt input:", input_prompt)
    graph = NewGraph()

    with open("graph.png", "wb") as f:
        f.write(graph.get_graph().draw_png())

    session_id = args.session_id if args.session_id != "" else shortuuid.uuid()
    print("Session ID:", session_id)
    state = sync_state(session_id, args.project_name, input_prompt)
    result = graph.invoke(
        state.model_dump(),
        {"recursion_limit": 100},
    )

    pprint(result)
