from promptstrap.llm import MixtralLLM, OpenAILLM

from promptstrap.graph import NewGraph
from promptstrap.state import PromptstrapState
from promptstrap.graph import node_analyze_prompt, node_create_repo, node_files

from pprint import pprint


llm = OpenAILLM()


def test_nodes():
    compiled_graph = NewGraph()

    input_dict = {
        "input": (
            """Design a web application that allows users to interact with an llm by inputting text prompts and receiving messages.
            Use a color palette based on Bright Blue, Dark Teal, Aqua Orange.
            The application sohuld be mobile-friendly and responsive.
            The header will display a futuristic image of a city skyline at night with neon lights.
            The footer will contain links to the terms of service, privacy policy, and contact information.
            """
        )
    }

    result = compiled_graph.invoke(input_dict)

    pprint(result)


input_dict = {
    "input": (
        """
Build a React-based user interface that mimics the layout, styling, and interaction patterns of a client account dashboard for a telecom or internet provider. The content should be fully mocked using placeholder text (e.g., lorem ipsum), but the layout and structure should follow the guidelines below.

Color Scheme and Aesthetic:

Use a clean white background (#FFFFFF).

Accent color should be bright orange (#FF7900), used for the logo, icons, and action arrows.

Primary text color should be black or dark gray (#000000 or #333333).

Buttons should have a black background with white text.

Use a modern sans-serif font throughout.

Overall Layout:

Top header bar (dark background):

Left side: company logo (use a placeholder image or text).

Center: search bar.

Right side: notification icon, mail icon, app grid icon, and user avatar/profile dropdown.

Navigation bar below the header (horizontal):

Add several category links such as "Category A", "Category B", "Category C" (use placeholder names).

Left sidebar (vertical navigation menu):

Header text for user plan (mocked string).

An expandable section titled "Main Section" with sub-items:

Subsection 1

Subsection 2

Subsection 3

Additional menu items below, e.g.:

Personal Info

Manage Profiles

Contractual Documents (this should be the active/selected item)

Order Tracking

Contact & Requests

Main Content Area:

Title at the top: use a placeholder for the section title.

Display a top banner with:

A header ("Section Banner Title" - mock this)

One or two lines of placeholder description text (e.g., lorem ipsum).

A call-to-action button with high contrast (black background, white text) labeled with mock text like "Action".

An image or placeholder illustration on the right side (representing document stacks or folders).

Below the banner, create a two-column layout with 3–4 clickable items per column. Each item should contain:

A mocked title (e.g., lorem ipsum text)

A navigation arrow using the orange accent color

On hover: slightly shaded background to indicate interactivity

Chatbox:

Add a floating chat/help button in the lower-right corner.

Use a circular icon (mocked with a person silhouette).

Add a tooltip or caption like "Need help? Ask me!" (mock this in lorem ipsum).

Start minimized, but should be expandable on click.

Additional Notes:

All text content must be mocked.

Make the layout responsive for desktop use.

Break UI into modular components (e.g., sidebar, header, banner, content grid).

Tailwind CSS is preferred for styling.

Add basic accessibility support (e.g., ARIA labels where needed).

Use clear component naming to reflect the UI structure.
        """
    )
}


def test_screenshot():
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    from langchain_core.messages import SystemMessage
    from langchain_core.output_parsers import JsonOutputParser
    from pydantic import BaseModel
    from typing import List
    import base64

    from promptstrap.state import PromptstrapState
    from promptstrap.graph import screenshot_project

    llm = ChatOpenAI(model="gpt-4.1", temperature=0.5)

    class UpdateActionList(BaseModel):
        current_state: str
        instructions: List[str] = []

    path = screenshot_project(
        PromptstrapState(
            input=input_dict["input"], repo_name="telecom-client-dashboard"
        )
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
                    {input_dict['input']}
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
    pprint(result)


def test_agent():
    graph = NewGraph()

    with open("graph.png", "wb") as f:
        f.write(graph.get_graph().draw_png())

    result = graph.invoke(input_dict, {"recursion_limit": 100})

    pprint(result)
