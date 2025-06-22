from pprint import pprint
from promptstrap.graph import NewGraph
import argparse


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a prompt file path.")
    parser.add_argument(
        "--prompt-path", type=str, required=True, help="Path to the input prompt file"
    )
    args = parser.parse_args()

    if args.prompt_path:
        with open(args.prompt_path, "r") as prompt_file:
            input_dict["input"] = prompt_file.read()
    print("Prompt input:", input_dict["input"])
    graph = NewGraph()

    with open("graph.png", "wb") as f:
        f.write(graph.get_graph().draw_png())

    result = graph.invoke(input_dict, {"recursion_limit": 100})

    pprint(result)
