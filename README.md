# Promptstrap - yet another simple agent for simple interfaces

Promptstrap is an agent that generates simple web apps using React and Tailwind. The reason for its existence is that I don't know React, Tailwind - or any other frontend tech for that matter - and needed a way to create user interfaces for other projects I'm building. 
The agent is implemented using LangChain and LangGraph. Example usage can be found in the notebook, `tests.ipynb`.

The state graph behind it:

![image](https://raw.githubusercontent.com/bigfatfrodo/promptstrap/refs/heads/main/graph.png)


Here's an example UI generated from the prompt in the notebook. Source code for that is included in the `neon-chat.tar.gz` archive.

```
input_dict = {
    "input": (
        """Design a web application that allows users to interact with an llm by inputting text prompts and receiving messages.
        Use a color palette based on Bright Blue, Dark Teal, Aqua Orange. These colors should be used for the background, text, and buttons.
        The chat components are an input field for user prompt, a submit button and a display area for the llm response. In the display area,
        the llm response should be shown in a chat bubble aligned to the left, and the user prompt should be shown in a chat bubble aligned to the right.
        Also, the chat components should be bounded by a thin round-corner border with a shadow effect. The button should be to the left of the input field.
        The display area should be above the input field and button.
        The LLM responses should be mocked with random sequences from the lorem ipsum text.
        The application should be mobile-friendly and responsive.
        The header will display a futuristic image of a city skyline at night with neon lights.
        The footer will contain links to the terms of service, privacy policy, and contact information. Also generate stubs for these pages.
        """
    )
}
```

<img width="1708" alt="image" src="https://github.com/user-attachments/assets/e66ea093-9bcd-4b0d-a637-7adda2d80a3e" />

You need to set up the project with `poetry`. 
```
% poetry install
...
% $(poetry env activate)
```

Entries needed in a file in top level `.env` file:
```
PYTHONPATH=src
OPENAI_API_KEY=sk-proj-UD......
```
How to run `promptstrap`:
```
% poetry run python -m promptstrap.promptstrap
```

To run the generated example in the `neon-chat` archive just untar it and run the following in the `neon-chat` folder:
```
% npm install

added 180 packages, and audited 181 packages in 1s

37 packages are looking for funding
  run `npm fund` for details

2 moderate severity vulnerabilities

To address all issues (including breaking changes), run:
  npm audit fix --force

Run `npm audit` for details.
```
```
 % npm run build

> neon-chat@0.1.0 build
> vite build

vite v5.4.19 building for production...
✓ 50 modules transformed.
dist/index.html                        0.49 kB │ gzip:  0.32 kB
dist/assets/skyline-C2RzQJvF.jpeg  2,874.15 kB
dist/assets/index-BI1JJAu_.css        12.05 kB │ gzip:  3.12 kB
dist/assets/index-CaKHZXJf.js        216.26 kB │ gzip: 70.34 kB
✓ built in 678ms
```
```
% npm run dev

  VITE v5.4.19  ready in 95 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
  ➜  press h + enter to show help
```


