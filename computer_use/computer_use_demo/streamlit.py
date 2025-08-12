"""
Entrypoint for streamlit (Nebius/OpenAI Chat compatible, no tool-version UI).
"""

import asyncio
import base64
import os
import subprocess
import traceback
from contextlib import contextmanager
from datetime import datetime
from enum import StrEnum
from functools import partial
from pathlib import PosixPath
from typing import cast

import httpx
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from computer_use_demo.loop import sampling_loop
from computer_use_demo.tools import ToolResult

# ---------------------------- Config ----------------------------------

DEFAULT_MODEL = "Qwen/Qwen2.5-32B-Instruct"  # or "qwen2-vl-72b-instruct" if you prefer

CONFIG_DIR = PosixPath("~/.nebius_demo").expanduser()

STREAMLIT_STYLE = """
<style>
    /* Highlight the stop button in red */
    button[kind=header] {
        background-color: rgb(255, 75, 75);
        border: 1px solid rgb(255, 75, 75);
        color: rgb(255, 255, 255);
    }
    button[kind=header]:hover {
        background-color: rgb(255, 51, 51);
    }
    /* Hide the streamlit deploy button */
    .stAppDeployButton { visibility: hidden; }
</style>
"""

WARNING_TEXT = "⚠️ Security Alert: Never provide access to sensitive accounts or data, as malicious web content can hijack the agent’s behavior."

class Sender(StrEnum):
    USER = "user"
    BOT = "assistant"
    TOOL = "tool"

# ---------------------------- App State --------------------------------

def setup_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "api_key" not in st.session_state:
        st.session_state.api_key = os.getenv("NEBIUS_API_KEY", "")
    if "model" not in st.session_state:
        st.session_state.model = DEFAULT_MODEL
    if "auth_validated" not in st.session_state:
        st.session_state.auth_validated = False
    if "responses" not in st.session_state:
        st.session_state.responses = {}
    if "tools" not in st.session_state:
        st.session_state.tools = {}
    if "only_n_most_recent_images" not in st.session_state:
        st.session_state.only_n_most_recent_images = 3
    if "custom_system_prompt" not in st.session_state:
        st.session_state.custom_system_prompt = ""
    if "hide_images" not in st.session_state:
        st.session_state.hide_images = False
    if "in_sampling_loop" not in st.session_state:
        st.session_state.in_sampling_loop = False
    if "output_tokens" not in st.session_state:
        st.session_state.output_tokens = 4096  # default max tokens per reply

# --------------------------- Main UI -----------------------------------

async def main():
    setup_state()
    st.markdown(STREAMLIT_STYLE, unsafe_allow_html=True)
    st.title("Computer Use Demo (Nebius)")

    if not os.getenv("HIDE_WARNING", False):
        st.warning(WARNING_TEXT)

    with st.sidebar:
        st.text_input(
            "Nebius API Key",
            type="password",
            key="api_key",
        )
        st.text_input("Model", key="model")
        st.number_input(
            "Only send N most recent images",
            min_value=0,
            key="only_n_most_recent_images",
            help="To decrease tokens, remove older screenshots from the conversation.",
        )
        st.text_area(
            "Custom System Prompt Suffix",
            key="custom_system_prompt",
            help="Extra instructions appended to the base system prompt.",
        )
        st.checkbox("Hide screenshots", key="hide_images")
        st.number_input("Max Output Tokens", key="output_tokens", step=1)

        if st.button("Reset", type="primary"):
            with st.spinner("Resetting..."):
                st.session_state.clear()
                setup_state()
                subprocess.run("pkill Xvfb; pkill tint2", shell=True)  # noqa: ASYNC221
                await asyncio.sleep(1)
                subprocess.run("./start_all.sh", shell=True)  # noqa: ASYNC221

    # Auth
    if not st.session_state.api_key:
        st.warning("Enter your Nebius API key in the sidebar to continue.")
        return
    else:
        st.session_state.auth_validated = True

    chat, http_logs = st.tabs(["Chat", "HTTP Exchange Logs"])
    new_message = st.chat_input("Type a message to control the computer...")

    # --------- Chat history ----------
    with chat:
        for message in st.session_state.messages:
            _render_any_message(message)

        # render past http exchanges (your loop may or may not populate these)
        for identity, (request, response) in st.session_state.responses.items():
            _render_api_response(request, response, identity, http_logs)

        if new_message:
            st.session_state.messages.append({"role": "user", "content": new_message})
            _render_message(Sender.USER, new_message)

        try:
            most_recent_message = st.session_state["messages"][-1]
        except IndexError:
            return

        if most_recent_message["role"] != "user":
            # nothing to respond to
            return

        with track_sampling_loop():
            st.session_state.messages = await sampling_loop(
                system_prompt_suffix=st.session_state.custom_system_prompt,
                model=st.session_state.model,
                provider=None,  # your loop ignores this
                messages=st.session_state.messages,
                output_callback=partial(_render_message, Sender.BOT),
                tool_output_callback=partial(
                    _tool_output_callback, tool_state=st.session_state.tools
                ),
                api_response_callback=partial(
                    _api_response_callback,
                    tab=http_logs,
                    response_state=st.session_state.responses,
                ),
                api_key=st.session_state.api_key,
                only_n_most_recent_images=st.session_state.only_n_most_recent_images,
                max_tokens=st.session_state.output_tokens,
                thinking_budget=None,  # not used with Nebius Chat Completions
            )

# -------------------------- Helpers ------------------------------------

@contextmanager
def track_sampling_loop():
    st.session_state.in_sampling_loop = True
    yield
    st.session_state.in_sampling_loop = False

def _api_response_callback(
    request: httpx.Request,
    response: httpx.Response | object | None,
    error: Exception | None,
    tab: DeltaGenerator,
    response_state: dict[str, tuple[httpx.Request, httpx.Response | object | None]],
):
    response_id = datetime.now().isoformat()
    response_state[response_id] = (request, response)
    _render_api_response(request, response, response_id, tab)

def _tool_output_callback(
    tool_output: ToolResult, tool_id: str, tool_state: dict[str, ToolResult]
):
    tool_state[tool_id] = tool_output
    _render_message(Sender.TOOL, tool_output)

def _render_api_response(
    request: httpx.Request,
    response: httpx.Response | object | None,
    response_id: str,
    tab: DeltaGenerator,
):
    with tab:
        with st.expander(f"Request/Response ({response_id})"):
            newline = "\n\n"
            st.markdown(
                f"`{request.method} {request.url}`{newline}{newline.join(f'`{k}: {v}`' for k, v in request.headers.items())}"
            )
            try:
                st.json(request.read().decode())
            except Exception:
                st.write(request.read())
            st.markdown("---")
            if isinstance(response, httpx.Response):
                st.markdown(
                    f"`{response.status_code}`{newline}{newline.join(f'`{k}: {v}`' for k, v in response.headers.items())}"
                )
                try:
                    st.json(response.text)
                except Exception:
                    st.write(response.text)
            else:
                st.write(response)

def _render_any_message(message: dict):
    """
    Render a stored chat message in st.session_state.messages (OpenAI/Nebius format).
    """
    role = message.get("role")
    content = message.get("content")

    if role == "tool":
        # Show the actual ToolResult we stored in tool_state (if available)
        tool_call_id = message.get("tool_call_id")
        tr = st.session_state.tools.get(tool_call_id)
        if tr:
            _render_message(Sender.TOOL, tr)
        else:
            _render_message(Sender.TOOL, content or "")
        return

    # Assistant / user
    if isinstance(content, str):
        _render_message(Sender.BOT if role == "assistant" else Sender.USER, content)
    elif isinstance(content, list):
        # multimodal: text / image_url parts
        with st.chat_message(Sender.BOT if role == "assistant" else Sender.USER):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    st.write(part.get("text", ""))
                elif part.get("type") == "image_url" and not st.session_state.hide_images:
                    url = (part.get("image_url") or {}).get("url")
                    if url and url.startswith("data:"):
                        # render embedded data URL
                        try:
                            header, b64 = url.split(",", 1)
                            st.image(base64.b64decode(b64))
                        except Exception:
                            st.write("[invalid image data]")
                    else:
                        st.image(url)
                else:
                    st.write(str(part))
    else:
        _render_message(Sender.BOT if role == "assistant" else Sender.USER, str(content))

def _render_message(sender: Sender, message: str | ToolResult):
    """Render a simple text or a ToolResult."""
    with st.chat_message(sender):
        if isinstance(message, ToolResult):
            if message.output:
                st.markdown(message.output)
            if message.error:
                st.error(message.error)
            if message.base64_image and not st.session_state.hide_images:
                st.image(base64.b64decode(message.base64_image))
        else:
            st.markdown(message)

# ----------------------------- Run -------------------------------------

if __name__ == "__main__":
    asyncio.run(main())