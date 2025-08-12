"""
Agentic sampling loop that calls the Anthropic API and local implementation of anthropic-defined computer use tools.
"""
import os
import platform
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Any, cast

import httpx
from anthropic import (
    Anthropic,
    AnthropicBedrock,
    AnthropicVertex,
    APIError,
    APIResponseValidationError,
    APIStatusError,
)
from anthropic.types.beta import (
    BetaCacheControlEphemeralParam,
    BetaContentBlockParam,
    BetaImageBlockParam,
    BetaMessage,
    BetaMessageParam,
    BetaTextBlock,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
    BetaToolUseBlockParam,
)

from .tools import (
    TOOL_GROUPS_BY_VERSION,
    ToolCollection,
    ToolResult,
    ToolVersion,
)

PROMPT_CACHING_BETA_FLAG = "prompt-caching-2024-07-31"


class APIProvider(StrEnum):
    NEBIUS = "nebius"

from openai import OpenAI
import json


# This system prompt is optimized for the Docker environment in this repository and
# specific tool combinations enabled.
# We encourage modifying this system prompt to ensure the model has context for the
# environment it is running in, and to provide any additional information that may be
# helpful for the task at hand.
SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
* You are utilising an Ubuntu virtual machine using {platform.machine()} architecture with internet access.
* You can only take screenshots, move a mouse and do a left click. 
* If you are not sure of location of something, ALWAYS take screenshot to verify it. 
* You can assume that the size of a screen is width = {int(os.getenv("WIDTH"))} , height = {int(os.getenv("HEIGHT"))}.
"""

vision_model = "Qwen/Qwen2-VL-72B-Instruct"


async def sampling_loop(
    *,
    model: str,
    provider = None,
    system_prompt_suffix = None,
    messages,
    output_callback = None,
    tool_output_callback = None,
    api_response_callback = None,
    api_key: str,
    only_n_most_recent_images: int | None = None,
    max_tokens: int = 4096,
    thinking_budget: int = 256,
    tool_version: ToolVersion = 'computer_use_20250124',
):
    """
    Agentic sampling loop for the assistant/tool interaction of computer use.
    """


    tool_group = TOOL_GROUPS_BY_VERSION[tool_version]
    tool_collection = ToolCollection(*(ToolCls() for ToolCls in tool_group.tools))

    tool = {
    "type": "function",
    "function": {
        "name": "computer",
        "description": (
            "Simulated screen/mouse control. Can ONLY do screenshots, move a mouse, and make a left clicks. In action field specify which action from `mouse_move` `left_click` `screenshot` you are doing. The name must ALWAYS be computer\n"
            "Rules:\n"
            "- `mouse_move` REQUIRE `coordinate: [x, y]` (JSON array of two ints).\n"
            "- `left_click` MUST NOT include `coordinate "
            "- `screenshot` MUST NOT include `coordinate`,.\n"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "mouse_move","left_click","screenshot"
                    ]
                },
                "coordinate": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2, "maxItems": 2,
                    "description": "Exactly [x, y] in screen pixels."
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        }
    }
}
    system = SYSTEM_PROMPT

    while True:
        client = client = OpenAI(
                # do **not** include /v1 twice – the SDK appends /chat/completions itself
                base_url="https://api.studio.nebius.ai/v1/",
                api_key=api_key,                                
            )




        # Call the API
        # we use raw_response to provide debug information to streamlit. Your
        # implementation may be able call the SDK directly with:
        # `response = client.messages.create(...)` instead.
        raw_response = client.chat.completions.with_raw_response.create(
            model=model,
            messages=[{"role": "system", "content": system}, *messages],
            tool_choice="auto",
            tools=[tool], 
            max_tokens=max_tokens
        )
            

        api_response_callback(
            raw_response.http_response.request, raw_response.http_response, None
        )



        
        response = raw_response.parse()
        
        response = response.choices[0].message

        output_callback(response.content or "")
 
        messages.append(
            {
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": response.tool_calls or [],
            }
        )

        tool_result_content = []
        # inside your loop, instead of calling the real screenshot tool:
        

        for tool_call in response.tool_calls:
            
            result = await tool_collection.run(
                name=tool_call.function.name,
                tool_input= json.loads(tool_call.function.arguments),
            )
            tool_result_content.append(
                _make_api_tool_result(client=client,result=result, tool_call_id=tool_call.id, context=messages)
            )


            tool_output_callback(result, tool_call.id)


        if not tool_result_content:
            return messages
        
        messages.extend(tool_result_content)
        
        
        








def _make_api_tool_result(result: ToolResult, tool_call_id: str, client, context) -> dict:
    """
    Turn ToolResult into an OpenAI Chat Completions 'tool' message.
    content MUST be a string; we JSON-encode the payload.
    """
    payload = {}
    
    if getattr(result, "error", None):
        payload["error"] = result.error
    if getattr(result, "base64_image", None):
        payload["image"] = {
            "type": "base64",
            "media_type": "image/png",
            "data": summarize_image_with_vision(client, vision_model, result.base64_image, context),
        }
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps(payload, ensure_ascii=False),
    }

def summarize_image_with_vision(client, vision_model: str, b64_png: str, context) -> str:
    """Return a short textual description for a PNG base64 image (no data: prefix)."""
    try:
        system = {"role": "system",
                 "content": f"""You describe screenshots for a desktop-control agent. You are given context of a conversation, and the last message is an image that you have to describe. 
                 You can assume that the size of a screen is width = {int(os.getenv("WIDTH"))} , height = {int(os.getenv("HEIGHT"))}.
                 Make sure to always give EXACT coordinates of what are you seeing."""
                 }
        prompt = {"role": "user", "content": 
                    [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_png}"}}]
                }
        resp = client.chat.completions.with_raw_response.create(
            model=vision_model, 
            messages=[system]+context+[prompt],
            max_tokens=120,
        )
        resp = resp.parse()
        return resp.choices[0].message.content or ""
    except Exception as e:
        return f"[vision summary failed: {e}]"


import base64, mimetypes, os

def read_image_b64(path: str):
    mt, _ = mimetypes.guess_type(path)
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), (mt or "image/jpeg")