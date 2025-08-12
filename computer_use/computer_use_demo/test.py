# debug_driver.py
import asyncio, logging
from computer_use_demo.loop import sampling_loop

import os










async def main():
    msgs = [{
        "role": "user",
        "content": "Click at 0 0 ",
    }]


    msgs = await sampling_loop( 
    model="Qwen/Qwen2.5-32B-Instruct",   
    messages=msgs,
    api_key=os.getenv("NEBIUS_API_KEY"),
    max_tokens=4096,
)
    print(len(msgs))
    await sampling_loop( 
    model="Qwen/Qwen2.5-32B-Instruct",   
    messages=msgs,
    api_key=os.getenv("NEBIUS_API_KEY"),
    max_tokens=4096,
)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main())