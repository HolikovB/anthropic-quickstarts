# Nebius Computer Use Demo

## Usage guidance 

> [!TIP]
> Make sure that you have NEBIUS_API_KEY environmental variable set to your nebius api key, and that you have docker.


```bash
docker build -t my-computer-use-demo:latest .
docker run -d \                              
    -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
    -e NEBIUS_API_KEY=$NEBIUS_API_KEY \
    -v $HOME/.anthropic:/home/computeruse/.anthropic \
    -p 5900:5900 \
    -p 8501:8501 \
    -p 6080:6080 \
    -p 8080:8080 \
    -it my-computer-use-demo:latest
```



Once the container is running, go to localhost:8800 on your browser to see it.

## Adaptation procces

Here are the main challenges that I encountered.

1) The OpenAI doesn't have built in tools set for an arbitrary model as Anthropic does (from what I read in documentation and heard from ChatGPT, it has it only for it's model that is not available on Nebius AI studio)

2) All Nebius AI studio vision models are not supporting auto choice of tools (as I understood)

3) Some models perform very bad in this setup.

4) Difference in formats of input/output for streamlit.


Here is how I approached them:

1) I just reduced number of tools that my agent is using to minimum, and implemented them by myself. Namely, I only left `screenshot`,`mouse_move`,`left_click`, since they are already producing most functional.

2) I used some trick, I added additional vision model that would be "eyes" of a main text model that supports `tool_usage="auto`, and it is just describing each picture to main model.

3) Firsty I though that I did something wrong, but then I switched from llama to Qwen, and it worked better (even though it still working not ideal at all).

4) It was the easiest, I just deleted all unnecessary stuff from `streamlit.py`, and made sure that everything works (actually chatGPT helped the most in this part).


## Evaluation
For a real word AI agent, I think that the best metric is how happy are users that are using it (because at the end of the day it's purpose is to make money for a company, and this is metric that affects it the most).
For a more technical metric, I would name how optimized is it (that is how cheap is each request), because again, this is what is important when we are talking about real word agents. 
