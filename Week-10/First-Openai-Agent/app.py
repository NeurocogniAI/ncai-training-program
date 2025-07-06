import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

# Load the environment variables from the .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

agent = Agent(name="Assistant",
               instructions="You are a helpful assistant",
            #    model=model
                 )


def main():
    print("Hello from first-openai-agent!")
    result = Runner.run_sync(
    starting_agent=agent,
    input="Write program that sum two numbers and numbers are 10 and 5.",
    run_config=config)

    print(result.final_output)


if __name__ == "__main__":
    main()
