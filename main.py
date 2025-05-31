from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import os

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

#Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
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

# Urdu Translator Agent
urdu_translator_agent = Agent(
    name = 'Urdu Translator Agent',
    instructions= 
    """You are a Urdu Translator agent.
    Translate only Urdu language into English language.
    Just translate Urdu into English;
    otherwise, provide a message like that
    This is not the Urdu language,
    it can't be translated into English. Enter only Urdu."""
)

response = Runner.run_sync(
    urdu_translator_agent,
    input = input("Type here: "),
    run_config = config
    )
print(response.final_output)