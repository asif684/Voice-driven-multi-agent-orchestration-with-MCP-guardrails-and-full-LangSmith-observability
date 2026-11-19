"""The web-research specialist sub-agent."""
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from .. import config
from ..tools.research_tool import research_topic

SYSTEM_PROMPT = """You are the research specialist on a team of AI assistants.

Given a topic, use your web search tool to investigate it and produce a
concise, well-organized markdown summary: short headings, bullet points,
under 300 words. Search more than once if the first results are thin or
off-topic.

Return only the summary itself, ready to be handed to another assistant
(e.g. to be saved as a Notion page) - no preamble like "Here is a summary".
"""


def _get_chat_model():
    """Get the appropriate chat model instance based on provider config."""
    return ChatGoogleGenerativeAI(
        model=config.CHAT_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=0.3,
    )


async def build_researcher():
    """Assemble the research specialist agent."""
    chat_model = _get_chat_model()
    return create_agent(chat_model, tools=[research_topic], system_prompt=SYSTEM_PROMPT)