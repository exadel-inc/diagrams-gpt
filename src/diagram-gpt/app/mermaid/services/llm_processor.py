import re
import os
import logging
from langchain.schema.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrock
from app.mermaid.models.mermaid_diagram_type import MermaidDiagramType
from prompts.prompt_provider import PromptProvider


class LlmProcessor:
    def __init__(self, prompt_provider: PromptProvider):
        self.prompt_provider = prompt_provider
        self.regex = re.compile(r"```(?:mermaid)?(.*)```", re.DOTALL)
        self.logger = logging.getLogger('uvicorn.error')
        self.logger.setLevel(logging.INFO)

    def prompt_generate(self, diagram_type: MermaidDiagramType, text: list[str]) -> str:
        messages = self._get_generate_messages(diagram_type, text)
        return self._messages_to_text(messages)
    
    def prompt_update(self, diagram_type: MermaidDiagramType, text: list[str], diagram_code: str) -> str: 
        messages = self._get_generate_messages(diagram_type, text[:-1])
        messages.append(AIMessage(content=diagram_code))
        messages.append(HumanMessage(content=text[-1]))

        return self._messages_to_text(messages)

    async def agenerate(
        self, diagram_type: MermaidDiagramType, text: list[str], openai_api_key: str
    ) -> str | None:
        """
        Process the given text and return the response.

        Args:
            text (str): The text to be processed.
            openai_api_key (str): The OpenAI API key.

        Returns:
            str or None: The processed response, or None if no response is available.
        """

        messages = self._get_generate_messages(diagram_type, text)
        result = await self._agenerate(messages, openai_api_key)

        if result is None or result == "NO_DIAGRAM":
            return None

        return self._match_diagram(result)

    async def aupdate(
        self,
        diagram_type: MermaidDiagramType,
        text: list[str],
        diagram_code: str,
        openai_api_key: str,
    ) -> str | None:
        """
        Process the given diagram code, apply updates and return the response.

        Args:
            text (str): The text of diagram updates.
            diagram_code (str): The code of diagram to be processed.
            openai_api_key (str): The OpenAI API key.

        Returns:
            str or None: The processed response, or None if no response is available.
        """

        messages = self._get_generate_messages(diagram_type, text[:-1])
        messages.append(AIMessage(content=diagram_code))
        messages.append(HumanMessage(content=text[-1]))

        result = await self._agenerate(messages, openai_api_key)

        if result is None or result == "NO_DIAGRAM":
            return diagram_code

        return self._match_diagram(result)

    def _get_generate_messages(self, diagram_type: MermaidDiagramType, text: list[str]):
        syntax_rules = self.prompt_provider.get_prompt(
            ["mermaid", "syntax"], diagram_type.value
        )

        messages = [
            SystemMessage(
                content=self.prompt_provider.get_prompt(
                    ["mermaid"],
                    "system_message",
                    syntax_rules=syntax_rules,
                    diagram_type=diagram_type.value,
                )
            )
        ]

        messages.extend(HumanMessage(content=text_item) for text_item in text)
        return messages

    async def _agenerate(self, messages, openai_api_key):
        self.logger.info(messages)
        response = await self._get_llm(openai_api_key).agenerate([messages])
        self.logger.info(response)
        result = response.generations[0][0].text
        return result

    def _match_diagram(self, result):
        match = self.regex.match(result)
        if match:
            result = match.group(1)

        return result.strip()

    def _get_llm(self, openai_api_key: str) -> ChatBedrock:
        if openai_api_key:
            self.logger.info("Using OpenAI service")
            return ChatOpenAI(temperature=0.0, openai_api_key=openai_api_key, model="gpt-4o")
        if "OPENAI_API_KEY" in os.environ:
            self.logger.info("Using OpenAI service")
            return ChatOpenAI(temperature=0.0, openai_api_key=os.environ.get("OPENAI_API_KEY"), model="gpt-4o")
        self.logger.info("Using Bedrock service")
        return ChatBedrock(
            model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            model_kwargs=dict(temperature=0.0),
        )

    def _messages_to_text(self, messages: list[HumanMessage | SystemMessage | AIMessage]) -> str:
        result = ""

        for message in messages:
            result += f"{'-'*30}\n**{self._message_type(message)}**: {message.content}\n\n"

        return result
    
    def _message_type(self, message: HumanMessage | SystemMessage | AIMessage) -> str:
        if isinstance(message, HumanMessage):
            return "HUMAN"
        elif isinstance(message, SystemMessage):
            return "INSTRUCTIONS"
        elif isinstance(message, AIMessage):
            return "AI"
        else:
            return "Unknown"
        
