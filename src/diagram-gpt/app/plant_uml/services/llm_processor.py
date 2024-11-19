import json
import re
import os
import inflection
import logging
from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrock
from langchain.schema.messages import AIMessage, HumanMessage, SystemMessage

from app.plant_uml.models.plant_uml_diagram_type import PlantUmlDiagramType
from prompts.prompt_provider import PromptProvider


class LlmProcessor:
    def __init__(self, prompt_provider: PromptProvider):
        self.prompt_provider = prompt_provider
        self.regex = re.compile(r"```(?:plantuml)?(.*)```", re.DOTALL)
        self.logger = logging.getLogger('uvicorn.error')
        self.logger.setLevel(logging.INFO)

    async def agenerate(
        self, diagram_type: PlantUmlDiagramType, text: list[str], openai_api_key: str
    ) -> str | None:
        """
        Process the given text and return the response.

        Args:
            text (str): The text to be processed.
            openai_api_key (str): The OpenAI API key.

        Returns:
            str or None: The processed response, or None if no response is available.
        """

        human_messages = self._human_messages(text)
        libs = await self._agenerate_required_libs(
            diagram_type, human_messages, openai_api_key
        )

        messages = self._get_generate_messages(diagram_type, human_messages, libs)
        result = await self._agenerate(messages, openai_api_key)

        if result is None or result == "NO_DIAGRAM":
            return None

        return self._match_diagram(result)

    async def aupdate(
        self,
        diagram_type: PlantUmlDiagramType,
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

        human_messages = self._human_messages(text)
        libs = await self._agenerate_required_libs(
            diagram_type, human_messages, openai_api_key
        )

        messages = self._get_generate_messages(diagram_type, human_messages[:-1], libs)
        messages.append(AIMessage(content=diagram_code))
        messages.append(human_messages[-1])

        result = await self._agenerate(messages, openai_api_key)

        if result is None or result == "NO_DIAGRAM":
            return diagram_code

        return self._match_diagram(result)

    def _get_generate_messages(
        self,
        diagram_type: PlantUmlDiagramType,
        human_messages: list[HumanMessage],
        libs: dict[str, dict],
    ):
        syntax_rules = self.prompt_provider.get_prompt(
            ["plant_uml", "syntax"], diagram_type.value
        )

        for lib, services_by_category in libs.items():
            lib_path = ["plant_uml", "libs", inflection.underscore(lib), "syntax"]
            services_syntax_rules = ""

            for category, services in services_by_category.items():
                category_path = lib_path + [inflection.underscore(category)]
                for service in services:
                    try:
                        service_syntax_rules = self.prompt_provider.get_prompt(
                            category_path,
                            inflection.underscore(service),
                        )
                    except FileNotFoundError as exc:
                        raise ValueError(f"Service {service} is not supported") from exc
                    services_syntax_rules += f"\n{service_syntax_rules}"

            lib_syntax_rules = self.prompt_provider.get_prompt(
                lib_path,
                "base",
                services_by_category=json.dumps(services_by_category),
                services_syntax_rules=services_syntax_rules,
            )
            syntax_rules += f"\n{lib_syntax_rules}"

        messages = [
            SystemMessage(
                content=self.prompt_provider.get_prompt(
                    ["plant_uml"],
                    "system_message",
                    syntax_rules=syntax_rules,
                    diagram_type=diagram_type.value,
                )
            )
        ]

        messages.extend(human_messages)
        return messages

    async def _agenerate_required_libs(
        self,
        diagram_type: PlantUmlDiagramType,
        human_messages: list[HumanMessage],
        openai_api_key: str,
    ):
        libs_messages = [
            SystemMessage(
                content=self.prompt_provider.get_prompt(
                    ["plant_uml", "libs"],
                    "system_message",
                    diagram_type=diagram_type.value,
                )
            )
        ]

        libs_messages.extend(human_messages)
        libs_result = await self._agenerate(libs_messages, openai_api_key)

        try:
            return json.loads(libs_result)
        except json.JSONDecodeError:
            return {}

    async def _agenerate(self, messages, openai_api_key: str):
        self.logger.info(messages)
        response = await self._get_llm(openai_api_key).agenerate([messages])
        self.logger.info(response)
        result = response.generations[0][0].text
        return result

    def _human_messages(self, text: list[str]):
        return [HumanMessage(content=text_item) for text_item in text]

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
