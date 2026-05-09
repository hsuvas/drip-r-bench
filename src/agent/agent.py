from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .conversation_state import ConversationState
from .llm_provider import LLMProvider

from .prompt_builder import (
    build_agent_messages,
    build_customer_messages,
)
from .response_parser import (
    AgentResponse,
    CustomerResponse,
    parse_agent_response,
    parse_customer_response,
)
from .tool_registry import get_agent_tools, get_customer_tools


class AgentInterface(ABC):
    """Contract that the orchestrator calls for each agent turn."""

    @abstractmethod
    def get_init_state_info(
        self,
        scenario: Dict[str, Any],
    ) -> Dict[str, Any]:
        ...

    @abstractmethod
    def generate_response(
        self,
        state: ConversationState,
        prior_variants_brief: str = "(none)",
    ) -> AgentResponse:
        ...


class LLMAgent(AgentInterface):
    """Agent driven by an LLM via the prompt template."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        scenario: Dict[str, Any],
        use_native_tools: bool = True,
        max_retries: int = 2,
        single_mode: bool = True,
    ):
        self.llm_provider = llm_provider
        self.scenario = scenario
        self.use_native_tools = use_native_tools
        self.max_retries = max_retries
        self.single_mode = single_mode

    def get_init_state_info(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def generate_response(
        self,
        state: ConversationState,
        prior_variants_brief: str = "(none)",
    ) -> AgentResponse:
        messages = build_agent_messages(
            scenario=self.scenario,
            state=state,
            prior_variants_brief=prior_variants_brief,
            use_native_tools=self.use_native_tools,
            single_mode=self.single_mode,
        )
        tools = get_agent_tools() if self.use_native_tools else None

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                llm_response = self.llm_provider.call_with_tools(
                    messages=messages,
                    tools=tools,
                )
                return parse_agent_response(llm_response)
            except Exception as e:
                last_error = e
                print(f"  [Agent parse fail attempt {attempt}] {type(e).__name__}: {e}")
                if attempt < self.max_retries:
                    messages.append({
                        "role": "user",
                        "content": (
                            f"CORRECTION: Your previous response was invalid. "
                            f"Return ONLY valid JSON matching the schema. Error: {e}"
                        ),
                    })

        raise RuntimeError(
            f"Agent failed after {self.max_retries + 1} attempts. "
            f"Last error: {last_error}"
        )


class LLMCustomer:
    """Customer simulator driven by an LLM."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        scenario: Dict[str, Any],
        max_retries: int = 2,
        single_mode: bool = True,
    ):
        self.llm_provider = llm_provider
        self.scenario = scenario
        self.max_retries = max_retries
        self.single_mode = single_mode

    def generate_response(
        self,
        state: ConversationState,
        prior_variants_brief: str = "(none)",
    ) -> CustomerResponse:
        messages = build_customer_messages(
            scenario=self.scenario,
            state=state,
            prior_variants_brief=prior_variants_brief,
            single_mode=self.single_mode,
        )
        tools = get_customer_tools()

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                llm_response = self.llm_provider.call_with_tools(
                    messages=messages,
                    tools=tools,
                )
                return parse_customer_response(llm_response)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    messages.append({
                        "role": "user",
                        "content": (
                            f"CORRECTION: Your previous response was invalid. "
                            f"Return ONLY valid JSON matching the schema. Error: {e}"
                        ),
                    })

        raise RuntimeError(
            f"Customer failed after {self.max_retries + 1} attempts. "
            f"Last error: {last_error}"
        )
