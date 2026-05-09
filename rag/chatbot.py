from __future__ import annotations

import argparse
import json
from typing import Any

from rag.chat_memory import ChatSessionStore
from rag.config import DEFAULT_CHAT_HISTORY_TURNS, DEFAULT_CHAT_TOOL_STEPS
from rag.models import UserTaxProfile
from rag.profile_store import UserProfileStore
from rag.query_engine import answer_question
from rag.settings import get_chat_model_name, get_openai_chat_client, require_openai_key
from rag.tax_tools import (
    calculate_tax,
    compare_old_vs_new_regime,
    explain_tax_breakdown,
    list_missing_information,
    suggest_applicable_deductions,
)


CHATBOT_SYSTEM_PROMPT = """You are an Indian tax-law chatbot with tool access.

Use tools actively.
- If the user shares personal tax facts, call update_user_profile.
- If the question asks about tax amount, liability, old-vs-new comparison, or deductions based on the user's situation, call the profile and tax tools.
- If the question asks about Indian tax law, sections, rules, eligibility, procedure, compliance, deduction law, or legal basis, call answer_tax_law_question.
- If the answer needs both legal grounding and tax calculation, use both the tax tools and the RAG tool.

Be careful with assumptions.
- Do not silently invent salary, deduction, or regime values.
- If important details are missing, call list_missing_information and say exactly what is still needed.
- Use concise language.
- When the RAG tool returns citations, preserve them in your final answer.
"""


COMMON_PROFILE_FIELDS = [
    "full_name",
    "profession_type",
    "tax_regime",
    "financial_year",
    "assessment_year",
    "age",
    "salary_income",
    "pension_income",
    "freelance_receipts",
    "freelance_expenses",
    "business_receipts",
    "business_expenses",
    "interest_income",
    "savings_interest_income",
    "fixed_deposit_interest_income",
    "rental_income",
    "other_income",
    "capital_gains_special_rate",
    "use_presumptive_profession",
    "use_presumptive_business",
    "house_property_interest_self_occupied",
    "employer_nps_contribution",
    "section_80c_total",
    "section_80ccd1b",
    "section_80d_self_family",
    "section_80d_parents",
    "section_80e_interest",
    "section_80g_donations",
    "section_80cch_contribution",
    "parents_are_senior_citizens",
]


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=True)


def _profile_snapshot(profile: UserTaxProfile) -> str:
    relevant = {
        "user_id": profile.user_id,
        "profession_type": profile.inferred_profession_type().value,
        "tax_regime": profile.tax_regime.value,
        "financial_year": profile.financial_year,
        "assessment_year": profile.assessment_year,
        "salary_income": profile.salary_income,
        "freelance_receipts": profile.freelance_receipts,
        "freelance_expenses": profile.freelance_expenses,
        "business_receipts": profile.business_receipts,
        "business_expenses": profile.business_expenses,
        "interest_income": profile.interest_income,
        "rental_income": profile.rental_income,
        "other_income": profile.other_income,
        "employer_nps_contribution": profile.employer_nps_contribution,
        "section_80c_total": profile.section_80c_total,
        "section_80ccd1b": profile.section_80ccd1b,
        "section_80d_self_family": profile.section_80d_self_family,
        "section_80d_parents": profile.section_80d_parents,
        "section_80e_interest": profile.section_80e_interest,
        "house_property_interest_self_occupied": profile.house_property_interest_self_occupied,
        "known_facts": profile.known_facts,
        "notes": profile.notes,
    }
    return _json_dumps(relevant)


def _format_rag_sources(tool_result: dict[str, Any]) -> str:
    citations = tool_result.get("citations") or []
    if not citations:
        return ""

    lines = ["Sources:"]
    for citation in citations:
        lines.append(
            f"{citation.get('label', '[S?]')} "
            f"{citation.get('source_file', 'Unknown')} | "
            f"page {citation.get('page_number', 'Unknown')} | "
            f"{citation.get('section_title') or 'Unknown'}"
        )
    return "\n".join(lines)


class TaxChatbot:
    def __init__(
        self,
        user_id: str = "default-user",
        session_id: str | None = None,
        history_turns: int = DEFAULT_CHAT_HISTORY_TURNS,
        max_tool_steps: int = DEFAULT_CHAT_TOOL_STEPS,
        profile_store: UserProfileStore | None = None,
        session_store: ChatSessionStore | None = None,
    ) -> None:
        require_openai_key()
        self._user_id = user_id
        self._history_turns = history_turns
        self._max_tool_steps = max_tool_steps
        self._profile_store = profile_store or UserProfileStore()
        self._session_store = session_store or ChatSessionStore()
        self._session = self._session_store.get_or_create(
            user_id=user_id,
            session_id=session_id,
        )
        self._client = get_openai_chat_client()
        self._model = get_chat_model_name()

    @property
    def session_id(self) -> str:
        return self._session.session_id

    def get_profile(self) -> UserTaxProfile:
        return self._profile_store.get(self._user_id) or UserTaxProfile(user_id=self._user_id)

    def get_recent_turns(self) -> list[dict[str, str]]:
        return self._session_store.recent_messages(
            self._session.session_id,
            max_turns=self._history_turns,
        )

    def _build_messages(self) -> list[dict[str, Any]]:
        profile = self.get_profile()

        system_content = (
            f"{CHATBOT_SYSTEM_PROMPT}\n\n"
            f"Current user profile:\n{_profile_snapshot(profile)}"
        )

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_content}]
        messages.extend(self._session_store.recent_messages(self._session.session_id, self._history_turns))
        return messages

    def _tool_definitions(self) -> list[dict[str, Any]]:
        profile_properties: dict[str, Any] = {
            "full_name": {"type": "string"},
            "profession_type": {
                "type": "string",
                "enum": ["salaried", "freelancer", "business", "mixed", "unknown"],
            },
            "tax_regime": {"type": "string", "enum": ["old", "new", "unknown"]},
            "financial_year": {"type": "string"},
            "assessment_year": {"type": "string"},
            "age": {"type": "integer"},
            "salary_income": {"type": "number"},
            "pension_income": {"type": "number"},
            "freelance_receipts": {"type": "number"},
            "freelance_expenses": {"type": "number"},
            "business_receipts": {"type": "number"},
            "business_expenses": {"type": "number"},
            "interest_income": {"type": "number"},
            "savings_interest_income": {"type": "number"},
            "fixed_deposit_interest_income": {"type": "number"},
            "rental_income": {"type": "number"},
            "other_income": {"type": "number"},
            "capital_gains_special_rate": {"type": "number"},
            "use_presumptive_profession": {"type": "boolean"},
            "use_presumptive_business": {"type": "boolean"},
            "house_property_interest_self_occupied": {"type": "number"},
            "employer_nps_contribution": {"type": "number"},
            "section_80c_total": {"type": "number"},
            "section_80ccd1b": {"type": "number"},
            "section_80d_self_family": {"type": "number"},
            "section_80d_parents": {"type": "number"},
            "section_80e_interest": {"type": "number"},
            "section_80g_donations": {"type": "number"},
            "section_80cch_contribution": {"type": "number"},
            "parents_are_senior_citizens": {"type": "boolean"},
            "notes_to_add": {"type": "array", "items": {"type": "string"}},
            "known_facts_to_add": {"type": "array", "items": {"type": "string"}},
        }

        return [
            {
                "type": "function",
                "function": {
                    "name": "get_user_profile",
                    "description": "Fetch the current stored tax profile for this user.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "update_user_profile",
                    "description": "Update the stored tax profile using facts the user has given in conversation.",
                    "parameters": {
                        "type": "object",
                        "properties": profile_properties,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "answer_tax_law_question",
                    "description": "Use the RAG engine to answer Indian tax-law questions with citations.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The tax-law question to answer using retrieved sources.",
                            }
                        },
                        "required": ["question"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_tax",
                    "description": "Calculate tax for the current user profile or for a requested regime.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "regime": {
                                "type": "string",
                                "enum": ["old", "new"],
                                "description": "Optional regime override.",
                            }
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "compare_old_vs_new_regime",
                    "description": "Compare tax between the old and new regime for the current user profile.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "suggest_applicable_deductions",
                    "description": "Suggest which deductions may apply for the current user profile.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_missing_information",
                    "description": "List missing profile fields needed for a more accurate personalised tax answer.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

    def _merge_list_field(
        self,
        existing_values: list[str],
        new_values: list[str] | None,
    ) -> list[str]:
        merged = list(existing_values)
        for item in new_values or []:
            if item not in merged:
                merged.append(item)
        return merged

    def _sanitize_profile_updates(self, raw_updates: dict[str, Any]) -> dict[str, Any]:
        allowed_updates = {key: raw_updates[key] for key in COMMON_PROFILE_FIELDS if key in raw_updates}
        if "profession_type" in allowed_updates:
            allowed_updates["profession_type"] = str(allowed_updates["profession_type"]).lower()
        if "tax_regime" in allowed_updates:
            allowed_updates["tax_regime"] = str(allowed_updates["tax_regime"]).lower()
        return allowed_updates

    def _execute_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "get_user_profile":
            profile = self.get_profile()
            return profile.model_dump(mode="json")

        if name == "update_user_profile":
            current_profile = self.get_profile()
            updates = self._sanitize_profile_updates(arguments)
            merged_payload = current_profile.model_dump(mode="json")
            merged_payload.update(updates)
            merged_payload["notes"] = self._merge_list_field(
                current_profile.notes,
                arguments.get("notes_to_add"),
            )
            merged_payload["known_facts"] = self._merge_list_field(
                current_profile.known_facts,
                arguments.get("known_facts_to_add"),
            )
            merged = UserTaxProfile.model_validate(merged_payload)
            saved = self._profile_store.save(merged)
            return {
                "status": "updated",
                "profile": saved.model_dump(mode="json"),
            }

        if name == "answer_tax_law_question":
            question = arguments["question"]
            result = answer_question(question)
            return result.model_dump(mode="json")

        if name == "calculate_tax":
            profile = self.get_profile()
            regime = arguments.get("regime")
            result = calculate_tax(profile, regime=regime)
            return {
                "calculation": result.model_dump(mode="json"),
                "breakdown_text": explain_tax_breakdown(result),
            }

        if name == "compare_old_vs_new_regime":
            profile = self.get_profile()
            result = compare_old_vs_new_regime(profile)
            return result.model_dump(mode="json")

        if name == "suggest_applicable_deductions":
            profile = self.get_profile()
            deductions = suggest_applicable_deductions(profile)
            return {
                "deductions": [item.model_dump(mode="json") for item in deductions]
            }

        if name == "list_missing_information":
            profile = self.get_profile()
            return {"missing_fields": list_missing_information(profile)}

        raise ValueError(f"Unknown tool: {name}")

    def chat(self, user_message: str) -> str:
        self._session = self._session_store.append_turn(
            self._session.session_id,
            role="user",
            content=user_message,
        )
        messages = self._build_messages()
        rag_tool_results: list[dict[str, Any]] = []

        for _ in range(self._max_tool_steps):
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=self._tool_definitions(),
                tool_choice="auto",
                temperature=0.1,
                parallel_tool_calls=False,
                user=self._user_id,
            )

            message = response.choices[0].message
            assistant_payload = message.model_dump(exclude_none=True)
            tool_calls = assistant_payload.get("tool_calls") or []

            if tool_calls:
                messages.append(assistant_payload)
                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    raw_arguments = tool_call["function"].get("arguments") or "{}"
                    try:
                        parsed_arguments = json.loads(raw_arguments)
                    except json.JSONDecodeError as exc:
                        parsed_arguments = {}
                        tool_result = {
                            "tool_error": f"Could not parse tool arguments: {exc}",
                            "tool_name": tool_name,
                        }
                    else:
                        try:
                            tool_result = self._execute_tool(tool_name, parsed_arguments)
                        except Exception as exc:  # noqa: BLE001
                            tool_result = {
                                "tool_error": str(exc),
                                "tool_name": tool_name,
                                "arguments": parsed_arguments,
                            }
                    if tool_name == "answer_tax_law_question" and "tool_error" not in tool_result:
                        rag_tool_results.append(tool_result)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": _json_dumps(tool_result),
                        }
                    )
                continue

            answer = (assistant_payload.get("content") or "").strip()
            if not answer:
                answer = "I could not produce a final answer for that turn."
            elif rag_tool_results and "Sources:" not in answer:
                rag_sources = _format_rag_sources(rag_tool_results[-1])
                if rag_sources:
                    answer = f"{answer}\n\n{rag_sources}"

            self._session = self._session_store.append_turn(
                self._session.session_id,
                role="assistant",
                content=answer,
            )
            return answer

        fallback = (
            "I hit the tool-call limit for this turn. Please ask again with a shorter or more specific question."
        )
        self._session = self._session_store.append_turn(
            self._session.session_id,
            role="assistant",
            content=fallback,
        )
        return fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Indian tax-law CLI chatbot.")
    parser.add_argument("--user-id", default="demo-user", help="Stable user id for profile memory.")
    parser.add_argument("--session-id", default=None, help="Existing session id to resume.")
    args = parser.parse_args()

    bot = TaxChatbot(user_id=args.user_id, session_id=args.session_id)

    print(f"Session: {bot.session_id}")
    print("Type '/quit' to exit, '/profile' to inspect saved profile, or '/history' to inspect recent turns.")

    while True:
        try:
            user_message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_message:
            continue

        if user_message.lower() in {"/quit", "/exit"}:
            print("Bye.")
            break

        if user_message.lower() == "/profile":
            print(_json_dumps(bot.get_profile().model_dump(mode="json")))
            continue

        if user_message.lower() == "/history":
            print(_json_dumps(bot.get_recent_turns()))
            continue

        answer = bot.chat(user_message)
        print(f"Bot: {answer}\n")


if __name__ == "__main__":
    main()
