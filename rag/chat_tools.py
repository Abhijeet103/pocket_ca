from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field


if TYPE_CHECKING:
    from rag.chatbot import TaxChatbot


class UpdateUserProfileArgs(BaseModel):
    full_name: str | None = Field(default=None, description="Full name of the user.")
    profession_type: Literal["salaried", "freelancer", "business", "mixed", "unknown"] | None = Field(
        default=None,
        description="Primary profession or income style of the user.",
    )
    tax_regime: Literal["old", "new", "unknown"] | None = Field(
        default=None,
        description="Tax regime currently used or preferred by the user.",
    )
    financial_year: str | None = Field(default=None, description="Financial year such as FY 2025-26.")
    assessment_year: str | None = Field(default=None, description="Assessment year such as AY 2026-27.")
    age: int | None = Field(default=None, description="Age of the user in years.")
    salary_income: float | None = Field(default=None, description="Annual salary income.")
    pension_income: float | None = Field(default=None, description="Annual pension income.")
    freelance_receipts: float | None = Field(default=None, description="Gross freelance or professional receipts.")
    freelance_expenses: float | None = Field(default=None, description="Freelance or professional expenses.")
    business_receipts: float | None = Field(default=None, description="Gross business receipts.")
    business_expenses: float | None = Field(default=None, description="Business expenses.")
    interest_income: float | None = Field(default=None, description="Total interest income.")
    savings_interest_income: float | None = Field(default=None, description="Savings account interest income.")
    fixed_deposit_interest_income: float | None = Field(default=None, description="Fixed deposit interest income.")
    rental_income: float | None = Field(default=None, description="Rental income.")
    other_income: float | None = Field(default=None, description="Any other taxable income.")
    capital_gains_special_rate: float | None = Field(default=None, description="Capital gains taxable at special rates.")
    use_presumptive_profession: bool | None = Field(default=None, description="Whether presumptive taxation is used for profession.")
    use_presumptive_business: bool | None = Field(default=None, description="Whether presumptive taxation is used for business.")
    house_property_interest_self_occupied: float | None = Field(default=None, description="Interest on self-occupied house property loan.")
    employer_nps_contribution: float | None = Field(default=None, description="Employer contribution to NPS.")
    section_80c_total: float | None = Field(default=None, description="Total amount for section 80C investments.")
    section_80ccd1b: float | None = Field(default=None, description="Additional NPS contribution under section 80CCD(1B).")
    section_80d_self_family: float | None = Field(default=None, description="Medical insurance premium for self and family.")
    section_80d_parents: float | None = Field(default=None, description="Medical insurance premium for parents.")
    section_80e_interest: float | None = Field(default=None, description="Education loan interest under section 80E.")
    section_80g_donations: float | None = Field(default=None, description="Donation amount relevant for section 80G.")
    section_80cch_contribution: float | None = Field(default=None, description="Contribution relevant for section 80CCH.")
    parents_are_senior_citizens: bool | None = Field(default=None, description="Whether the user's parents qualify as senior citizens.")
    notes_to_add: list[str] | None = Field(default=None, description="Notes to append to the stored profile.")
    known_facts_to_add: list[str] | None = Field(default=None, description="Known facts to append to the stored profile.")


class AnswerTaxLawQuestionArgs(BaseModel):
    question: str = Field(
        description="Indian tax-law question to answer using the graph RAG engine.",
    )


class CalculateTaxArgs(BaseModel):
    regime: Literal["old", "new"] | None = Field(
        default=None,
        description="Optional regime override for the tax calculation.",
    )


def build_chat_tools(chatbot: TaxChatbot) -> list[BaseTool]:
    @tool
    def get_user_profile() -> dict[str, Any]:
        """Fetch the current stored tax profile for this user."""
        return chatbot.get_profile().model_dump(mode="json")

    @tool(args_schema=UpdateUserProfileArgs)
    def update_user_profile(
        full_name: str | None = None,
        profession_type: str | None = None,
        tax_regime: str | None = None,
        financial_year: str | None = None,
        assessment_year: str | None = None,
        age: int | None = None,
        salary_income: float | None = None,
        pension_income: float | None = None,
        freelance_receipts: float | None = None,
        freelance_expenses: float | None = None,
        business_receipts: float | None = None,
        business_expenses: float | None = None,
        interest_income: float | None = None,
        savings_interest_income: float | None = None,
        fixed_deposit_interest_income: float | None = None,
        rental_income: float | None = None,
        other_income: float | None = None,
        capital_gains_special_rate: float | None = None,
        use_presumptive_profession: bool | None = None,
        use_presumptive_business: bool | None = None,
        house_property_interest_self_occupied: float | None = None,
        employer_nps_contribution: float | None = None,
        section_80c_total: float | None = None,
        section_80ccd1b: float | None = None,
        section_80d_self_family: float | None = None,
        section_80d_parents: float | None = None,
        section_80e_interest: float | None = None,
        section_80g_donations: float | None = None,
        section_80cch_contribution: float | None = None,
        parents_are_senior_citizens: bool | None = None,
        notes_to_add: list[str] | None = None,
        known_facts_to_add: list[str] | None = None,
    ) -> dict[str, Any]:
        """Update the stored tax profile using facts the user has given in conversation."""
        payload = {
            key: value
            for key, value in locals().items()
            if key != "chatbot"
        }
        return chatbot.apply_profile_updates(payload)

    @tool(args_schema=AnswerTaxLawQuestionArgs)
    def answer_tax_law_question(question: str) -> dict[str, Any]:
        """Use the graph RAG engine to answer an Indian tax-law question with citations."""
        return chatbot.answer_tax_law_question(question)

    @tool(args_schema=CalculateTaxArgs)
    def calculate_tax_tool(regime: str | None = None) -> dict[str, Any]:
        """Calculate tax for the current user profile, optionally forcing old or new regime."""
        return chatbot.calculate_tax_for_profile(regime=regime)

    calculate_tax_tool.name = "calculate_tax"

    @tool
    def compare_old_vs_new_regime() -> dict[str, Any]:
        """Compare tax between the old and new regime for the current user profile."""
        return chatbot.compare_tax_regimes()

    @tool
    def suggest_applicable_deductions() -> dict[str, Any]:
        """Suggest which deductions may apply for the current user profile."""
        return chatbot.suggest_deductions()

    @tool
    def list_missing_information() -> dict[str, Any]:
        """List missing profile fields needed for a more accurate personalised tax answer."""
        return chatbot.list_missing_information_for_profile()

    return [
        get_user_profile,
        update_user_profile,
        answer_tax_law_question,
        calculate_tax_tool,
        compare_old_vs_new_regime,
        suggest_applicable_deductions,
        list_missing_information,
    ]
