from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from rag.config import DEFAULT_ASSESSMENT_YEAR, DEFAULT_FINANCIAL_YEAR


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class ProfessionType(str, Enum):
    SALARIED = "salaried"
    FREELANCER = "freelancer"
    BUSINESS = "business"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class TaxRegime(str, Enum):
    OLD = "old"
    NEW = "new"
    UNKNOWN = "unknown"


class ResidentialStatus(str, Enum):
    RESIDENT = "resident"
    NON_RESIDENT = "non_resident"
    RNOR = "resident_not_ordinary"


class EmployerType(str, Enum):
    OTHER = "other"
    PSU = "psu"
    CENTRAL_GOVERNMENT = "central_government"
    STATE_GOVERNMENT = "state_government"


class Citation(BaseModel):
    label: str
    source_file: str
    page_number: str | int | None = None
    section_title: str | None = None
    statute_reference: str | None = None
    chunk_id: str | None = None
    score: float | None = None
    excerpt: str


class QueryResult(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved_chunks: int = 0
    generated_at: str = Field(default_factory=utc_now_iso)


class ChunkCatalogRecord(BaseModel):
    chunk_id: str
    text: str
    metadata: dict[str, Any]


class UserTaxProfile(BaseModel):
    user_id: str
    full_name: str | None = None
    profession_type: ProfessionType = ProfessionType.UNKNOWN
    tax_regime: TaxRegime = TaxRegime.UNKNOWN
    financial_year: str = DEFAULT_FINANCIAL_YEAR
    assessment_year: str = DEFAULT_ASSESSMENT_YEAR
    age: int | None = None
    residential_status: ResidentialStatus = ResidentialStatus.RESIDENT
    employer_type: EmployerType = EmployerType.OTHER

    salary_income: float = 0.0
    pension_income: float = 0.0
    freelance_receipts: float = 0.0
    freelance_expenses: float = 0.0
    business_receipts: float = 0.0
    business_expenses: float = 0.0
    interest_income: float = 0.0
    savings_interest_income: float = 0.0
    fixed_deposit_interest_income: float = 0.0
    rental_income: float = 0.0
    other_income: float = 0.0
    capital_gains_special_rate: float = 0.0

    use_presumptive_profession: bool = False
    presumptive_profession_rate: float = 0.5
    use_presumptive_business: bool = False
    presumptive_business_rate: float = 0.08

    salary_standard_deduction_enabled: bool = True
    exempt_allowances_old_regime: float = 0.0
    house_property_interest_self_occupied: float = 0.0
    employer_nps_contribution: float = 0.0
    section_80c_total: float = 0.0
    section_80ccd1b: float = 0.0
    section_80d_self_family: float = 0.0
    section_80d_parents: float = 0.0
    section_80e_interest: float = 0.0
    section_80g_donations: float = 0.0
    section_80cch_contribution: float = 0.0

    parents_are_senior_citizens: bool = False
    notes: list[str] = Field(default_factory=list)
    known_facts: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    updated_at: str = Field(default_factory=utc_now_iso)

    def touch(self) -> "UserTaxProfile":
        self.updated_at = utc_now_iso()
        return self

    def has_business_or_profession_income(self) -> bool:
        return any(
            value > 0
            for value in (
                self.freelance_receipts,
                self.business_receipts,
            )
        )

    def inferred_profession_type(self) -> ProfessionType:
        if self.profession_type != ProfessionType.UNKNOWN:
            return self.profession_type

        has_salary = self.salary_income > 0 or self.pension_income > 0
        has_freelance = self.freelance_receipts > 0
        has_business = self.business_receipts > 0

        if has_salary and (has_freelance or has_business):
            return ProfessionType.MIXED
        if has_business:
            return ProfessionType.BUSINESS
        if has_freelance:
            return ProfessionType.FREELANCER
        if has_salary:
            return ProfessionType.SALARIED
        return ProfessionType.UNKNOWN


class AppliedDeduction(BaseModel):
    section: str
    label: str
    amount: float
    note: str | None = None


class ApplicableDeduction(BaseModel):
    section: str
    label: str
    regimes: list[str]
    max_amount: float | None = None
    likely_applicable: bool = False
    reason: str
    notes: list[str] = Field(default_factory=list)


class TaxCalculationResult(BaseModel):
    user_id: str
    regime: TaxRegime
    financial_year: str
    assessment_year: str
    income_breakdown: dict[str, float]
    gross_total_income: float
    total_deductions: float
    taxable_income: float
    slab_tax: float
    tax_before_rebate: float
    rebate: float
    surcharge: float
    cess: float
    total_tax: float
    applied_deductions: list[AppliedDeduction] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    computed_at: str = Field(default_factory=utc_now_iso)


class RegimeComparisonResult(BaseModel):
    recommended_regime: TaxRegime
    old_regime: TaxCalculationResult
    new_regime: TaxCalculationResult
    tax_saving: float


class ChatTurn(BaseModel):
    role: str
    content: str
    name: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)


class ChatSession(BaseModel):
    session_id: str
    user_id: str
    title: str | None = None
    turns: list[ChatTurn] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)

    def append_turn(self, role: str, content: str, name: str | None = None) -> "ChatSession":
        self.turns.append(ChatTurn(role=role, content=content, name=name))
        self.updated_at = utc_now_iso()
        return self

    def recent_turns(self, limit: int) -> list[ChatTurn]:
        if limit <= 0:
            return []
        return self.turns[-limit:]
