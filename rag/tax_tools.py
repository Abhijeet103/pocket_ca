from __future__ import annotations

from math import inf

from rag.config import DEFAULT_STANDARD_DEDUCTION
from rag.models import (
    AppliedDeduction,
    ApplicableDeduction,
    EmployerType,
    ProfessionType,
    RegimeComparisonResult,
    ResidentialStatus,
    TaxCalculationResult,
    TaxRegime,
    UserTaxProfile,
)


OLD_REGIME_SLABS_UNDER_60 = [
    (250000.0, 0.00),
    (500000.0, 0.05),
    (1000000.0, 0.20),
    (inf, 0.30),
]

OLD_REGIME_SLABS_SENIOR = [
    (300000.0, 0.00),
    (500000.0, 0.05),
    (1000000.0, 0.20),
    (inf, 0.30),
]

OLD_REGIME_SLABS_SUPER_SENIOR = [
    (500000.0, 0.00),
    (1000000.0, 0.20),
    (inf, 0.30),
]

NEW_REGIME_SLABS = [
    (400000.0, 0.00),
    (800000.0, 0.05),
    (1200000.0, 0.10),
    (1600000.0, 0.15),
    (2000000.0, 0.20),
    (2400000.0, 0.25),
    (inf, 0.30),
]

SURCHARGE_RATES_OLD = [
    (5000000.0, 0.00),
    (10000000.0, 0.10),
    (20000000.0, 0.15),
    (50000000.0, 0.25),
    (inf, 0.37),
]

SURCHARGE_RATES_NEW = [
    (5000000.0, 0.00),
    (10000000.0, 0.10),
    (20000000.0, 0.15),
    (inf, 0.25),
]

SURCHARGE_THRESHOLDS_OLD = [5000000.0, 10000000.0, 20000000.0, 50000000.0]
SURCHARGE_THRESHOLDS_NEW = [5000000.0, 10000000.0, 20000000.0]


def _round_money(value: float) -> float:
    return round(float(value), 2)


def _coerce_profile(profile: UserTaxProfile | dict) -> UserTaxProfile:
    if isinstance(profile, UserTaxProfile):
        return profile.model_copy(deep=True)
    return UserTaxProfile(**profile)


def _applied_deduction(
    section: str,
    label: str,
    amount: float,
    note: str | None = None,
) -> AppliedDeduction | None:
    amount = _round_money(max(amount, 0.0))
    if amount <= 0:
        return None
    return AppliedDeduction(section=section, label=label, amount=amount, note=note)


def _slab_tax(taxable_income: float, slabs: list[tuple[float, float]]) -> float:
    tax = 0.0
    previous_limit = 0.0
    for upper_limit, rate in slabs:
        if taxable_income <= previous_limit:
            break
        amount_in_slab = min(taxable_income, upper_limit) - previous_limit
        tax += amount_in_slab * rate
        previous_limit = upper_limit
    return _round_money(tax)


def _old_regime_slabs(profile: UserTaxProfile) -> list[tuple[float, float]]:
    if profile.age is None:
        return OLD_REGIME_SLABS_UNDER_60
    if profile.age >= 80:
        return OLD_REGIME_SLABS_SUPER_SENIOR
    if profile.age >= 60:
        return OLD_REGIME_SLABS_SENIOR
    return OLD_REGIME_SLABS_UNDER_60


def _compute_business_income(profile: UserTaxProfile, warnings: list[str]) -> float:
    if profile.use_presumptive_business:
        return _round_money(profile.business_receipts * profile.presumptive_business_rate)

    business_income = profile.business_receipts - profile.business_expenses
    if business_income < 0:
        warnings.append(
            "Business loss set-off is not modelled yet; business income was floored at 0."
        )
        return 0.0
    return _round_money(business_income)


def _compute_professional_income(profile: UserTaxProfile, warnings: list[str]) -> float:
    if profile.use_presumptive_profession:
        return _round_money(
            profile.freelance_receipts * profile.presumptive_profession_rate
        )

    professional_income = profile.freelance_receipts - profile.freelance_expenses
    if professional_income < 0:
        warnings.append(
            "Professional loss set-off is not modelled yet; freelance income was floored at 0."
        )
        return 0.0
    return _round_money(professional_income)


def _standard_deduction_amount(profile: UserTaxProfile) -> float:
    gross_salary_like_income = profile.salary_income + profile.pension_income
    if not profile.salary_standard_deduction_enabled or gross_salary_like_income <= 0:
        return 0.0
    return _round_money(min(DEFAULT_STANDARD_DEDUCTION, gross_salary_like_income))


def _compute_gross_total_income(
    profile: UserTaxProfile,
    regime: TaxRegime,
    assumptions: list[str],
    warnings: list[str],
) -> tuple[dict[str, float], list[AppliedDeduction]]:
    standard_deduction = _standard_deduction_amount(profile)
    salary_income = profile.salary_income + profile.pension_income
    taxable_salary = salary_income

    if regime == TaxRegime.OLD and profile.exempt_allowances_old_regime > 0:
        taxable_salary -= profile.exempt_allowances_old_regime
    elif regime == TaxRegime.NEW and profile.exempt_allowances_old_regime > 0:
        warnings.append(
            "Old-regime salary exemptions were ignored because the calculation is under the new regime."
        )

    taxable_salary = max(taxable_salary - standard_deduction, 0.0)
    if standard_deduction > 0:
        assumptions.append(
            f"Applied a default salary/pension standard deduction of up to Rs. {int(DEFAULT_STANDARD_DEDUCTION):,}."
        )

    professional_income = _compute_professional_income(profile, warnings)
    business_income = _compute_business_income(profile, warnings)

    income_breakdown = {
        "salary_and_pension_after_standard_deduction": _round_money(taxable_salary),
        "professional_income": professional_income,
        "business_income": business_income,
        "interest_income": _round_money(profile.interest_income),
        "rental_income": _round_money(profile.rental_income),
        "other_income": _round_money(profile.other_income),
    }

    if profile.capital_gains_special_rate > 0:
        warnings.append(
            "Special-rate capital gains were not included in slab-tax computation yet."
        )

    standard_deduction_entry = _applied_deduction(
        section="16(ia)",
        label="Standard deduction",
        amount=standard_deduction,
        note="Applied to salary/pension income.",
    )

    initial_deductions = [entry for entry in [standard_deduction_entry] if entry]
    return income_breakdown, initial_deductions


def _employer_nps_limit(profile: UserTaxProfile, regime: TaxRegime) -> float:
    salary_base = max(profile.salary_income + profile.pension_income, 0.0)
    if salary_base <= 0:
        return 0.0

    if regime == TaxRegime.NEW:
        return salary_base * 0.14

    if profile.employer_type in (
        EmployerType.CENTRAL_GOVERNMENT,
        EmployerType.STATE_GOVERNMENT,
    ):
        return salary_base * 0.14
    return salary_base * 0.10


def _compute_deductions(
    profile: UserTaxProfile,
    regime: TaxRegime,
    assumptions: list[str],
    warnings: list[str],
) -> list[AppliedDeduction]:
    deductions: list[AppliedDeduction] = []

    employer_nps_amount = min(
        max(profile.employer_nps_contribution, 0.0),
        _employer_nps_limit(profile, regime),
    )
    deduction = _applied_deduction(
        section="80CCD(2)",
        label="Employer NPS contribution",
        amount=employer_nps_amount,
        note="Capped as per the selected regime and employer category.",
    )
    if deduction:
        deductions.append(deduction)

    agniveer_deduction = _applied_deduction(
        section="80CCH",
        label="Agniveer Corpus contribution",
        amount=max(profile.section_80cch_contribution, 0.0),
    )
    if agniveer_deduction:
        deductions.append(agniveer_deduction)

    if regime == TaxRegime.NEW:
        if profile.house_property_interest_self_occupied > 0:
            warnings.append(
                "Self-occupied house-property interest was not deducted because it is not modelled under the new regime."
            )
        if profile.section_80c_total > 0 or profile.section_80ccd1b > 0 or profile.section_80d_self_family > 0 or profile.section_80d_parents > 0:
            warnings.append(
                "Old-regime Chapter VI-A deductions such as 80C/80CCD(1B)/80D were ignored under the new regime."
            )
        if profile.section_80g_donations > 0:
            warnings.append(
                "Section 80G donations were not auto-applied because qualifying limits depend on the notified donee category."
            )
        return deductions

    house_property_deduction = _applied_deduction(
        section="24(b)",
        label="Self-occupied housing-loan interest",
        amount=min(max(profile.house_property_interest_self_occupied, 0.0), 200000.0),
        note="Capped at Rs. 2,00,000 for self-occupied property.",
    )
    if house_property_deduction:
        deductions.append(house_property_deduction)

    section_80c = _applied_deduction(
        section="80C/80CCC/80CCD(1)",
        label="Combined Chapter VI-A investment deduction",
        amount=min(max(profile.section_80c_total, 0.0), 150000.0),
        note="Combined cap of Rs. 1,50,000.",
    )
    if section_80c:
        deductions.append(section_80c)

    section_80ccd1b = _applied_deduction(
        section="80CCD(1B)",
        label="Additional NPS self-contribution",
        amount=min(max(profile.section_80ccd1b, 0.0), 50000.0),
    )
    if section_80ccd1b:
        deductions.append(section_80ccd1b)

    self_family_cap = 50000.0 if (profile.age or 0) >= 60 else 25000.0
    parents_cap = 50000.0 if profile.parents_are_senior_citizens else 25000.0

    section_80d_self = _applied_deduction(
        section="80D",
        label="Health insurance for self/family",
        amount=min(max(profile.section_80d_self_family, 0.0), self_family_cap),
        note=f"Capped at Rs. {int(self_family_cap):,}.",
    )
    if section_80d_self:
        deductions.append(section_80d_self)

    section_80d_parents = _applied_deduction(
        section="80D",
        label="Health insurance for parents",
        amount=min(max(profile.section_80d_parents, 0.0), parents_cap),
        note=f"Capped at Rs. {int(parents_cap):,}.",
    )
    if section_80d_parents:
        deductions.append(section_80d_parents)

    section_80e = _applied_deduction(
        section="80E",
        label="Education-loan interest",
        amount=max(profile.section_80e_interest, 0.0),
    )
    if section_80e:
        deductions.append(section_80e)

    if (profile.age or 0) >= 60:
        ttb_base = max(
            profile.fixed_deposit_interest_income + profile.savings_interest_income, 0.0
        )
        section_80ttb = _applied_deduction(
            section="80TTB",
            label="Interest on deposits for senior citizens",
            amount=min(ttb_base, 50000.0),
            note="Assumes the eligible deposit interest was provided in savings/fixed-deposit fields.",
        )
        if section_80ttb:
            deductions.append(section_80ttb)
    else:
        section_80tta = _applied_deduction(
            section="80TTA",
            label="Savings-account interest",
            amount=min(max(profile.savings_interest_income, 0.0), 10000.0),
        )
        if section_80tta:
            deductions.append(section_80tta)

    if profile.section_80g_donations > 0:
        warnings.append(
            "Section 80G donations were not auto-applied because qualifying limits depend on the notified donee category."
        )

    return deductions


def _surcharge_rate(taxable_income: float, regime: TaxRegime) -> float:
    slab = SURCHARGE_RATES_NEW if regime == TaxRegime.NEW else SURCHARGE_RATES_OLD
    for upper_limit, rate in slab:
        if taxable_income <= upper_limit:
            return rate
    return 0.0


def _compute_tax_at_threshold(threshold: float, regime: TaxRegime, slabs) -> float:
    base_tax = _slab_tax(threshold, slabs)
    rate = _surcharge_rate(threshold, regime)
    return base_tax + (base_tax * rate)


def _apply_marginal_relief(
    taxable_income: float,
    tax_before_cess: float,
    regime: TaxRegime,
    slabs,
) -> float:
    thresholds = (
        SURCHARGE_THRESHOLDS_NEW if regime == TaxRegime.NEW else SURCHARGE_THRESHOLDS_OLD
    )
    for threshold in thresholds:
        if taxable_income > threshold:
            threshold_tax = _compute_tax_at_threshold(threshold, regime, slabs)
            max_tax = threshold_tax + (taxable_income - threshold)
            tax_before_cess = min(tax_before_cess, max_tax)
    return _round_money(tax_before_cess)


def calculate_tax(
    profile: UserTaxProfile | dict,
    regime: TaxRegime | str | None = None,
) -> TaxCalculationResult:
    profile_obj = _coerce_profile(profile)
    selected_regime = TaxRegime(regime) if regime else profile_obj.tax_regime
    if selected_regime == TaxRegime.UNKNOWN:
        selected_regime = TaxRegime.NEW

    assumptions: list[str] = []
    warnings: list[str] = []

    income_breakdown, initial_deductions = _compute_gross_total_income(
        profile_obj,
        selected_regime,
        assumptions,
        warnings,
    )
    gross_total_income = _round_money(sum(income_breakdown.values()))

    deductions = initial_deductions + _compute_deductions(
        profile_obj,
        selected_regime,
        assumptions,
        warnings,
    )
    total_deductions = _round_money(sum(deduction.amount for deduction in deductions))
    taxable_income = _round_money(max(gross_total_income - total_deductions, 0.0))

    slabs = NEW_REGIME_SLABS if selected_regime == TaxRegime.NEW else _old_regime_slabs(profile_obj)
    slab_tax = _slab_tax(taxable_income, slabs)

    rebate = 0.0
    if selected_regime == TaxRegime.OLD and taxable_income <= 500000:
        rebate = min(slab_tax, 12500.0)
    elif selected_regime == TaxRegime.NEW and taxable_income <= 1200000:
        rebate = min(slab_tax, 60000.0)

    tax_after_rebate = _round_money(max(slab_tax - rebate, 0.0))
    surcharge_rate = _surcharge_rate(taxable_income, selected_regime)
    surcharge = _round_money(tax_after_rebate * surcharge_rate)

    tax_before_cess = _apply_marginal_relief(
        taxable_income,
        tax_after_rebate + surcharge,
        selected_regime,
        slabs,
    )
    surcharge = _round_money(max(tax_before_cess - tax_after_rebate, 0.0))

    cess = _round_money(tax_before_cess * 0.04)
    total_tax = _round_money(tax_before_cess + cess)

    if profile_obj.has_business_or_profession_income() and selected_regime == TaxRegime.OLD:
        assumptions.append(
            "Old-regime computation assumes the taxpayer is eligible to opt out of the default new regime, including Form 10-IEA compliance where required."
        )

    if profile_obj.age is None and selected_regime == TaxRegime.OLD:
        assumptions.append(
            "Age was not provided, so old-regime slabs were calculated using the under-60 slab."
        )

    return TaxCalculationResult(
        user_id=profile_obj.user_id,
        regime=selected_regime,
        financial_year=profile_obj.financial_year,
        assessment_year=profile_obj.assessment_year,
        income_breakdown=income_breakdown,
        gross_total_income=gross_total_income,
        total_deductions=total_deductions,
        taxable_income=taxable_income,
        slab_tax=slab_tax,
        tax_before_rebate=slab_tax,
        rebate=_round_money(rebate),
        surcharge=surcharge,
        cess=cess,
        total_tax=total_tax,
        applied_deductions=deductions,
        assumptions=assumptions,
        warnings=warnings,
    )


def compare_old_vs_new_regime(
    profile: UserTaxProfile | dict,
) -> RegimeComparisonResult:
    profile_obj = _coerce_profile(profile)
    old_result = calculate_tax(profile_obj, regime=TaxRegime.OLD)
    new_result = calculate_tax(profile_obj, regime=TaxRegime.NEW)

    recommended = TaxRegime.OLD if old_result.total_tax < new_result.total_tax else TaxRegime.NEW
    tax_saving = _round_money(abs(old_result.total_tax - new_result.total_tax))

    return RegimeComparisonResult(
        recommended_regime=recommended,
        old_regime=old_result,
        new_regime=new_result,
        tax_saving=tax_saving,
    )


def suggest_applicable_deductions(
    profile: UserTaxProfile | dict,
) -> list[ApplicableDeduction]:
    profile_obj = _coerce_profile(profile)
    profession = profile_obj.inferred_profession_type()
    recommendations: list[ApplicableDeduction] = [
        ApplicableDeduction(
            section="16(ia)",
            label="Standard deduction on salary/pension",
            regimes=["old", "new"],
            max_amount=DEFAULT_STANDARD_DEDUCTION,
            likely_applicable=profile_obj.salary_income > 0 or profile_obj.pension_income > 0,
            reason="Salary and pension income can usually use the standard deduction.",
        ),
        ApplicableDeduction(
            section="80CCD(2)",
            label="Employer NPS contribution",
            regimes=["old", "new"],
            likely_applicable=profile_obj.salary_income > 0,
            reason="Useful when the employer contributes to NPS.",
            notes=[
                "Under the new regime this is one of the key remaining deductions.",
            ],
        ),
        ApplicableDeduction(
            section="80C/80CCC/80CCD(1)",
            label="PF, LIC, tuition fee, home-loan principal, and similar investments",
            regimes=["old"],
            max_amount=150000.0,
            likely_applicable=profession in (ProfessionType.SALARIED, ProfessionType.MIXED),
            reason="Common old-regime investment deduction bucket.",
        ),
        ApplicableDeduction(
            section="80CCD(1B)",
            label="Additional self-contribution to NPS",
            regimes=["old"],
            max_amount=50000.0,
            likely_applicable=True,
            reason="Often used after the main 80C limit is exhausted.",
        ),
        ApplicableDeduction(
            section="80D",
            label="Health insurance premium",
            regimes=["old"],
            likely_applicable=True,
            reason="Common old-regime deduction for self/family and parents.",
        ),
        ApplicableDeduction(
            section="24(b)",
            label="Housing-loan interest on self-occupied property",
            regimes=["old"],
            max_amount=200000.0,
            likely_applicable=profile_obj.house_property_interest_self_occupied > 0,
            reason="Available in the old regime for self-occupied property within the cap.",
        ),
        ApplicableDeduction(
            section="24(b)",
            label="Housing-loan interest on let-out property",
            regimes=["new", "old"],
            likely_applicable=profile_obj.rental_income > 0,
            reason="Let-out house-property rules may still allow interest treatment, subject to set-off rules.",
            notes=[
                "This repo does not fully model house-property loss set-off yet.",
            ],
        ),
        ApplicableDeduction(
            section="80E",
            label="Education-loan interest",
            regimes=["old"],
            likely_applicable=profile_obj.section_80e_interest > 0,
            reason="Useful when the taxpayer has an eligible education loan.",
        ),
        ApplicableDeduction(
            section="80TTA/80TTB",
            label="Interest deduction on savings/deposits",
            regimes=["old"],
            likely_applicable=profile_obj.interest_income > 0
            or profile_obj.savings_interest_income > 0
            or profile_obj.fixed_deposit_interest_income > 0,
            reason="Old regime may allow deductions on eligible savings or deposit interest.",
        ),
        ApplicableDeduction(
            section="80CCH",
            label="Agniveer Corpus contribution",
            regimes=["old", "new"],
            likely_applicable=profile_obj.section_80cch_contribution > 0,
            reason="Applies where the user is enrolled in the Agnipath scheme.",
        ),
        ApplicableDeduction(
            section="80G",
            label="Eligible donations",
            regimes=["old"],
            likely_applicable=profile_obj.section_80g_donations > 0,
            reason="Can apply under the old regime, but the exact deduction depends on the donee category.",
            notes=[
                "This deduction is not auto-calculated yet because qualifying limits vary.",
            ],
        ),
    ]

    return recommendations


def list_missing_information(profile: UserTaxProfile | dict) -> list[str]:
    profile_obj = _coerce_profile(profile)
    missing: list[str] = []

    if profile_obj.tax_regime == TaxRegime.UNKNOWN:
        missing.append("tax_regime")

    if profile_obj.profession_type == ProfessionType.UNKNOWN and profile_obj.inferred_profession_type() == ProfessionType.UNKNOWN:
        missing.append("profession_type")

    if profile_obj.salary_income > 0 and profile_obj.age is None:
        missing.append("age")

    if profile_obj.freelance_receipts > 0 and not profile_obj.use_presumptive_profession and profile_obj.freelance_expenses == 0:
        missing.append("freelance_expenses_or_presumptive_choice")

    if profile_obj.business_receipts > 0 and not profile_obj.use_presumptive_business and profile_obj.business_expenses == 0:
        missing.append("business_expenses_or_presumptive_choice")

    if profile_obj.interest_income > 0 and profile_obj.savings_interest_income == 0 and profile_obj.fixed_deposit_interest_income == 0:
        missing.append("interest_split_between_savings_and_deposits")

    if profile_obj.house_property_interest_self_occupied > 0 and profile_obj.tax_regime == TaxRegime.UNKNOWN:
        missing.append("regime_for_house_property_deduction")

    return sorted(set(missing))


def explain_tax_breakdown(result: TaxCalculationResult) -> str:
    deduction_lines = [
        f"- {item.section}: {item.label} = Rs. {item.amount:,.2f}"
        for item in result.applied_deductions
    ]
    assumptions = [f"- {item}" for item in result.assumptions]
    warnings = [f"- {item}" for item in result.warnings]

    parts = [
        f"Regime: {result.regime.value}",
        f"Gross total income: Rs. {result.gross_total_income:,.2f}",
        f"Total deductions applied: Rs. {result.total_deductions:,.2f}",
        f"Taxable income: Rs. {result.taxable_income:,.2f}",
        f"Slab tax before rebate: Rs. {result.tax_before_rebate:,.2f}",
        f"Rebate: Rs. {result.rebate:,.2f}",
        f"Surcharge: Rs. {result.surcharge:,.2f}",
        f"Cess: Rs. {result.cess:,.2f}",
        f"Total tax: Rs. {result.total_tax:,.2f}",
    ]

    if deduction_lines:
        parts.append("Applied deductions:")
        parts.extend(deduction_lines)

    if assumptions:
        parts.append("Assumptions:")
        parts.extend(assumptions)

    if warnings:
        parts.append("Warnings:")
        parts.extend(warnings)

    return "\n".join(parts)
