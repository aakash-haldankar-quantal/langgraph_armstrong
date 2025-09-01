from datetime import datetime
from typing import Dict, List, Any, Tuple

# ────────────────────────── Time helpers ──────────────────────────
current_year = datetime.today().year

# ───────────────────── Financial math helpers ─────────────────────
def lumpsum_required(corpus_needed: float, r_annual: float, years: int) -> float:
    """Present value needed today to reach corpus_needed in `years` years at rate r_annual."""
    if years <= 0:
        return corpus_needed
    return corpus_needed / ((1 + r_annual) ** years)

def sip_required(corpus_needed: float, r_annual: float, n_months: int) -> float:
    """Monthly SIP needed to reach corpus_needed with monthly compounding at r_annual."""
    if n_months <= 0:
        return float('inf')
    if r_annual < 0:
        # Fallback: linear divide when negative rate passed accidentally.
        return corpus_needed / n_months
    r = r_annual / 12.0
    if r == 0:
        return corpus_needed / n_months
    denom = (((1 + r) ** n_months - 1) / r) * (1 + r)
    if denom == 0:
        return float('inf')
    return corpus_needed / denom

def fv_sip(P: float, r_annual: float, n_months: int) -> float:
    """Future value at month-end contributions P over n_months."""
    if n_months <= 0 or P <= 0:
        return 0.0
    r = r_annual / 12.0
    if r == 0:
        return P * n_months
    return P * (((1 + r) ** n_months - 1) / r) * (1 + r)

def future_value_lumpsum(principal: float, r_annual: float, years: float, compounding_frequency: int = 1) -> float:
    """Future value of a lumpsum with compounding."""
    if years <= 0 or principal <= 0:
        return principal
    n = max(1, compounding_frequency)
    rate_per = r_annual / n
    total_n = n * years
    return principal * ((1 + rate_per) ** total_n)

# ─────────────────────── freed_sip utilities ───────────────────────
def add_to_schedule(schedule: Dict[int, float], year: int, amount: float) -> None:
    if amount <= 0:
        return
    schedule[year] = schedule.get(year, 0.0) + amount

def reduce_from_schedule(schedule: Dict[int, float], year: int, amount: float) -> float:
    """Reduce up to `amount` from schedule[year]; returns actually reduced."""
    if amount <= 0:
        return 0.0
    avail = schedule.get(year, 0.0)
    used = min(avail, amount)
    if used > 0:
        new_val = avail - used
        if new_val <= 1e-9:
            schedule.pop(year, None)
        else:
            schedule[year] = new_val
    return used

def move_freed_monthly(schedule: Dict[int, float], from_year: int, to_year: int, monthly: float) -> None:
    """Move `monthly` from being free at from_year to becoming free at to_year (after commitment ends)."""
    if monthly <= 0:
        return
    # Deduct from source (only up to what's available).
    used = reduce_from_schedule(schedule, from_year, monthly)
    # Add to destination (what will be freed after the new goal ends).
    add_to_schedule(schedule, to_year, used)

# ─────────────────────── allocation primitives ───────────────────────
def apply_freed_sip_to_goal(
    goal_gap: float,
    freed_sip: Dict[int, float],
    r_annual: float,
    start_year: int,
    end_year: int,
    funded_from: List[Dict[str, Any]]
) -> Tuple[float, Dict[int, float]]:
    """
    Use freed SIPs that start before end_year to reduce goal_gap.
    Each freed monthly amount starting at Y contributes until end_year.
    We may use part/all of a freed monthly. We then move what we used to free at end_year.
    """
    if goal_gap <= 0:
        return goal_gap, freed_sip

    years_sorted = sorted([y for y in freed_sip.keys() if y < end_year])
    # Work on a copy; we'll mutate safely and only return final dict.
    sched = dict(freed_sip)

    for y in years_sorted:
        if goal_gap <= 0:
            break
        months = max(0, (end_year - y) * 12)
        if months == 0:
            continue

        monthly_available = sched.get(y, 0.0)
        if monthly_available <= 1e-9:
            continue

        # Monthly SIP needed if we only started at year y for the remaining gap:
        monthly_needed = sip_required(goal_gap, r_annual, months)

        if monthly_available >= monthly_needed:
            # Use only the part needed
            used = monthly_needed
            # Reduce at y; add to end_year
            reduce_from_schedule(sched, y, used)
            add_to_schedule(sched, end_year, used)

            # Record and close the gap
            fv_contrib = fv_sip(used, r_annual, months)
            funded_from.append({
                "type": "freed_sip",
                "monthly": used,
                "from_year": y,
                "to_year": end_year,
                "months": months,
                "fv_contribution": fv_contrib
            })
            goal_gap -= fv_contrib
            break  # fully funded

        else:
            # Use the entire freed monthly_available
            used = monthly_available
            reduce_from_schedule(sched, y, used)
            add_to_schedule(sched, end_year, used)

            fv_contrib = fv_sip(used, r_annual, months)
            funded_from.append({
                "type": "freed_sip",
                "monthly": used,
                "from_year": y,
                "to_year": end_year,
                "months": months,
                "fv_contribution": fv_contrib
            })
            goal_gap -= fv_contrib
            # continue; still gap left

    return goal_gap, sched

def apply_surplus_sip_to_goal(
    goal_gap: float,
    monthly_surplus: float,
    r_annual: float,
    start_year: int,
    end_year: int,
    funded_from: List[Dict[str, Any]],
    freed_sip: Dict[int, float]
) -> Tuple[float, float, Dict[int, float]]:
    """
    Use monthly_surplus toward the goal until end_year. Whatever is used becomes freed again at end_year.
    """
    if goal_gap <= 0 or monthly_surplus <= 1e-9:
        return goal_gap, monthly_surplus, freed_sip

    months = max(0, (end_year - start_year) * 12)
    if months == 0:
        return goal_gap, monthly_surplus, freed_sip

    monthly_needed = sip_required(goal_gap, r_annual, months)

    if monthly_surplus >= monthly_needed:
        used = monthly_needed
        fv_contrib = fv_sip(used, r_annual, months)
        funded_from.append({
            "type": "sip_from_surplus",
            "monthly": used,
            "from_year": start_year,
            "to_year": end_year,
            "months": months,
            "fv_contribution": fv_contrib
        })
        # Commit this monthly until end_year → becomes free then
        add_to_schedule(freed_sip, end_year, used)
        monthly_surplus -= used
        goal_gap -= fv_contrib
    else:
        # Use all available surplus
        used = monthly_surplus
        fv_contrib = fv_sip(used, r_annual, months)
        funded_from.append({
            "type": "sip_from_partial_surplus",
            "monthly": used,
            "from_year": start_year,
            "to_year": end_year,
            "months": months,
            "fv_contribution": fv_contrib
        })
        add_to_schedule(freed_sip, end_year, used)
        monthly_surplus = 0.0
        goal_gap -= fv_contrib

    return goal_gap, monthly_surplus, freed_sip

def apply_lumpsum_to_goal(
    goal_gap: float,
    liquid_pool: float,
    r_annual: float,
    start_year: int,
    end_year: int,
    funded_from: List[Dict[str, Any]]
) -> Tuple[float, float]:
    """
    Try to close remaining gap with lumpsum today. If insufficient, use all liquid to reduce gap.
    Records principal used and its FV contribution.
    """
    if goal_gap <= 0 or liquid_pool <= 1e-9:
        return goal_gap, liquid_pool

    years = max(0, end_year - start_year)

    # PV needed today to exactly close the goal_gap
    pv_needed = lumpsum_required(goal_gap, r_annual, years)

    if pv_needed <= liquid_pool:
        # Fully close
        fv_contrib = future_value_lumpsum(pv_needed, r_annual, years)
        funded_from.append({
            "type": "lumpsum_from_liquid",
            "principal_used_today": pv_needed,
            "from_year": start_year,
            "to_year": end_year,
            "years": years,
            "fv_contribution": fv_contrib
        })
        liquid_pool -= pv_needed
        goal_gap -= fv_contrib
        return goal_gap, liquid_pool
    else:
        # Use everything; reduce the gap by its FV
        principal_used = liquid_pool
        fv_contrib = future_value_lumpsum(principal_used, r_annual, years)
        funded_from.append({
            "type": "lumpsum_from_liquid_partial",
            "principal_used_today": principal_used,
            "from_year": start_year,
            "to_year": end_year,
            "years": years,
            "fv_contribution": fv_contrib
        })
        liquid_pool = 0.0
        goal_gap -= fv_contrib
        return goal_gap, liquid_pool

# ───────────────────────── Master planner ─────────────────────────
def plan_goals(
    updated_goals: List[Dict[str, Any]],
    monthly_surplus_init: float,
    liquid_pool_init: float,
    freed_sip_init: Dict[int, float],
    r_annual: float = 0.08,
    near_term_years: int = 2,
    max_postpone: int = 10
) -> Dict[str, Any]:
    """
    Greedy allocator per goal (in given order), allowing up to `max_postpone` years of postponement.
    Strategy:
      - If horizon <= near_term_years, try lumpsum-first (then freed SIP, then surplus SIP, then more lumpsum).
      - Else (longer horizon), try freed SIP -> surplus SIP -> lumpsum.
      - If still gap, postpone by +1 year (up to max_postpone).
    Outputs final states and per-goal funding trails with a correct freed_sip schedule.
    """
    monthly_surplus = float(monthly_surplus_init)
    liquid_pool = float(liquid_pool_init)
    freed_sip = dict(freed_sip_init)  # year -> monthly freed at that year

    results = []
    for g in updated_goals:
        goal = {
            "goal_name": g["goal_name"],
            "target_year": int(g["target_year"]),
            "corpus_needed": float(g["corpus_needed"]),
            "corpus_gap": float(g["corpus_gap"]),
            "funded_from": []
        }

        if goal["corpus_gap"] <= 1e-9:
            results.append(goal)
            continue

        achieved = False
        for postpone in range(0, max_postpone + 1):
            end_year = goal["target_year"] + postpone
            if end_year < current_year:
                # Cannot fund goals in the past
                goal["funded_from"].append({
                    "type": "in_past",
                    "note": f"End year {end_year} < current_year {current_year}"
                })
                continue

            gap = goal["corpus_gap"]
            funded_from = []
            local_freed = dict(freed_sip)     # work on copies; commit only if achieved
            local_surplus = monthly_surplus
            local_liquid = liquid_pool

            short_horizon = (end_year - current_year) <= near_term_years

            # Order of operations
            if short_horizon:
                # 1) Lumpsum first
                gap, local_liquid = apply_lumpsum_to_goal(
                    gap, local_liquid, r_annual, current_year, end_year, funded_from
                )
                if gap > 1e-9:
                    # 2) Freed SIP
                    gap, local_freed = apply_freed_sip_to_goal(
                        gap, local_freed, r_annual, current_year, end_year, funded_from
                    )
                if gap > 1e-9:
                    # 3) Monthly surplus SIP
                    gap, local_surplus, local_freed = apply_surplus_sip_to_goal(
                        gap, local_surplus, r_annual, current_year, end_year, funded_from, local_freed
                    )
                if gap > 1e-9:
                    # 4) Try lumpsum again (maybe gap smaller now)
                    gap, local_liquid = apply_lumpsum_to_goal(
                        gap, local_liquid, r_annual, current_year, end_year, funded_from
                    )
            else:
                # Longer horizon: prioritize SIP flows first
                # 1) Freed SIP
                gap, local_freed = apply_freed_sip_to_goal(
                    gap, local_freed, r_annual, current_year, end_year, funded_from
                )
                # 2) Surplus SIP
                if gap > 1e-9:
                    gap, local_surplus, local_freed = apply_surplus_sip_to_goal(
                        gap, local_surplus, r_annual, current_year, end_year, funded_from, local_freed
                    )
                # 3) Lumpsum if still needed
                if gap > 1e-9:
                    gap, local_liquid = apply_lumpsum_to_goal(
                        gap, local_liquid, r_annual, current_year, end_year, funded_from
                    )

            if gap <= 1e-9:
                # Commit local states since goal is achieved for this postponement
                goal["corpus_gap"] = 0.0
                if postpone > 0:
                    # mark postponed info once
                    funded_from.insert(0, {
                        "type": "postponed",
                        "postponed_years": postpone,
                        "from_year": goal["target_year"],
                        "to_year": end_year
                    })
                goal["funded_from"].extend(funded_from)

                # Update global state
                monthly_surplus = local_surplus
                liquid_pool = local_liquid
                freed_sip = local_freed
                achieved = True
                break
            # else: try next postponement year

        if not achieved:
            # Keep what we could do in the last attempt (no state change), just record failure note
            goal["funded_from"].append({
                "type": "unfunded",
                "note": f"Insufficient resources even after postponing up to {max_postpone} years."
            })
            # corpus_gap remains > 0

        results.append(goal)

    return {
        "goals": results,
        "ending_monthly_surplus": monthly_surplus,
        "ending_liquid_pool": liquid_pool,
        "ending_freed_sip_schedule": freed_sip
    }

# ───────────────────────── Example usage ─────────────────────────
if __name__ == "__main__":
    updated_goals = [
        {'goal_name': 'retirement', 'corpus_needed': 14697458.230000004, 'corpus_gap': 14697458.230000004, 'target_year': 2045, 'funded_from': []},
        {'goal_name': 'Aarav Mehta under_graduation', 'target_year': 2030, 'corpus_needed': 4599498.77, 'corpus_gap': 4599498.77, 'funded_from': []},
        {'goal_name': 'Raghav Mehta under_graduation', 'target_year': 2031, 'corpus_needed': 3996976.34, 'corpus_gap': 3996976.34, 'funded_from': []},
        {'goal_name': 'Aarav Mehta post_graduation', 'target_year': 2034, 'corpus_needed': 3982490.24, 'corpus_gap': 3982490.24, 'funded_from': []},
        {'goal_name': 'Raghav Mehta post_graduation', 'target_year': 2035, 'corpus_needed': 0.0, 'corpus_gap': 0.0, 'funded_from': []},
        {'goal_name': 'House Renovation', 'target_year': 2028, 'corpus_needed': 1520324.0, 'corpus_gap': 1520324.0, 'funded_from': []},
        {'goal_name': 'Bike', 'target_year': 2030, 'corpus_needed': 254030.89440000002, 'corpus_gap': 254030.89440000002, 'funded_from': []},
        {'goal_name': 'Car', 'target_year': 2045, 'corpus_needed': 5326838.700801875, 'corpus_gap': 5326838.700801875, 'funded_from': []}
    ]

    monthly_surplus_o = 105000.0
    liquid_pool_o = 2550000.0
    freed_sip_o = {}  # {year:int -> monthly_amount:float}

    plan = plan_goals(
        updated_goals=updated_goals,
        monthly_surplus_init=monthly_surplus_o,
        liquid_pool_init=liquid_pool_o,
        freed_sip_init=freed_sip_o,
        r_annual=0.08,
        near_term_years=2,
        max_postpone=10
    )

    # Example: print compact summary
    # import json
    # print(json.dumps(plan, indent=2))