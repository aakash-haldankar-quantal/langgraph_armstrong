from langgraph.graph import StateGraph, START, END
from langchain_openai import AzureChatOpenAI
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import TypedDict, Literal, Annotated, List, Dict
import os
import operator
from dotenv import load_dotenv
from datetime import datetime
from datetime import date
import json
from client_data_updated import (future_value, calculate_required_sip, ulip_future_value, epf_future_value, 
                                 ppf_future_value, nps_future_value, client_data) 

from Armstrong_goal_allocation import (lumpsum_required, sip_required, future_value_lumpsum, fv_sip,
                                       add_to_schedule, reduce_from_schedule, move_freed_monthly, 
                                       apply_freed_sip_to_goal, apply_surplus_sip_to_goal, apply_lumpsum_to_goal)

from datetime import datetime
from collections import defaultdict
from math import ceil

#print('hello')

load_dotenv()

AZURE_API_KEY=os.getenv('AZURE_API_KEY')
AZURE_API_BASE=os.getenv('AZURE_API_BASE')
AZURE_API_VERSION=os.getenv('AZURE_API_VERSION')
AZURE_DEPLOYMENT_NAME=os.getenv('AZURE_DEPLOYMENT_NAME')

GROQ_API_KEY=os.getenv('GROQ_API_KEY')
# model=ChatOpenAI(model='openai-4o')

llm_groq=ChatGroq(model='qwen/qwen3-32b', api_key=GROQ_API_KEY, temperature=0.5)

llm_azure = AzureChatOpenAI(
    api_key=AZURE_API_KEY,  # AZURE_API_KEY
    azure_endpoint=AZURE_API_BASE,  # AZURE_API_BASE
    api_version=AZURE_API_VERSION,  # AZURE_API_VERSION
    deployment_name=AZURE_DEPLOYMENT_NAME,  # AZURE_DEPLOYMENT_NAME
    temperature=0  # Optional
)

class RiskSchema(BaseModel):

    risk_assessment: Literal['Low', 'Medium', 'Medium to High', 'Medium to Low'] = Field(
        description='Marks risk appetite of the user')
    reason_of_risk_assessment: str = Field(
        description='Reason for the risk assessment chosen based on the conditions provided')
    
risk_llm=llm_azure.with_structured_output(RiskSchema) 

class ClientState(TypedDict):

    client_data: json                         # input
    freed_funds : Dict[int, float]            # currently taken as input
    risk_appetite : dict
    children_education_planning: dict 
    required_retirement_corpus: json
    retirement_schemes_fv: json
    goals: list 
    retirement_assets: list
    liquid_assets : list
    fixed_assets : list
    liquid_pool : float  
    fixed_asset_pool : float
    goal_funding : dict
    asset_percentages : dict
    

##################################################################################### Nodes ##################################################################################################################################

def calculate_age(state: ClientState): # calculates ages of all the individual mentioed in the client data
    """ 
     calculate_age(): calculates current age of individuals mentioned in the json object.
     input argument: client data in json format
     output: client data with populated current ages of all the mentioned individuals
    """
    client_data=state['client_data']
    current_date = date.today() 
    if client_data['client_data']['date_of_birth']:
        client_age=current_date.year-datetime.strptime(client_data['client_data']['date_of_birth'], '%Y-%m-%d').year
        client_data['client_data']['client_age']=client_age
     
    if client_data['client_data']['spouse_dob']:
        spouse_age=current_date.year-datetime.strptime(client_data['client_data']['spouse_dob'], '%Y-%m-%d').year
        client_data['client_data']['spouse_age']=spouse_age

    if client_data['client_data']['if_any_kids']:
        for index, child_info in enumerate(client_data['client_data']['children']):
            child_age=current_date.year-datetime.strptime(child_info['child_dob'], '%Y-%m-%d').year
            client_data['client_data']['children'][index]['child_age']=child_age
            
    return {'client_data': client_data}


def goals_future_value(state: ClientState):  # calculate the actual goal gap and allocating surplus for all the financial goals mentioned 
    """
    function:  
    input argument: 
    output argument: 
    """
    client_data=state['client_data']
    goals=client_data['financial_goals']
    current_date=date.today()
    for goal in goals:
        if goal['target_year']-current_date.year >= 0:
            future_value_of_saved=future_value(goal['amount_saved_for_goal'], goal['target_year']-current_date.year, 0.10 )
            capital_required_at_target_year=future_value(goal['capital_required_today'], goal['target_year']-current_date.year, 0.06  )
            goal['future_value_of_saved_amount']=future_value_of_saved
            goal['capital_required_at_target_year']=capital_required_at_target_year
    
    goals.sort(key=lambda x:x['target_year'])

    surplus=0.0
    for goal in goals:
        
        goal_gap=goal['capital_required_at_target_year']-goal['future_value_of_saved_amount']
        if goal_gap>0:
            if surplus>0: 
                #print(f"considering the surplus collected from previous goals: {surplus}, the new gap is: {goal_gap}")
                goal_gap=goal_gap-surplus 
                goal['goal_gap']=goal_gap
                goal['funded_from'].append({'surplus_left_from_previous_goal': surplus})
                # sip_required=calculate_required_sip(goal_gap,0.09, goal['target_year']-current_date.year)
                # goal['sip_required']=sip_required
                # goal['sip_month']=int((goal['target_year']-current_date.year)*12)
            else: 
                goal['goal_gap']=goal_gap
                # sip_required=calculate_required_sip(goal_gap,0.09, goal['target_year']-current_date.year)
                # goal['sip_required']=sip_required
                # goal['sip_month']=int((goal['target_year']-current_date.year)*12)
        elif goal_gap==0:
            goal['goal_gap']=0
            # goal['sip_required']=0
            # goal['sip_month']=int((goal['target_year']-current_date.year)*12)
        elif goal_gap<0: 
            goal['goal_gap']=0 
            # goal['sip_required']=0  
            # goal['sip_month']=int((goal['target_year']-current_date.year)*12)
            surplus+= -1*goal_gap  

    return {'client_data': client_data}


def calculate_education_funding(state: ClientState):
    """
    Analyzes and calculates the funding for the client's children's education goals.

    This function processes each child's undergraduate (UG) and postgraduate (PG)
    education goals, calculates future costs and the future value of existing
    investments, and determines any funding gaps or surpluses. Surpluses from
    earlier goals are utilized for later ones. Crucially, it tracks used investment
    schemes to ensure they are not double-counted for multiple goals. Finally, it
    computes the required monthly SIP to bridge any remaining gaps.

    Args:
        client_data (dict): A dictionary containing the client's financial and personal data.

    Returns:
        dict: A dictionary containing the original client data updated with a detailed
              education planning summary. The summary includes the future cost of
              each goal, the value of allocated funds, any gaps or surpluses,
              and the required monthly investment to meet the goals.
    """
    client_data=state['client_data']
    # --- Helper Functions ---
    def calculate_future_value(present_value, annual_rate, years):
        """Calculates the future value of a lump sum investment."""
        if years < 0:
            return present_value
        return present_value * ((1 + annual_rate) ** years)

    def calculate_sip_future_value(monthly_sip, annual_rate, years):
        """Calculates the future value of a Systematic Investment Plan (SIP)."""
        if years <= 0:
            return 0.0
        monthly_rate = annual_rate / 12
        months = int(years * 12)
        if monthly_rate == 0:
            return monthly_sip * months
        return monthly_sip * (((1 + monthly_rate) ** months - 1) / monthly_rate) * (1 + monthly_rate)

    def calculate_required_sip(target_value, annual_rate, years):
        """Calculates the monthly SIP required to reach a target future value."""
        if years <= 0 or target_value <= 0:
            return 0.0
        monthly_rate = annual_rate / 12
        months = int(years * 12)
        if monthly_rate == 0:
            return target_value / months
        return (target_value * monthly_rate) / (((1 + monthly_rate) ** months - 1))

    # --- Main Logic ---
    
    today = date.today()
    education_goals = []
    
    # 1. Consolidate all education goals into a single list
    for child in client_data['education_planning']:
        child_dob = None
        for c in client_data['client_data']['children']:
            if c['child_name'] == child['name_of_kid']:
                child_dob = datetime.strptime(c['child_dob'], '%Y-%m-%d').date()
                break
        
        if not child_dob:
            continue

        # Undergraduate (UG) Goal
        ug_target_year = child_dob.year + 18
        ug_years_to_goal = ug_target_year - today.year
        education_goals.append({
            "name": child['name_of_kid'],
            "type": "under_graduation",
            "target_year": ug_target_year,
            "years_to_goal": ug_years_to_goal,
            "current_cost": child['current_fees_of_graduation'],
            "allocated_funds": child.get('fund_allocated_for_graduation'),
            "schemes": child.get('scheme_for_education', [])
        })

        # Postgraduate (PG) Goal
        pg_target_year = child_dob.year + 22
        pg_years_to_goal = pg_target_year - today.year
        education_goals.append({
            "name": child['name_of_kid'],
            "type": "post_graduation",
            "target_year": pg_target_year,
            "years_to_goal": pg_years_to_goal,
            "current_cost": child['current_fees_of_post_graduation'],
            "allocated_funds": child.get('fund_allocated_for_post_graduation'),
            "schemes": child.get('scheme_for_education', [])
        })
        
    # 2. Sort goals chronologically by target year
    education_goals.sort(key=lambda x: x['target_year'])
    
    surplus_pool = 0.0
    # NEW: Initialize a set to track schemes that have been allocated to a goal.
    used_schemes = set()
    
    # 3. Process each goal in chronological order
    for goal in education_goals:
        # Calculate future cost of education (assuming 6% inflation)
        future_cost = calculate_future_value(goal['current_cost'], 0.06, goal['years_to_goal'])
        goal['future_cost'] = round(future_cost, 2)
        
        total_future_corpus = 0.0
        
        # Calculate FV of funds already allocated (assuming 9% growth)
        if goal['allocated_funds']:
            fv_allocated = calculate_future_value(goal['allocated_funds'], 0.09, goal['years_to_goal'])
            total_future_corpus += fv_allocated
            goal['fv_of_allocated_funds'] = round(fv_allocated, 2)
        else:
            goal['fv_of_allocated_funds'] = 0.0

        # Calculate FV of dedicated education schemes
        goal['fv_of_schemes'] = 0.0
        for scheme in goal['schemes']:
            # NEW: Check if the scheme has already been used for a prior goal.
            if scheme['scheme_name'] in used_schemes:
                continue # Skip this scheme as it's already allocated.

            scheme_end_date = datetime.strptime(scheme['end_date'], '%Y-%m-%d').date()
            # Only use scheme if it matures on or before the goal's target year
            if scheme_end_date.year <= goal['target_year']:
                years_in_scheme = (scheme_end_date - datetime.strptime(scheme['start_date'], '%Y-%m-%d').date()).days / 365.25
                fv_scheme = calculate_sip_future_value(scheme['monthly_investment'], scheme['interest_rate'], years_in_scheme)
                
                # If scheme matures before goal, grow the lump sum until the goal year
                years_post_maturity = goal['target_year'] - scheme_end_date.year
                if years_post_maturity > 0:
                    fv_scheme = calculate_future_value(fv_scheme, 0.09, years_post_maturity)
                
                total_future_corpus += fv_scheme
                goal['fv_of_schemes'] += round(fv_scheme, 2)
                
                # NEW: Mark this scheme as used so it's not double-counted.
                used_schemes.add(scheme['scheme_name'])

        goal['total_future_corpus'] = round(total_future_corpus, 2)
        
        # Determine the initial funding gap
        initial_gap = future_cost - total_future_corpus
        
        # Utilize surplus from previous goals
        if initial_gap > 0 and surplus_pool > 0:
            used_surplus = min(initial_gap, surplus_pool)
            initial_gap -= used_surplus
            surplus_pool -= used_surplus
            goal['surplus_utilized'] = round(used_surplus, 2)
        else:
            goal['surplus_utilized'] = 0.0
            
        # Final calculation for gap or surplus
        if initial_gap > 0:
            goal['final_gap'] = round(initial_gap, 2)
            goal['surplus_generated'] = 0.0
            #Calculate required monthly SIP to cover the remaining gap (assuming 12% return)
            #required_sip = calculate_required_sip(initial_gap, 0.12, goal['years_to_goal'])
            #goal['required_monthly_investment'] = round(required_sip, 2)
        else: 
            goal['final_gap'] = 0.0
            # Add the surplus to the pool for next goals
            surplus = -initial_gap
            surplus_pool += surplus
            goal['surplus_generated'] = round(surplus, 2)
            #goal['required_monthly_investment'] = 0.0
            
    # 4. Attach the detailed plan to the original client data for a complete report
    client_data['education_planning_summary'] = education_goals
    
    return {'client_data': client_data, 'children_education_planning': education_goals}


def calculate_retirement_corpus(state: ClientState):
    """
    Calculates the required retirement corpus using two methods:
    1. Standard Method: Flat expenses throughout retirement
    2. Segmented Cash Flow Method: Lifestyle-based phases with varying expenses
    
    Args:
        client_data (dict): Client's financial and personal data
        retirement_age (int): Age at which client plans to retire (default: 60)
        life_expectancy (int): Expected life expectancy (default: 85)
        inflation_rate (float): Annual inflation rate (default: 6%)
    
    Returns:
        dict: Detailed retirement corpus calculation with both methods
    """
    client_data=state['client_data']
    life_expectancy=85
    inflation_rate=0.06
    # Helper Functions
    def calculate_future_value(present_value, annual_rate, years):
        """Calculate future value with compound growth"""
        if years <= 0:
            return present_value
        return present_value * ((1 + annual_rate) ** years)
    
    def calculate_present_value_annuity(annual_payment, discount_rate, years):
        """Calculate present value of annuity (series of payments)"""
        if years <= 0 or discount_rate == 0:
            return annual_payment * years
        return annual_payment * (1 - (1 + discount_rate) ** -years) / discount_rate
    
    # Extract client information
    client_dob = datetime.strptime(client_data['client_data']['date_of_birth'], '%Y-%m-%d').date()
    current_age = date.today().year - client_dob.year
    monthly_expenses = client_data['investment_details']['financial_summary'][0]['monthly_expenses_excl_emis']
    annual_expenses = monthly_expenses*12 #+ (client_data['investment_details']['financial_summary'][0]["miscellaneous_kids_education_expenses_monthly"])*12 + client_data['investment_details']['financial_summary'][0]["annual_vacation_expenses"]
    retirement_age=client_data['client_data']['retirement_age']
    # Calculate years to retirement and retirement duration
    years_to_retirement = retirement_age - current_age
    retirement_duration = life_expectancy - retirement_age
    
    # Future annual expenses at retirement (adjusted for inflation)
    future_annual_expenses = calculate_future_value(annual_expenses, inflation_rate, years_to_retirement)
    
    retirement_plan = {
        "client_info": {
            "current_age": current_age,
            "retirement_age": retirement_age,
            "life_expectancy": life_expectancy,
            "years_to_retirement": years_to_retirement,
            "retirement_duration": retirement_duration,
            "current_monthly_expenses": monthly_expenses,
            "current_annual_expenses": annual_expenses,
            "future_annual_expenses_at_retirement": round(future_annual_expenses, 2)
        }
    }
    
    # METHOD A: STANDARD METHOD (Flat Expense Method)
    print("=" * 60)
    print("METHOD A: STANDARD METHOD (Flat Expense Method)")
    print("=" * 60)
    
    # Assume 4% real return during retirement (post-inflation)
    real_return_rate = 0.04
    
    # Calculate corpus needed using present value of annuity
    standard_corpus = calculate_present_value_annuity(
        future_annual_expenses, 
        real_return_rate, 
        retirement_duration
    )
    
    retirement_plan["standard_method"] = {
        "annual_expenses_throughout_retirement": round(future_annual_expenses, 2),
        "real_return_rate_assumed": real_return_rate,
        "required_corpus": round(standard_corpus, 2)
    }
    
    print(f"Annual Expenses at Retirement: ₹{future_annual_expenses:,.2f}")
    print(f"Retirement Duration: {retirement_duration} years")
    print(f"Real Return Rate (Post-Inflation): {real_return_rate*100}%")
    print(f"Required Corpus (Standard Method): ₹{standard_corpus:,.2f}")
    
    # METHOD B: SEGMENTED CASH FLOW METHOD (Lifestyle-Based Phases)
    print("\n" + "=" * 60)
    print("METHOD B: SEGMENTED CASH FLOW METHOD (Lifestyle-Based)")
    print("=" * 60)
    
    # Define retirement phases
    phases = [
        {
            "name": "Early Retirement",
            "age_range": "55-65",
            "start_age": retirement_age,
            "end_age": 65,
            "expense_multiplier": 1.1,  # 10% higher
            "description": "Active lifestyle, travel"
        },
        {
            "name": "Middle Retirement", 
            "age_range": "65-75",
            "start_age": 65,
            "end_age": 75,
            "expense_multiplier": 1.0,  # Normal expenses
            "description": "Baseline expenses"
        },
        {
            "name": "Late Retirement",
            "age_range": "75-85", 
            "start_age": 75,
            "end_age": life_expectancy,
            "expense_multiplier": 1.2,  # 20% higher
            "description": "Healthcare, support needs"
        }
    ]
    
    segmented_phases = []
    total_segmented_corpus = 0
    
    for phase in phases:
        # Adjust phase to actual retirement age if needed
        #phase_start = max(phase["start_age"], retirement_age)
        phase_start = max(phase['start_age'], retirement_age)
        #phase_end = min(phase["end_age"], life_expectancy)
        phase_end = phase["end_age"]
        phase_duration = phase_end - phase_start
        
        if phase_duration <= 0:
            continue
            
        # Calculate expenses for this phase
        phase_annual_expenses = future_annual_expenses * phase["expense_multiplier"]
        
        # Calculate years from retirement to start of this phase
        years_to_phase_start = phase_start - retirement_age
        
        # Discount the required corpus back to retirement age
        if years_to_phase_start > 0:
            # Phase starts later, so discount the annuity back
            phase_corpus_at_phase_start = calculate_present_value_annuity(
                phase_annual_expenses, real_return_rate, phase_duration
            )
            phase_corpus_at_retirement = phase_corpus_at_phase_start / ((1 + real_return_rate) ** years_to_phase_start)
        else:
            # Phase starts immediately at retirement
            phase_corpus_at_retirement = calculate_present_value_annuity(
                phase_annual_expenses, real_return_rate, phase_duration
            )
        
        phase_info = {
            "phase_name": phase["name"],
            "age_range": f"{phase_start}-{phase_end}",
            "duration_years": phase_duration,
            "expense_multiplier": phase["expense_multiplier"],
            "annual_expenses": round(phase_annual_expenses, 2),
            "corpus_required": round(phase_corpus_at_retirement, 2),
            "description": phase["description"]
        }
        
        segmented_phases.append(phase_info)
        total_segmented_corpus += phase_corpus_at_retirement
        
        print(f"\n{phase['name']} ({phase_start}-{phase_end}):")
        print(f"  Duration: {phase_duration} years")
        print(f"  Expense Level: {phase['expense_multiplier']*100:.0f}% of baseline")
        print(f"  Annual Expenses: ₹{phase_annual_expenses:,.2f}")
        print(f"  Corpus Required: ₹{phase_corpus_at_retirement:,.2f}")
    
    retirement_plan["segmented_method"] = {
        "phases": segmented_phases,
        "total_required_corpus": round(total_segmented_corpus, 2)
    }
    
    # Summary and Comparison
    print("\n" + "=" * 60)
    print("RETIREMENT CORPUS SUMMARY")
    print("=" * 60)
    print(f"Standard Method Corpus:           ₹{standard_corpus:,.2f}")
    print(f"Segmented Method Corpus:          ₹{total_segmented_corpus:,.2f}")
    difference = total_segmented_corpus - standard_corpus
    percentage_diff = (difference / standard_corpus) * 100
    print(f"Difference:                       ₹{difference:,.2f} ({percentage_diff:+.1f}%)")
    
    retirement_plan["comparison"] = {
        "standard_corpus": round(standard_corpus, 2),
        "segmented_corpus": round(total_segmented_corpus, 2),
        "difference": round(difference, 2),
        "percentage_difference": round(percentage_diff, 1)
    }
    
    # Recommended corpus (higher of the two)
    recommended_corpus = max(standard_corpus, total_segmented_corpus)
    retirement_plan["recommendation"] = {
        "recommended_corpus": round(recommended_corpus, 2),
        "method_used": "Segmented Method" if total_segmented_corpus > standard_corpus else "Standard Method",
        "rationale": "Taking the higher estimate to ensure adequate retirement funding"
    }
    
    print(f"\nRECOMMENDED RETIREMENT CORPUS: ₹{recommended_corpus:,.2f}")
    print(f"Method: {retirement_plan['recommendation']['method_used']}")
    
    return {'required_retirement_corpus': retirement_plan}


def calculate_all_retirement_investments(state: ClientState):
    """
    Accepts ANY number of schemes per category and returns:
      1. detailed per-scheme output
      2. per-category totals
      3. grand total
    """
    retirement_investments=state['client_data']['investment_details']['retirement_investments']
    current_age = state['client_data']['client_data']['client_age'] 
    retirement_age=60
    today=datetime.today()

    results          = defaultdict(list)   # per-scheme data
    category_totals  = {}                  # per-category sum
    grand_total      = 0.0

    for category, scheme_list in retirement_investments.items():
        cat_total = 0.0

        for i, sc in enumerate(scheme_list, start=1):
            # ---------- ULIP (UPDATED field names) ----------
            if category.lower() == "ulip":
                # Create end date from final year of premium + term
                start_date = sc["commencement_date_of_ulip_policy_1"]
                final_premium_year = sc["final_year_of_premium_to_be_paid"]
                term = sc["term"]
                
                # Calculate end date as final premium year + term
                end_year = final_premium_year + term
                # Assuming same month/day as start date for end date
                start_dt = datetime.strptime(start_date, "%d-%m-%Y")
                end_date = f"{start_dt.day:02d}-{start_dt.month:02d}-{end_year}"
                
                fv = ulip_future_value(
                        sc["monthly_premium_amount"],
                        start_date,
                        end_date,
                        sc["expected_rate_of_return"])
                
                # Calculate months from start to final premium year
                months = (final_premium_year - start_dt.year) * 12 + (start_dt.month - start_dt.month)
                invested = sc["monthly_premium_amount"] * months

            # ---------- EPF (field names remain same) ----------
            elif category.lower() == "epf":
                # EPF doesn't have maturity_year in new structure, use retirement age
                years_left = retirement_age - current_age
                fv = epf_future_value(
                        sc["current_value"],
                        sc["employee_employer_contribution_monthly"],
                        sc["interest_rate"],
                        years_left)
                invested = sc["current_value"] + \
                           sc["employee_employer_contribution_monthly"] * 12 * max(years_left, 0)

            # ---------- PPF (field names remain same) ----------
            elif category.lower() == "ppf":
                # PPF doesn't have lock_in_end_year, assume 15-year lock-in from current year
                years_left = 15  # Standard PPF lock-in period
                fv = ppf_future_value(
                        sc["current_value"],
                        sc["annual_contribution"],
                        sc["interest_rate"],
                        years_left)
                invested = sc["current_value"] + sc["annual_contribution"] * years_left

            # ---------- NPS (UPDATED field names) ----------
            elif category.lower() == "nps":
                # Calculate months left until maturity year
                months_left = max((sc["maturity_year"] - today.year) * 12, 0)
                fv = nps_future_value(
                        sc["current_value"],
                        sc["monthly_contribution"],
                        sc["expected_corpus_growth_rate"],
                        months_left)
                invested = sc["current_value"] + sc["monthly_contribution"] * months_left

            else:
                # Skip unknown categories gracefully
                continue

            results[category].append({
                "scheme_no"     : i,
                "future_value"  : fv,
                "total_invested": round(invested, 2)
            })
            cat_total += fv

        if cat_total:
            category_totals[category] = round(cat_total, 2)
            grand_total += cat_total

    retirement_schemes_fv={
        "schemes"         : dict(results),
        "category_totals" : category_totals,
        "grand_total"     : round(grand_total, 2)
    }

    return {"retirement_schemes_fv": retirement_schemes_fv}

  
def retirement_goal(state: ClientState):

    client_data3=state['client_data']
    required_retirement_corpus= state['required_retirement_corpus']['recommendation']['recommended_corpus']
    estimated_retirement_corpus = state['retirement_schemes_fv']['grand_total'] 
    sip_annual_rate = 0.10
    retirement_age = state['client_data']['client_data']['retirement_age']
    current_date=date.today()
    current_year=current_date.year

    if required_retirement_corpus>estimated_retirement_corpus:
        print(f"""required retirement corpus: {required_retirement_corpus} \n estimated retirement corpus: {estimated_retirement_corpus} \n
                retirement_gap: {required_retirement_corpus-estimated_retirement_corpus}
 """)
        #sip_amount=calculate_required_sip(required_retirement_corpus-estimated_retirement_corpus, sip_annual_rate, client_data['client_data']['retirement_age']-(current_date.year-int(client_data['client_data']['date_of_birth'].split('-')[0])))
        years_to_retire= retirement_age - client_data3['client_data']['client_age']
        result={}
        result['goal_name']="retirement"
        result['corpus_needed']=required_retirement_corpus-estimated_retirement_corpus
        result['corpus_gap']=required_retirement_corpus-estimated_retirement_corpus
        result['target_year']= current_year + years_to_retire 
        result['funded_from']=[]
        #result['sip_amount']=sip_amount
        #result['sip_years']=client_data['client_data']['retirement_age']-(current_date.year-int(client_data['client_data']['date_of_birth'].split('-')[0]))
        result['surplus']=0
    
    elif required_retirement_corpus==estimated_retirement_corpus:
        print(f"""
             required retirement corpus: {required_retirement_corpus} \n estimated retirement corpus: {estimated_retirement_corpus} \n
             retirement_gap = 0
""")
        result={}
        result['goal_name']="retirement"
        result['corpus_needed']=0
        result['corpus_gap']=0
        result['funded_from']=[]
        result['target_year']= current_year + years_to_retire
        #result['sip_amount']=0
        #result['sip_years']=0
        result['surplus']=0
    
    elif required_retirement_corpus<estimated_retirement_corpus: 
        print(f"""
             required retirement corpus: {required_retirement_corpus} \n estimated retirement corpus: {estimated_retirement_corpus} \n
             retirement_gap = 0, \n surplus_corpus: {estimated_retirement_corpus-required_retirement_corpus}
""")
        result={}
        result['goal_name']="retirement"
        result['corpus_needed']=0
        result['corpus_gap']=0
        result['funded_from']=[]
        result['target_year']= current_year + years_to_retire
        #result['sip_amount']=0
        #result['sip_years']=0
        result['surplus']=estimated_retirement_corpus-required_retirement_corpus

    education_planning=[]
    for goal in client_data3['education_planning_summary']:
        education_planning.append({'goal_name': goal['name'] + " " + goal['type'], 'target_year': goal['target_year'], 'corpus_needed': goal['final_gap'], 'corpus_gap': goal['final_gap'], 'funded_from':[] })
    education_goals=education_planning

    financial_goals=[]
    for goal in client_data3['financial_goals']:
        financial_goals.append({'goal_name': goal['goal_name'] , 'target_year': goal['target_year'], 'corpus_needed': goal['goal_gap'], 'corpus_gap': goal['goal_gap'], 'funded_from':[] })
    other_goals=financial_goals

    total_goals=[result]+education_goals+other_goals

    print(f"total_goals: {total_goals}")

    return {'goals': total_goals}


def asset_basket_classification(state: ClientState):

    client_data=state['client_data']
    asset_basket=client_data['investment_details']
    i=0
    for asset in asset_basket:
        
        for instrument in asset_basket[f"{asset}"]:
            if asset=='real_estate_investment':
                instrument['asset_tag']='fixed_asset'
                instrument['asset_id']=i
            if asset=='retirement_investments':
                for retirement_instrument in asset_basket[asset][f'{instrument}']:
                    retirement_instrument['asset_tag']='retirement_asset'
                    retirement_instrument['asset_id']=i
                    i+=1
            if asset=='bonds':
                instrument['asset_tag']='fixed_asset'
                instrument['asset_id']=i
            if asset=='mutual_funds':
                instrument['asset_tag']='liquid_asset'
                instrument['asset_id']=i
            if asset=='direct_equity':
                instrument['asset_tag']='liquid_asset'
                instrument['asset_id']=i
            if asset=='reits':
                instrument['asset_tag']='liquid_asset'
                instrument['asset_id']=i
            if asset=='pms_aif':
                instrument['asset_tag']='fixed_asset'
                instrument['asset_id']=i
            if asset=='esops':
                instrument['asset_tag']='fixed_asset'
                instrument['asset_id']=i
            if asset=='ncd_govt':
                instrument['asset_tag']='fixed_asset'
                instrument['asset_id']=i
            if asset=='fixed_deposits':
                instrument['asset_tag']='fixed_asset'
                instrument['asset_id']=i
            i+=1

    liquid_assets=[]
    fixed_assets=[]
    retirement_assets=[]
    asset_basket=client_data['investment_details']  
    for asset in asset_basket:
        
        if asset=='financial_summary':
            continue       
        for instrument in asset_basket[f"{asset}"]:
            print(f"instrument: {instrument}")
            if asset=='retirement_investments':
                for retirement_instrument in asset_basket[asset][f'{instrument}']:
                        #print(asset_basket[asset][f'{instrument}'])
                        print(f"retirement instrument: {retirement_instrument}")
                        if retirement_instrument['asset_tag']=='retirement_asset':
                            retirement_assets.append({f'{instrument}':retirement_instrument})
                        
            elif instrument['asset_tag']=='liquid_asset':
                #liquid_assets[f'{asset}']=instrument
                liquid_assets.append({f'{asset}':instrument})

            elif instrument['asset_tag']=='fixed_asset':
                #fixed_assets[f'{asset}']=instrument
                fixed_assets.append({f'{asset}':instrument})
            
    return {'retirement_assets': retirement_assets, 'liquid_assets': liquid_assets, 'fixed_assets': fixed_assets}


def calculate_total_asset_value(state: ClientState):
    """
    Calculates the total value of all assets from a given list of
    asset dictionaries. It assumes all provided assets are liquid.

    Args:
        assets_list (list): A list of dictionaries, where each dictionary
                            represents an asset.

    Returns:
        dict: A dictionary containing the 'total_asset_value'.
    """

    assets_list=state['liquid_assets']
    total_asset_value = 0.0

    for asset_item in assets_list:
        # Skip any empty or invalid entries in the list
        if not asset_item or not isinstance(asset_item, dict):
            continue

        # Extract the inner dictionary which holds the actual asset data
        asset_data = list(asset_item.values())[0]

        # Find the asset's value, checking different possible keys.
        # Defaults to 0 if the value is None or the key doesn't exist.
        current_value = asset_data.get('current_value') or asset_data.get('portfolio_value') or 0

        # Ensure the value is a number before adding, defaulting to 0 otherwise
        if not isinstance(current_value, (int, float)):
            current_value = 0

        # Add the value to the total
        total_asset_value += current_value

    return {
        "liquid_pool": total_asset_value
    }


def calculate_fixed_assets_value(state: ClientState):
    """
    Calculates the total current value of fixed assets from a list.

    The function approximates the current value based on the most relevant
    field available for each asset type (e.g., principal for FDs, vested
    value for ESOPs).

    Args:
        assets_list (list): A list of dictionaries, with each dictionary
                            representing a different fixed asset.

    Returns:
        dict: A dictionary containing the 'total_fixed_asset_value'.
    """

    assets_list=state['fixed_assets']
    total_fixed_asset_value = 0.0

    for asset_item in assets_list:
        # Skip any empty or malformed entries in the list
        if not asset_item or not isinstance(asset_item, dict):
            continue

        # Each item is a dict with one key (the asset type), so we get its value
        asset_data = list(asset_item.values())[0]
        current_value = 0

        # Determine the value based on the asset type and available fields
        if 'current_market_value' in asset_data:
            # For Real Estate
            current_value = asset_data.get('current_market_value')
        elif 'investment_amount' in asset_data:
            # For Bonds
            current_value = asset_data.get('investment_amount')
        elif 'vested_esops_value' in asset_data:
            # For ESOPs, only consider the vested portion
            current_value = asset_data.get('vested_esops_value')
        elif 'principal_amount' in asset_data:
            # For Fixed Deposits, use the principal as the current value
            current_value = asset_data.get('principal_amount')
        elif 'current_value' in asset_data:
            # For PMS/AIF and other similar assets
            current_value = asset_data.get('current_value')

        # Add to the total, ensuring the value is a number
        if isinstance(current_value, (int, float)):
            total_fixed_asset_value += current_value

    return {
        "fixed_asset_pool": total_fixed_asset_value
    }


def plan_goals(state: ClientState):
    
    """
    Greedy allocator per goal (in given order), allowing up to `max_postpone` years of postponement.
    Strategy:
      - If horizon <= near_term_years, try lumpsum-first (then freed SIP, then surplus SIP, then more lumpsum).
      - Else (longer horizon), try freed SIP -> surplus SIP -> lumpsum.
      - If still gap, postpone by +1 year (up to max_postpone).
    Outputs final states and per-goal funding trails with a correct freed_sip schedule.
    """
    updated_goals = state['goals']   #List[Dict[str, Any]],
    monthly_surplus_init =state['client_data']['investment_details']['financial_summary'][0]['monthly_surplus']
    liquid_pool_init = state['liquid_pool']
    freed_sip_init =  state['freed_funds']      #Dict[int, float],
    r_annual = 0.08
    near_term_years = 2
    max_postpone = 10

    monthly_surplus = float(monthly_surplus_init)
    liquid_pool = float(liquid_pool_init)
    freed_sip = dict(freed_sip_init)  # year -> monthly freed at that year

    current_date=datetime.today()
    current_year=current_date.year
    
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
            
            print(f'type of end year - current year: {type(near_term_years)}')
            short_horizon = end_year - current_year <= near_term_years

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

    fund_allocation={
        "goals": results,
        "ending_monthly_surplus": monthly_surplus,
        "ending_liquid_pool": liquid_pool,
        "ending_freed_sip_schedule": freed_sip
    }

    return {'goal_funding': fund_allocation}


# assets percentage calculation:
def calculate_asset_percentages(state: ClientState):
    """
    Calculate percentage allocation of assets across categories and within each category.
    
    Args:
        retirement_assets: List of retirement assets
        liquid_assets: List of liquid assets  
        fixed_assets: List of fixed assets
        
    Returns:
        Dictionary containing category percentages and individual asset percentages
    """
    
    retirement_assets = state['retirement_assets'] 
    liquid_assets = state['liquid_assets']
    fixed_assets = state['fixed_assets']
    
    def get_retirement_value(asset):
        key = list(asset.keys())[0]
        val = asset[key]
        if key == 'ulip':
            return val['maturity_value']
        elif key == 'epf':
            return val['current_value']
        elif key == 'ppf':
            return val['current_value']
        elif key == 'nps':
            return val['current_value']
        else:
            return 0

    def get_liquid_value(asset):
        key = list(asset.keys())[0]
        val = asset[key]
        if key == 'mutual_funds':
            if val['current_value'] is not None:
                return val['current_value']
            elif val['sip_amount'] is not None:
                return val['sip_amount'] * 12  # Estimate annual value
            else:
                return 0
        elif key == 'direct_equity':
            return val['portfolio_value']
        elif key == 'reits':
            return val['current_value']
        else:
            return 0

    def get_fixed_value(asset):
        key = list(asset.keys())[0]
        val = asset[key]
        if key == 'real_estate_investment':
            return val['current_market_value']
        elif key == 'bonds':
            return val['investment_amount']
        elif key == 'pms_aif':
            return val['current_value']
        elif key == 'esops':
            return val.get('vested_esops_value', 0) + val.get('unvested_esops_value', 0)
        elif key == 'fixed_deposits':
            return val['principal_amount']
        else:
            return 0

    # Calculate category totals
    retirement_total = sum(get_retirement_value(a) for a in retirement_assets)
    liquid_total = sum(get_liquid_value(a) for a in liquid_assets)
    fixed_total = sum(get_fixed_value(a) for a in fixed_assets)
    grand_total = retirement_total + liquid_total + fixed_total

    # Calculate category percentages
    retirement_percent = (retirement_total / grand_total) * 100 if grand_total != 0 else 0
    liquid_percent = (liquid_total / grand_total) * 100 if grand_total != 0 else 0
    fixed_percent = (fixed_total / grand_total) * 100 if grand_total != 0 else 0

    # Calculate individual asset percentages within each category
    retirement_assets_percent = {}
    for a in retirement_assets:
        key = list(a.keys())[0]
        val = get_retirement_value(a)
        retirement_assets_percent[key] = (val / retirement_total) * 100 if retirement_total != 0 else 0

    liquid_assets_percent = {}
    for a in liquid_assets:
        key = list(a.keys())[0]
        val = get_liquid_value(a)
        liquid_assets_percent[key] = (val / liquid_total) * 100 if liquid_total != 0 else 0

    fixed_assets_percent = {}
    for a in fixed_assets:
        key = list(a.keys())[0]
        val = get_fixed_value(a)
        fixed_assets_percent[key] = (val / fixed_total) * 100 if fixed_total != 0 else 0

    result={
        'category_percentage': {
            'retirement': retirement_percent,
            'liquid': liquid_percent,
            'fixed': fixed_percent
        },
        'retirement_assets_percent': retirement_assets_percent,
        'liquid_assets_percent': liquid_assets_percent,
        'fixed_assets_percent': fixed_assets_percent
    }

    return {'asset_percentages': result}

# risk assessment using client's assets
def risk_appetite_assessment(state: ClientState):

    client_assets=state['client_data']["investment_details"]

    prompt=f"""
           You will be provided with user's assets, you must analyse his assets and determine user's risk appetite based only on the given conditions:

           1. If current assets include Direct Equity, Equity mutual funds, or any other equity instruments 
               then mark EQUITY EXPOSURE as True
              Else mark EQUITY EXPOSURE as False

            2. if Years to retire < 5 and EQUITY EXPOSURE is True then mark RISK APPETITE as 'Medium'
               if Years to retire < 5 and EQUITY EXPOSURE is False then mark RISK APPETITE as 'Low'

            3. If Years to retire >= 5 and EQUITY EXPOSURE is True then mark RISK APPETITE as 'Medium to High'
               if Years to retire >= 5 and EQUITY EXPOSURE is False then mark RISK APPETITE as 'Medium to Low'
            
            User's assets: {client_assets}
            Years to retire: {state['client_data']['client_data']['retirement_age']-state['client_data']['client_data']['client_age']}
            
            Your output must align with the structured schema provided to as shown below:

            """
    
    response=risk_llm.invoke(prompt)

    risk_appetite_analysis={'risk_appetite': response.risk_assessment, 'reason': response.reason_of_risk_assessment}

    return {'risk_appetite': risk_appetite_analysis} 

# financial health checkup
# def financial_health(state: ClientState):

#     liquidity
#     flexibility
#     asset_allocation
#     saving_adequacy
#     spending_behaviour

##################################################################################### Nodes ##################################################################################################################################


graph=StateGraph(ClientState)

# add nodes
graph.add_node('calculate_age', calculate_age)
graph.add_node('goals_future_value', goals_future_value)
graph.add_node('calculate_education_funding', calculate_education_funding)
graph.add_node('calculate_retirement_corpus', calculate_retirement_corpus)
graph.add_node('calculate_all_retirement_investments', calculate_all_retirement_investments)
graph.add_node('retirement_goal', retirement_goal)
graph.add_node('asset_basket_classification', asset_basket_classification)
graph.add_node('calculate_total_asset_value', calculate_total_asset_value)
graph.add_node('calculate_fixed_assets_value', calculate_fixed_assets_value)
graph.add_node('plan_goals', plan_goals)
graph.add_node('calculate_asset_percentages', calculate_asset_percentages)
graph.add_node('risk_appetite_assessment', risk_appetite_assessment)


# add edges
graph.add_edge(START, 'calculate_age')
graph.add_edge('calculate_age', 'goals_future_value')
graph.add_edge('goals_future_value', 'calculate_education_funding')
graph.add_edge('calculate_education_funding', 'calculate_retirement_corpus')
graph.add_edge('calculate_retirement_corpus', 'calculate_all_retirement_investments')
graph.add_edge('calculate_all_retirement_investments', 'retirement_goal') 
graph.add_edge('retirement_goal', 'asset_basket_classification')
graph.add_edge('asset_basket_classification', 'risk_appetite_assessment')
graph.add_edge('risk_appetite_assessment', 'calculate_total_asset_value')
graph.add_edge('calculate_total_asset_value', 'calculate_fixed_assets_value')
graph.add_edge('calculate_total_asset_value', 'calculate_asset_percentages')
graph.add_edge('calculate_asset_percentages', 'plan_goals')
graph.add_edge('calculate_fixed_assets_value', END)

# define workflow
workflow=graph.compile()

# execute the workflow
initial_state={'client_data': client_data, 'freed_funds': {2025: 3000}}

final_state=workflow.invoke(initial_state)
# final_state['goal_funding']
# final_state['retirement_schemes_fv']
# final_state['required_retirement_corpus']
# final_state['freed_funds']
# final_state['retirement_assets']
# final_state['fixed_assets']
# final_state['liquid_assets']
# final_state['fixed_asset_pool']
# final_state['goals']
# final_state['retirement_schemes_fv']
# final_state['children_education_planning']
# final_state['calculate_asset_percentages']
# final_state['client_data']['client_data']
# final_state['risk_appetite']
# #years_to_retire=final_state['client_data']['client_data']['retirement_age']-final_state['client_data']['client_data']['client_age']
# final_state['client_data']['investment_details']['financial_summary']







