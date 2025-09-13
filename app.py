from langgraph.graph import StateGraph, START, END
from langchain_openai import AzureChatOpenAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import TypedDict, Literal, Annotated, List, Dict, Any
from langchain_core.messages import AnyMessage, SystemMessage, AIMessage, HumanMessage, ToolMessage
import os 
import operator
from langchain.tools import tool
from dotenv import load_dotenv
from datetime import datetime
from datetime import date
import json
from client_data_updated import (future_value, calculate_required_sip, ulip_future_value, epf_future_value, 
                                 ppf_future_value, nps_future_value) 

from Armstrong_goal_allocation import (lumpsum_required, sip_required, future_value_lumpsum, fv_sip,
                                       add_to_schedule, reduce_from_schedule, move_freed_monthly, 
                                       apply_freed_sip_to_goal, apply_surplus_sip_to_goal, apply_lumpsum_to_goal)

from datetime import datetime
from collections import defaultdict
from math import ceil
#from __future__ import annotations
from math import log, ceil, isfinite
from datetime import date
from langchain_community.tools.tavily_search import TavilySearchResults
from math import isclose
#print('hello')

load_dotenv()

AZURE_API_KEY=os.getenv('AZURE_API_KEY')
AZURE_API_BASE=os.getenv('AZURE_API_BASE')
AZURE_API_VERSION=os.getenv('AZURE_API_VERSION')
AZURE_DEPLOYMENT_NAME=os.getenv('AZURE_DEPLOYMENT_NAME')

GROQ_API_KEY=os.getenv('GROQ_API_KEY')
# model=ChatOpenAI(model='openai-4o')
TAVILY_API_KEY=os.getenv('TAVILY_API_KEY')


#llm_groq=ChatGroq(model='qwen/qwen3-32b', api_key=GROQ_API_KEY, temperature=0.5)

llm_azure = AzureChatOpenAI(
    api_key=AZURE_API_KEY,  # AZURE_API_KEY
    azure_endpoint=AZURE_API_BASE,  # AZURE_API_BASE
    api_version=AZURE_API_VERSION,  # AZURE_API_VERSION
    deployment_name=AZURE_DEPLOYMENT_NAME,  # AZURE_DEPLOYMENT_NAME
    temperature=0  # Optional
)

#response=llm_azure.invoke("Explain why is being a good communicator is importan, explain this like Richard Feynman.")
#response.content

tavily_tool = [TavilySearchResults(max_results=1, api_key=TAVILY_API_KEY)] # tavily search tool from langchain community
edu_tavily_llm=llm_azure.bind_tools(tavily_tool)
tavily_tool_dict={t.name: t for t in tavily_tool}

def college_web_scrapper(user_query: str):
    tool_call=True
    results=[HumanMessage(content=user_query)]
    while tool_call==True: 
        result=edu_tavily_llm.invoke(results)
        
        results.append(result)
        if len(results[-1].tool_calls)>0:
            tool_calls=[result][-1].tool_calls
            for t in tool_calls:
                print(f"calling: {t}")
                if not t['name'] in tavily_tool_dict:
                    print("\n ...bad tool call...")
                    output=f'bad tool call retry, tool name read is {t['name']}'
                else: 
                    output=tavily_tool_dict[t['name']].invoke(t['args']) 
                results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(output))) 
            print('back to model') 
        else:
            tool_call=False 
    
    return result

# result=college_web_scrapper("List top 10 colleges in India with their fees for undergraduate course in Electrical Engineering")
# result.content  
# OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
# llm_azure=ChatOpenAI(api_key=OPENAI_API_KEY, model='gpt-4o')
#print(OPENAI_API_KEY)

# response=llm_azure.invoke("hello")
llm_fees_scrapper_=ChatGroq(model='compound-beta', api_key=GROQ_API_KEY, temperature=0.5 , disable_streaming=False)
# response=llm_fees_scrapper_.invoke('hi')

####################################################################################### freed timelines utility functions ################################################################################################

def _months_to_close(principal: float, annual_rate: float, emi: float) -> int | None:
    """
    Solve for remaining months 'n' in the annuity formula:
      EMI = P * r * (1+r)^n / ((1+r)^n - 1)
    given P (principal), r (monthly rate), EMI.
    Returns ceil(n). If emi <= P*r (i.e., non-amortizing), returns None.
    """
    r = annual_rate / 12.0
    if emi <= principal * r:  # EMI too low to amortize the loan
        return None
    n = log(emi / (emi - principal * r)) / log(1 + r)
    return ceil(n)

def _add_months(start: date, months: int) -> date:
    """Add a positive number of months to a date (month-end alignment not required)."""
    y, m = start.year, start.month
    m_total = m + months
    y += (m_total - 1) // 12
    m = (m_total - 1) % 12 + 1
    # keep the same day; if day overflow matters for you, clamp to month-end
    return date(y, m, min(start.day, 28))  # safe clamp to 28

####################################################################################### freed timelines utility functions #########################################################################################################
######################################################################### liability analysis utility functions ########################################################################################

def _months_to_close(principal: float, r_month: float, payment: float) -> int | None:
    """Closed-form remaining months for an amortizing loan at constant payment."""
    if payment <= principal * r_month:
        return None  # non-amortizing at this payment level
    n = log(payment / (payment - principal * r_month)) / log(1 + r_month)
    return ceil(n)

def _balance_after_k(principal: float, r_month: float, payment: float, k: int) -> float:
    """Remaining balance after k payments of `payment`."""
    if r_month == 0:
        return max(0.0, principal - payment * k)
    return principal * (1 + r_month) ** k - payment * ((1 + r_month) ** k - 1) / r_month

def _add_months(d: date, months: int) -> date:
    """Add months to a date (clamped to day 28 for safety)."""
    y, m = d.year, d.month
    m_total = m + months
    y += (m_total - 1) // 12
    m = (m_total - 1) % 12 + 1
    return date(y, m, min(d.day, 28))

def _freed_by_year(close_dt: date, monthly_freed: float) -> dict[int, float]:
    """Record freed monthly amount in the year when the loan closes."""
    return {close_dt.year: float(monthly_freed)}

def _safe(n):
    return None if (n is None or not isfinite(n)) else int(n)

######################################################################### liability analysis tool and components ########################################################################################

############################################################ education web scrapper components and tools ##########################################################################################

@tool
def clarify_with_user(question: str)->str:
    "This function is used to interact with the user and ask any question "
    answer=input(question)
    return str(answer)

calrify_toools=[clarify_with_user]
clarify_tools={t.name : t for t in calrify_toools}
education_plan_llm=llm_azure.bind_tools([clarify_with_user])

class EducationSchema(BaseModel): 

    education_preference: Literal['Domestic', 'International']
    country_preference: str
    college_preferences: List[str]
    #course_preference: str
    stream_preference: str

education_plan_structured_llm=llm_azure.with_structured_output(EducationSchema)

class Education(TypedDict): 

    education_preference: Literal['Domestic', 'International']   
    country_preference: str
    college_preferences: List[str]
    stream_preference: str
    #course_preference: str 
    avg_college_fees: float
    list_of_colleges: List[str] 
    messages: Annotated[AnyMessage, operator.add]

class FeesFC(BaseModel):
    top_8_colleges: List[str] = Field(description="college name -> 4-year UG fee in INR")
    average_fees: float = Field(ge=0, description="Average fee across the listed colleges")

llm_avg_fees=llm_azure.with_structured_output(FeesFC)

edu_prompt="""You are an {edu_type} education profiling expert based in India. Your role is to collect details from the client about their child's ** {edu_type} education plan**. 

                You will be provided with:
                - The client’s name 
                - The child’s name 
                - The client’s stated expectation for the child’s education: either **Domestic** or **International**
                - Stream which the client's child wants to pursue. 

                Your tasks are strictly limited to the following step:

                1. **Confirm the country of {edu_type} education:**
                - If the client has mentioned **International**, you must ask in which country they expect their child to pursue {edu_type} education, you must provide only two options to client 1. UK 2. US(America)
                    → Use the tool `clarify_with_user` for this question.
                - If the client has mentioned **Domestic**, assume the {edu_type} education is planned in **India**. No need to confirm further.

                **Tool usage:**
                - The tool `clarify_with_user` must be used for both types of questions.
                - Tool input format:    
                   ```{
                    "question": "<your question text here>"
                   }```

                **Final Output:**
                At the end, provide a structured summary that includes:
                - Education type: Domestic or International
                - Child’s education country (resolved from the conversation)
            """

def _edu_plan_llm(state: Education):
        messages=state['messages']
        messages=[SystemMessage(edu_prompt)]+messages
        message=education_plan_llm.invoke(messages)
        return {'messages': [message]}

def action(state: Education):
        tool_calls=state['messages'][-1].tool_calls
        results=[]
        for t in tool_calls:
            print(f"calling: {t}")
            result=clarify_tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("back to model")
        return {'messages': results}

def check_condition(state: Education):
    result=state['messages'][-1].tool_calls
    return len(result)>0

def structured_edu(state: Education):

    last_text=state['messages'][-1]
    final_result=education_plan_structured_llm.invoke(f"""You are provided with client's kid's education plan, 
                                                            you must present it according to the expected 
                                                            schema {last_text}""") 
    
    return {'education_preference': final_result.education_preference, 'country_preference': final_result.country_preference, 'stream_preference': final_result.stream_preference }

class CalculateOverallFeesInput(BaseModel):
    annual_fees: float = Field(..., description="Annual fees in the college's native currency")
    duration: int = Field(..., ge=1, description="Duration of the program in years")

class CurrencyConversionInput(BaseModel):
    amount_to_convert: float = Field(..., description="Amount to convert to Indian Rupees (INR)")
    conversion_ratio: float = Field(..., gt=0, description="FX rate: target_INR / source_currency")

class AvgFeesInput(BaseModel):
    colleges: List[float] = Field(description="List of overall fees (one per college) in a common currency")

    def _non_empty(cls, v):
        if not v or len(v) == 0:
            raise ValueError("colleges must contain at least one number")
        return v
    
# ---- Tools ----
@tool
def calculate_overall_fees(annual_fees: float, duration: int) -> float:
    """
    Calculate total program fees as annual_fees * duration.
    Returns the total in the same currency as the annual_fees.
    """
    return float(annual_fees) * int(duration)

@tool
def currency_conversion(amount_to_convert: float, conversion_ratio: float) -> float:
    """
    Convert a given amount to INR using the provided conversion ratio.
    Example: if 1 USD = 83.25 INR, pass conversion_ratio=83.25.
    """
    return float(amount_to_convert) * float(conversion_ratio)
    
@tool
def avg_fees(colleges: List[float]) -> float:
    """
    Compute the arithmetic mean of overall fees for the given colleges.
    Expects all values in the SAME currency (e.g., INR).
    """
    # defensive numeric cast + basic validation
    vals = [float(x) for x in colleges]
    return sum(vals) / len(vals)
    
toolset = [calculate_overall_fees, currency_conversion, avg_fees]
llm_scrapper_tools = llm_azure.bind_tools(toolset)

tool_index_tool = {t.name: t for t in toolset}

def run_with_tools(user_prompt: str, max_iters: int = 5) -> AIMessage:
        messages: List = [HumanMessage(content=user_prompt)]
        for _ in range(max_iters):
            ai: AIMessage = llm_scrapper_tools.invoke(messages)
            messages.append(ai)

            # If the model didn't request any tools, we're done.
            if not ai.tool_calls:
                return ai.content

            # Execute each requested tool and append ToolMessage responses
            for call in ai.tool_calls:
                tool_name = call["name"]
                tool_args = call["args"]
                tool = tool_index_tool.get(tool_name)
                if tool is None:
                    # Return an error ToolMessage if unknown tool
                    messages.append(
                        ToolMessage(
                            name=tool_name,
                            tool_call_id=call["id"],
                            content=f"Tool '{tool_name}' not found."
                        )
                    )
                    continue

                try:
                    result = tool.invoke(tool_args)  # safely handles schema
                except Exception as e:
                    result = f"Error running {tool_name}: {e}"

                messages.append(
                    ToolMessage(
                        name=tool_name,
                        tool_call_id=call["id"],
                        content=str(result)
                    )
                )
        # If we exit the loop, return the last AI message
        return ai.content
    
def college_info_scrapper(state: Education):
    
    education_preference=state['education_preference']
    country_preference=state['country_preference']
    #college_preferences=state['college_preferences']
    #course_preference=state['course_preference']
    stream_preference=state['stream_preference']

    #response=llm_fees_scrapper_.invoke(f"Gather annual fees of the stream: {stream_preference} of top 5 colleges in country {country_preference} along with the duration of each stream and currency.")   

    response=college_web_scrapper(f"Gather annual fees of the stream: {stream_preference} of top 5 colleges in country {country_preference} along with the duration of each stream and currency. DO ENSURE THAT THE AMOUNT IS IN FULL FIGURES, SO NOT EXPRESS THEM IN Lakh, thousands or any othe unit just full figure and it's currency.")

    user_prompt=str("""You are an expert financial assistant with access to three tools:
                        1) `calculate_overall_fees`(annual_fees: float, duration: int) -> float
                        2) `currency_conversion`(amount_to_convert: float, conversion_ratio: float) -> float
                        3) `avg_fees`(colleges: List[float]) -> float

                        Your task is to read the COLLEGE INPUT below and produce, for each college, the total program cost and a final overall average (in INR). STRICTLY follow the steps and rules:

                        — STEPS (use tools exactly as specified) —
                        A. For each college:
                        A1. Determine the program duration (years).
                            • If duration is present, use it.
                            • If missing, infer from the stream/degree using these defaults:
                                - BTech/BE/Engineering UG: 4
                                - BA/BSc/BCom/General UG: 3
                                - MBBS/Medicine UG: 5
                                - MBA/PGDM: 2
                                - MS/MSc: 2
                                - PhD: 4
                                If none match, assume 3 and note "assumed".
                        A2. Call `calculate_overall_fees` tool with:
                            {{
                                "annual_fees": <annual fee in native currency>,
                                "duration": <duration years>
                            }}
                            The result is total_native.
                        A3. Convert to INR (total_in_inr):
                            • If college country is India or currency already INR, set total_in_inr = total_native and conversion_ratio_used = 1.0 (no tool call).
                            • Otherwise, choose a sensible conversion ratio (float, > 0), then call currency_conversion with:
                                {{
                                "amount_to_convert": total_native,
                                "conversion_ratio": <FX to INR, e.g., if 1 USD = 83.25 INR then 83.25>
                                }}
                            Record conversion_ratio_used.

                        B. After all colleges are processed:
                        B1. Call `avg_fees` with the list of all total_in_inr values in the SAME order as processed:
                            {{
                                "colleges": [<total_in_inr_1>, ..., <total_in_inr_n>]
                            }}
                        B2. If exactly 8 colleges are present, this is the “overall_average_fees_in_inr”.
                            If not 8, still compute the average of all provided and set "note" to indicate the count.
                                                
                        — OUTPUT FORMAT (return ONLY one JSON object) —
                        {
                        "colleges": [
                            "name": "<college name>"
                            // ... one entry per college
                        ],
                        "overall_average_fees_in_inr": <float>,
                        "note": "<Only include if number of colleges != 8>"
                        }

                        — RULES & GUARDRAILS —
                        • ALWAYS use calculate_overall_fees for each college to get total_native.
                        • ONLY use currency_conversion for non-INR totals. Never reconvert INR.
                        • ALWAYS use avg_fees ONCE at the end with the INR totals list.
                        • Keep numbers as floats; round to 2 decimals in the final JSON.
                        • If any critical field is missing (e.g., annual fees), skip that college and add a short assumption in “assumptions”.

                            — COLLEGE INPUT — \n
                                    """ + response.content)
    
    response2=run_with_tools(user_prompt+response.content)

    avg_fees=llm_avg_fees.invoke(f"You are provided with content of college infos, your task is to represent them based on the schema provided, content: {response2}")
    print(f'List of colleges along with average fees \n : {avg_fees}') 
    top_8_colleges=avg_fees.top_8_colleges
    avg_cllg_fees=avg_fees.average_fees

    return {'avg_college_fees': avg_cllg_fees, 'list_of_colleges': top_8_colleges}

############################################################### education web scrapper components #################################################################################################
####################### tools, llms and classes for goal prioritization ##################################################################################

class FundingSource(BaseModel):
    name: str
    amount: float

from typing import List
from pydantic import BaseModel, Field

class Goal(BaseModel):
    goal_name: str = Field(description="Name of the financial goal")
    target_year: int = Field(description="The year the goal should be achieved")
    corpus_needed: float = Field(description="Total corpus required for the goal")
    corpus_gap: float = Field(description="Gap between required and available corpus")
    funded_from: List[str] = Field(default_factory=list, description="Funding sources for this goal")
    surplus: int = Field(default=0, description="Any surplus funds allocated to the goal")
    priority_score: float = Field(description="The calculated priority score for the goal")

class PrioritizedGoals(BaseModel):
    """A list of financial goals sorted by priority score."""
    goals: List[Goal]

@tool
def calculate_priority_score(weight: float, target_year: int) -> float:
    """Calculates the priority score for a financial goal."""
    time_left = target_year - datetime.now().year
    if time_left <= 0: time_left = 0.1
    return (weight * 0.6) + ((1 / time_left) * 0.4)

@tool
def sort_goals_by_priority(goals: List[Dict]) -> List[Dict]:
    """Sorts a list of goals based on their 'priority_score'."""
    #goals=goals['goals']
    if not all("priority_score" in g for g in goals):
        raise ValueError("Each goal must include a 'priority_score' before sorting.")
    return sorted(goals, key=lambda x: x["priority_score"], reverse=True)

toools=[calculate_priority_score, sort_goals_by_priority]
tools={t.name : t for t in toools}

goal_llm=llm_azure.bind_tools(toools)  

structured_goal_llm=llm_azure.with_structured_output(PrioritizedGoals) 
######################## tools for goal prioritization #################################

class RiskSchema(BaseModel):

    risk_assessment: Literal['Low', 'Medium', 'Medium to High', 'Medium to Low'] = Field(
        description='Marks risk appetite of the user')
    reason_of_risk_assessment: str = Field(
        description='Reason for the risk assessment chosen based on the conditions provided')
    
risk_llm=llm_azure.with_structured_output(RiskSchema) 

class ClientState(TypedDict):

    client_data: json                         # input
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
    goal_funding : Annotated[list[dict], operator.add]
    asset_percentages : dict
    financial_overview : dict
    sorted_goals: dict
    surplus_from_goals : float
    financial_goals : list 
    liability_allocation: Annotated[list[dict], operator.add]
    used_monthly_surplus : Annotated[list[float], operator.add]            #float
    used_liquid_surplus : Annotated[list[float], operator.add]  #float 
    EMI_allocated : bool 
    freed_timeline : Annotated[list[dict], operator.add]   #dict 
    EMI_allocation: bool 
    loan_prepayed_times : int
    liabilities : list
    unused_monthly_surplus : Annotated[List[float], operator.add] 
    extra_monthly_surplus_investment_plan : dict
    optimal_goal_allocation : dict
    optimal_loan_allocation : dict
    final_unused_monthly_surplus : float

##################################################################################### Nodes ##################################################################################################################################

def calculate_age(state: ClientState): # calculates ages of all the individual mentioned in the client data
    """
    Computes the current ages of all individuals (client, spouse, and children) 
    based on their dates of birth in the provided client data.

    Args:
        state (ClientState): A state object containing a "client_data" dictionary 
            with the following structure:
                - client_data.date_of_birth (str, YYYY-MM-DD): Client's DOB
                - client_data.spouse_dob (str, YYYY-MM-DD): Spouse's DOB
                - client_data.if_any_kids (bool): Whether children are present
                - client_data.children (list[dict]): Each child includes:
                    - child_dob (str, YYYY-MM-DD): Child's DOB

    Returns:
        dict: The updated "client_data" dictionary with added fields:
                - client_age (int): Client's current age
                - spouse_age (int): Spouse's current age
                - children[].child_age (int): Each child's current age

    Purpose:
        Enhances client_data by populating the ages of all mentioned individuals 
        using the current system date.
    """
    client_data=state['client_data']
    current_date = date.today()  
    if client_data['client_data']['date_of_birth']: 
        client_age=current_date.year-datetime.strptime(client_data['client_data']['date_of_birth'], '%Y-%m-%d').year
        client_data['client_data']['client_age']=client_age 
     
    if client_data['client_data']['spouse_dob']:
        spouse_age=current_date.year-datetime.strptime(client_data['client_data']['spouse_dob'], '%Y-%m-%d').year
        client_data['client_data']['spouse_age']=spouse_age 

    if client_data['client_data']['if_any_kids']==True:
        for index, child_info in enumerate(client_data['client_data']['children']):
            child_age=current_date.year-datetime.strptime(child_info['child_dob'], '%Y-%m-%d').year
            client_data['client_data']['children'][index]['child_age']=child_age
    
    monthly_surplus=(state['client_data']['investment_details']['financial_summary'][0]['monthly_salary'] + state['client_data']['investment_details']['financial_summary'][0]['other_income(rental/interest/other)']) - ( state['client_data']['investment_details']['financial_summary'][0]['monthly_expenses_excl_emis'] + 
                                                                                                                                                                                                    state['client_data']['investment_details']['financial_summary'][0]['miscellaneous_kids_education_expenses_monthly'] + 
                                                                                                                                                                                                     (state['client_data']['investment_details']['financial_summary'][0]['annual_vacation_expenses'])/12 )  
    client_data['investment_details']['financial_summary'][0]['monthly_surplus']=monthly_surplus     
    return {'client_data': client_data, 'children_education_planning': [], 'financial_goals': [], 'liabilities': state['client_data']['liabilities']}

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
            if c['child_name'] == child['name_of_kid']:              # maybe this will have to be converted into similarity check.
                child_dob = datetime.strptime(c['child_dob'], '%Y-%m-%d').date()
                break
        
        if not child_dob:
            continue

        # Undergraduate (UG) Goal
        ug_target_year = child_dob.year + 18                            # ug target year is not provided as input in the current form, so default year is choosen is considered when the child reaches at the age of 18.
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
        pg_target_year = child_dob.year + 22                         # similarly target year of post graduation is not given, so default choosen when the child reaches the age 22.
        pg_years_to_goal = pg_target_year - today.year       
        education_goals.append({
            "name": child['name_of_kid'],
            "type": "post_graduation",
            "target_year": pg_target_year,
            "years_to_goal": pg_years_to_goal,
            "current_cost": child['current_fees_of_post_graduation'],
            "allocated_funds": child.get('fund_allocated_for_post_graduation'),
            "schemes": child.get('scheme_for_education', []),
            'funded_from': []
        })
        
    # 2. Sort goals chronologically by target year
    education_goals.sort(key=lambda x: x['target_year'])
    
    surplus_pool = 0.0
    # NEW: Initialize a set to track schemes that have been allocated to a goal.
    used_schemes = set()
    
    # 3. Process each goal in chronological order
    for goal in education_goals:
        # Calculate future cost of education (assuming 6% inflation)
        goal['current_cost']
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
            goal['funded_from'].append({'surplus_from_previous_education': used_surplus})  
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

    education_planning=[]
    for goal in education_goals:
        education_planning.append({'goal_name': goal['name'] + " " + goal['type'], 'target_year': goal['target_year'], 'corpus_needed': goal['final_gap'], 'corpus_gap': goal['final_gap'], 'funded_from':[] })
    
    return {'client_data': client_data, 'children_education_planning': education_planning}

def goals_future_value(state: ClientState):  # calculate the actual goal gap and allocating surplus for all the financial goals mentioned 
    """
    Calculates the future value of savings and required capital for each financial goal, 
    determines the funding gap, and reallocates any surplus from earlier goals 
    to later ones.

    Args:
        state (ClientState): A state object containing a "client_data" dictionary 
            with the following structure:
                - financial_goals (list[dict]): Each goal dictionary includes:
                    - target_year (int): Year when the goal must be achieved
                    - amount_saved_for_goal (float): Amount already saved toward the goal
                    - capital_required_today (float): Present-day value of capital needed 
                      to achieve the goal
                    - funded_from (list, optional): Records sources used to fund gaps 
                      (e.g., surplus from previous goals)

    Returns:
        dict: Updated "client_data" with enriched "financial_goals", where each goal includes:
                - future_value_of_saved_amount (float): Projected value of current savings 
                  at the target year (assumed 10% annual growth)
                - capital_required_at_target_year (float): Future value of required capital 
                  at the target year (assumed 6% annual inflation)
                - goal_gap (float): Remaining shortfall to fund the goal after considering 
                  savings and surplus reallocation
                - funded_from (list): Updated with surplus allocations from previous goals, 
                  if applicable

    Purpose:
        This function performs financial goal gap analysis by:
        1. Projecting savings and capital requirements into the future.
        2. Identifying shortfalls (goal gaps).
        3. Reallocating surplus from earlier goals to reduce gaps in later goals.
    """
    client_data=state['client_data']
    goals=client_data['financial_goals']              
    current_date=date.today()
    for goal in goals:
        if goal['target_year']-current_date.year >= 0:            # target year should not be less than current year
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

    financial_goals=[] 
    for goal in client_data['financial_goals']: 
        financial_goals.append({'goal_name': goal['goal_name'] , 'target_year': goal['target_year'], 'corpus_needed': goal['goal_gap'], 'corpus_gap': goal['goal_gap'], 'funded_from':[] })

    return {'client_data': client_data, 'surplus_from_goals': surplus, 'financial_goals': financial_goals} 

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
    # print("=" * 60)
    # print("METHOD A: STANDARD METHOD (Flat Expense Method)")
    # print("=" * 60)
    
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
    
    # print(f"Annual Expenses at Retirement: ₹{future_annual_expenses:,.2f}")
    # print(f"Retirement Duration: {retirement_duration} years")
    # print(f"Real Return Rate (Post-Inflation): {real_return_rate*100}%")
    # print(f"Required Corpus (Standard Method): ₹{standard_corpus:,.2f}")
    
    # METHOD B: SEGMENTED CASH FLOW METHOD (Lifestyle-Based Phases)
    # print("\n" + "=" * 60)
    # print("METHOD B: SEGMENTED CASH FLOW METHOD (Lifestyle-Based)")
    # print("=" * 60)
    
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
        
        # print(f"\n{phase['name']} ({phase_start}-{phase_end}):")
        # print(f"  Duration: {phase_duration} years")
        # print(f"  Expense Level: {phase['expense_multiplier']*100:.0f}% of baseline")
        # print(f"  Annual Expenses: ₹{phase_annual_expenses:,.2f}")
        # print(f"  Corpus Required: ₹{phase_corpus_at_retirement:,.2f}")
    
    retirement_plan["segmented_method"] = {
        "phases": segmented_phases,
        "total_required_corpus": round(total_segmented_corpus, 2)
    }
    
    # Summary and Comparison
    # print("\n" + "=" * 60)
    # print("RETIREMENT CORPUS SUMMARY")
    # print("=" * 60)
    # print(f"Standard Method Corpus:           ₹{standard_corpus:,.2f}")
    # print(f"Segmented Method Corpus:          ₹{total_segmented_corpus:,.2f}")
    difference = total_segmented_corpus - standard_corpus
    percentage_diff = (difference / standard_corpus) * 100
    # print(f"Difference:                       ₹{difference:,.2f} ({percentage_diff:+.1f}%)")
    
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
    
    # print(f"\nRECOMMENDED RETIREMENT CORPUS: ₹{recommended_corpus:,.2f}")
    # print(f"Method: {retirement_plan['recommendation']['method_used']}")
    
    return {'required_retirement_corpus': retirement_plan}


def calculate_all_retirement_investments(state: ClientState):
    """
    Computes future values of all retirement investment schemes, returning 
    per-scheme details, per-category totals, and an overall grand total.

    Args:
        state (ClientState): Contains client data with:
            - client_data.client_age (int)
            - investment_details.retirement_investments (dict) where keys are 
              categories ("ulip", "epf", "ppf", "nps") and values are lists of schemes.
              Each scheme includes fields like start date, contribution amounts, 
              rates, term/maturity, etc. (varies by category).

    Returns:
        dict: {
            "retirement_schemes_fv": {
                "schemes": {category: [ {scheme_no, future_value, total_invested}, ... ]},
                "category_totals": {category: float},
                "grand_total": float
            }
        }

    Purpose:
        Projects the corpus for each scheme until maturity/retirement, aggregates 
        by category, and produces a consolidated total.
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
    """
    Evaluates the retirement goal against the estimated corpus and consolidates it 
    with other education and financial goals.

    Args:
        state (ClientState): Contains client data and prior calculations with:
            - client_data.client_data.client_age (int)
            - client_data.client_data.retirement_age (int)
            - required_retirement_corpus.recommendation.recommended_corpus (float)
            - retirement_schemes_fv.grand_total (float)
            - education_planning_summary (list[dict]): Goals with name, type, target_year, final_gap
            - financial_goals (list[dict]): Goals with goal_name, target_year, goal_gap

    Returns:
        dict: {
            "goals": [
                {
                    "goal_name": str,
                    "corpus_needed": float,
                    "corpus_gap": float,
                    "target_year": int,
                    "funded_from": list,
                    "surplus": float (only for retirement goal)
                },
                ...
            ]
        }

    Purpose:
        - Compares required vs. estimated retirement corpus to compute gaps or surplus.
        - Builds a unified list of all goals (retirement, education, financial).
        - Standardizes each goal with common fields for downstream planning.
    """
    client_data3=state['client_data']
    required_retirement_corpus= state['required_retirement_corpus']['recommendation']['recommended_corpus']
    estimated_retirement_corpus = state['retirement_schemes_fv']['grand_total']
    sip_annual_rate = 0.10
    retirement_age = state['client_data']['client_data']['retirement_age']
    current_date=date.today()
    current_year=current_date.year

    if required_retirement_corpus>estimated_retirement_corpus:
#         print(f"""required retirement corpus: {required_retirement_corpus} \n estimated retirement corpus: {estimated_retirement_corpus} \n
#                 retirement_gap: {required_retirement_corpus-estimated_retirement_corpus}
#  """)
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
#         print(f"""
#              required retirement corpus: {required_retirement_corpus} \n estimated retirement corpus: {estimated_retirement_corpus} \n
#              retirement_gap = 0
# """)
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
#         print(f"""
#              required retirement corpus: {required_retirement_corpus} \n estimated retirement corpus: {estimated_retirement_corpus} \n
#              retirement_gap = 0, \n surplus_corpus: {estimated_retirement_corpus-required_retirement_corpus}
# """)
        result={}
        result['goal_name']="retirement"
        result['corpus_needed']=0
        result['corpus_gap']=0
        result['funded_from']=[]
        result['target_year']= current_year + years_to_retire
        #result['sip_amount']=0
        #result['sip_years']=0
        result['surplus']=estimated_retirement_corpus-required_retirement_corpus

    retirement_goal=[result]
    
    total_goals=retirement_goal
    if state['children_education_planning'] != []:
        total_goals+=state['children_education_planning']

    if state['financial_goals'] != []:
            total_goals+=state['financial_goals']

    return {'retirement_goal': retirement_goal, 'goals': total_goals}

def asset_basket_classification(state: ClientState):
    """
    Classifies all client investment instruments into liquid, fixed, or retirement assets, 
    and assigns each instrument a unique asset ID and tag.

    Args:
        state (ClientState): Contains client data with an "investment_details" dictionary, 
            which may include categories like:
                - real_estate_investment
                - retirement_investments (nested schemes)
                - bonds, mutual_funds, direct_equity, reits
                - pms_aif, esops, ncd_govt, fixed_deposits
                - financial_summary (ignored)

    Returns:
        dict: {
            "retirement_assets": [ {scheme_name: instrument_data_with_tags}, ... ],
            "liquid_assets":     [ {category: instrument_data_with_tags}, ... ],
            "fixed_assets":      [ {category: instrument_data_with_tags}, ... ]
        }

    Purpose:
        - Tags each investment with "asset_tag" (liquid, fixed, or retirement).
        - Assigns a unique "asset_id" to every instrument.
        - Produces categorized lists of instruments for further analysis.
    """
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
            # print(f"instrument: {instrument}")
            if asset=='retirement_investments':
                for retirement_instrument in asset_basket[asset][f'{instrument}']:
                        #print(asset_basket[asset][f'{instrument}'])
                        # print(f"retirement instrument: {retirement_instrument}")
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
    Computes the total value of all liquid assets in the client's portfolio.

    Args:
        state (ClientState): Contains a "liquid_assets" list, where each element 
            is a dictionary of the form {category_name: asset_data}. 
            Each asset_data may include keys like:
                - "current_value" (float)
                - "portfolio_value" (float, fallback)

    Returns:
        dict: {
            "liquid_pool": float   # Sum of all liquid asset values
        }

    Purpose:
        Iterates through all liquid assets, extracts their current/portfolio values, 
        and produces a consolidated liquid pool value.
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
    Allocates available resources (liquid pool, monthly surplus, freed SIPs) 
    to fund client goals using a greedy strategy with optional postponement.

    Args:
        state (ClientState): Contains:
            - sorted_goals (list[dict]): Goals with name, target_year, and corpus_needed
            - client_data.investment_details.financial_summary[0].monthly_surplus (float)
            - liquid_pool (float): Initial lump-sum funds available
            - freed_funds (dict[int, float]): Year → freed SIP amounts

    Returns:
        dict: {
            "goal_funding": {
                "goals": [ 
                    {
                        "goal_name": str,
                        "target_year": int,
                        "corpus_needed": float,
                        "corpus_gap": float,
                        "funded_from": list[dict]  # sources: lumpsum, freed_sip, surplus_sip, postponed, unfunded
                    },
                    ...
                ],
                "ending_monthly_surplus": float,
                "ending_liquid_pool": float,
                "ending_freed_sip_schedule": dict[int, float]
            }
        }

    Purpose:
        - Funds goals in order using a greedy allocation strategy:
            * Short horizon (≤ near_term_years): lumpsum → freed SIP → surplus SIP → lumpsum.
            * Long horizon: freed SIP → surplus SIP → lumpsum.
        - Allows postponement up to `max_postpone` years if gaps remain.
        - Tracks funding sources and updates global resource pools accordingly.
    """ 
    updated_goals = state['sorted_goals']   #List[Dict[str, Any]],
    if state['EMI_allocated']==True: 
        monthly_surplus_init = state['client_data']['investment_details']['financial_summary'][0]['monthly_surplus'] - state['used_monthly_surplus'][-1]
        liquid_pool_init = state['liquid_pool']-state['used_liquid_surplus'][-1]
        freed_sip_init = state['freed_timeline'][-1] 
    else: 
        monthly_surplus_init = state['client_data']['investment_details']['financial_summary'][0]['monthly_surplus']
        liquid_pool_init = state['liquid_pool']
        freed_sip_init =  state['freed_timeline'][-1]      #Dict[int, float],
    r_annual = 0.08 
    near_term_years = 2
    max_postpone = 10

    monthly_surplus = float(monthly_surplus_init)
    liquid_pool = float(liquid_pool_init)
    freed_sip = dict(freed_sip_init)  # year -> monthly freed at that year

    current_date=datetime.today()
    current_year=current_date.year
    postponed=False 
    EMI_allocation=False
    results = []
    for g in updated_goals:
        goal = {
            "goal_name": g["goal_name"],
            "target_year": int(g["target_year"]),
            "corpus_needed": float(g["corpus_needed"]),
            "corpus_gap": float(g["corpus_needed"]),
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
            
            # print(f'type of end year - current year: {type(near_term_years)}')
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
            postponed=True

        results.append(goal)

    fund_allocation={
        "goals": results,
        "ending_monthly_surplus": monthly_surplus,
        "ending_liquid_pool": liquid_pool,
        "ending_freed_sip_schedule": freed_sip
    }

    if fund_allocation['ending_monthly_surplus'] > 0.1*state['client_data']['investment_details']['financial_summary'][0]['monthly_surplus'] and postponed==False :
        EMI_allocation=True                    # EMI allocation will happen only if the ending_monthly_surplus is more than 10% of the monthly surplus.

    return {'goal_funding': [fund_allocation], 'EMI_allocation': EMI_allocation}


# assets percentage calculation:
def calculate_asset_percentages_and_ratios(state: ClientState): 
    """
    Analyze client’s assets and financial habits to compute allocation, ratios, and behavior flags.

    Args:
        state (ClientState): Contains:
            - retirement_assets (list[dict]): EPF, PPF, NPS, ULIP, etc.
            - liquid_assets (list[dict]): Mutual funds, equity, REITs, etc.
            - fixed_assets (list[dict]): Real estate, bonds, FD, PMS/AIF, ESOPs, etc.
            - client_data.investment_details.financial_summary (list[dict]): Salary, expenses, etc.
            - client_data.client_data.client_age (int): Client’s age

    Returns:
        dict: {
            "financial_overview": {
                "category_percentage": {
                    "retirement": float,  # % of total portfolio
                    "liquid": float,
                    "fixed": float
                },
                "retirement_assets_percent": dict,  # % breakdown within retirement assets
                "liquid_assets_percent": dict,      # % breakdown within liquid assets
                "fixed_assets_percent": dict,       # % breakdown within fixed assets
                "liquidity_ratio": float,           # liquid / total assets
                "liquidity_flag": str,              # 'illiquidity' or 'liquidity ok'
                "flexibility": str,                 # 'low flexibility' or 'medium to high flexibility'
                "spending_behavior": {
                    "saving_ratio": float,          # savings / income
                    "expense_ratio": float,         # expenses / income
                    "red_flag": bool                # True if unhealthy spending/saving pattern
                }
            }
        }

    Purpose:
        - Calculate % allocation across retirement, liquid, and fixed assets.
        - Assess liquidity (ability to meet short-term needs).
        - Evaluate flexibility (redeemable/market-linked vs locked-in/fixed assets).
        - Analyze spending & saving habits with age-based thresholds and red-flag warnings.
    """
    retirement_assets=state['retirement_assets']
    liquid_assets=state['liquid_assets']
    fixed_assets=state['fixed_assets']
    financial_info=state['client_data']['investment_details']['financial_summary']
    age=state['client_data']['client_data']['client_age']

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

    # 1. Liquidity Ratio calculation
    liquidity_ratio = liquid_total / grand_total if grand_total != 0 else 0
    liquidity_flag = 'illiquidity' if liquidity_ratio < 0.15 else 'liquidity ok'

    # 2. Flexibility calculation
    fixed_income_and_real_estate = 0
    market_linked_redeemable = 0

    # Calculate fixed income and real estate assets
    for a in fixed_assets:
        key = list(a.keys())[0]
        if key == 'real_estate_investment':
            fixed_income_and_real_estate += get_fixed_value(a)
        elif key in ['bonds', 'fixed_deposits']:
            fixed_income_and_real_estate += get_fixed_value(a)

    for a in retirement_assets:
        key = list(a.keys())[0]
        val = a[key]
        if key in ['epf', 'ppf']:
            fixed_income_and_real_estate += get_retirement_value(a)
        elif key == 'nps':
            # NPS annuity portion considered fixed income
            annuity_pct = val.get('annuity_allocation_pct', 0)
            if annuity_pct > 0:
                fixed_income_and_real_estate += get_retirement_value(a) * annuity_pct
            # Equity portion considered market linked
            market_linked_redeemable += get_retirement_value(a) * (1 - annuity_pct)
        elif key == 'ulip':
            # ULIP considered market linked
            market_linked_redeemable += get_retirement_value(a)

    # All liquid assets are market linked/redeemable
    market_linked_redeemable += liquid_total
    
    # Add PMS/AIF to market linked assets
    for a in fixed_assets:
        key = list(a.keys())[0]
        if key in ['pms_aif', 'esops']:
            market_linked_redeemable += get_fixed_value(a)

    flexibility = 'medium to high flexibility' if market_linked_redeemable > fixed_income_and_real_estate else 'low flexibility'

    # 3. Spending Behavior calculation
    info = financial_info[0]
    monthly_salary = info.get('monthly_salary', 0)
    monthly_expenses = info.get('monthly_expenses_excl_emis', 0) + info.get('miscellaneous_kids_education_expenses_monthly', 0)
    other_income = info.get('other_income(rental/interest/other)', 0)
    annual_vacation = info.get('annual_vacation_expenses', 0)
    
    total_monthly_income = monthly_salary + other_income
    total_monthly_expenses = monthly_expenses + (annual_vacation / 12)
    monthly_saving = total_monthly_income - total_monthly_expenses
    
    saving_ratio = monthly_saving / total_monthly_income if total_monthly_income > 0 else 0
    expense_ratio = total_monthly_expenses / total_monthly_income if total_monthly_income > 0 else 0

    red_flag = False
    if ((saving_ratio < 0.2 and age < 40) or 
        (saving_ratio < 0.3 and age >= 40) or 
        (expense_ratio > 0.7)):
        red_flag = True

    spending_behavior = {
        'saving_ratio': saving_ratio,
        'expense_ratio': expense_ratio,
        'red_flag': red_flag
    }

    result={
        'category_percentage': {
            'retirement': round(retirement_percent, 2),
            'liquid': round(liquid_percent, 2),
            'fixed': round(fixed_percent, 2)
        },
        'retirement_assets_percent': retirement_assets_percent,
        'liquid_assets_percent': liquid_assets_percent,
        'fixed_assets_percent': fixed_assets_percent,
        'liquidity_ratio': liquidity_ratio,
        'liquidity_flag': liquidity_flag,
        'flexibility': flexibility,
        'spending_behavior': spending_behavior
    }

    return {'financial_overview': result}

# risk assessment using client's assets
def risk_appetite_assessment(state: ClientState): 
    """
    Assess the client’s risk appetite based on equity exposure and years to retirement.

    Logic:
        1. Check if assets include equity (direct equity, equity mutual funds, or similar).
        2. Compute years to retirement = retirement_age - client_age.
        3. Determine risk appetite:
            - If years_to_retire < 5:
                - Equity exposure → "Medium"
                - No equity exposure → "Low"
            - If years_to_retire ≥ 5:
                - Equity exposure → "Medium to High"
                - No equity exposure → "Medium to Low"

    Args:
        state (ClientState): Must include:
            - client_data.investment_details (assets list)
            - client_data.client_data.retirement_age (int)
            - client_data.client_data.client_age (int)

    Returns:
        dict: {
            "risk_appetite": {
                "risk_appetite": str,   # "Low" / "Medium" / "Medium to Low" / "Medium to High"
                "reason": str           # Explanation for the classification
            }
        }

    Purpose:
        Provides a structured risk appetite classification that downstream LLMs or planning
        modules can use in financial recommendations.
    """

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

class GoalState(TypedDict):

        goals: list
        financial_data: dict
        client_age: int 
        sorted_goals: list
        messages: Annotated[List[AnyMessage], operator.add]

def goal_prioritization(state: ClientState):
    """
    Prioritize financial goals using LLM + tool-assisted scoring and sorting.

    Process:
        1. Assign a weight to each goal based on type (retirement, education, housing, etc.).
        2. Adjust weight using client’s age and financial asset mix.
        3. Call `calculate_priority_score` for each goal with weight and target_year.
        4. Attach `priority_score` to each goal while preserving original fields.
        5. Sort all goals using `sort_goals_by_priority` (highest score first).
        6. Return sorted goals aligned with the required schema.

    Args:
        state (ClientState): Must include:
            - goals (list[dict]): Each with keys `goal_name`, `target_year`, `corpus_needed`, etc.
            - client_data.client_data.client_age (int)
            - financial_overview (dict): Includes asset percentages used for housing weight adjustments.

    Returns:
        dict: {
            "sorted_goals": list[dict]  # Original goals + "priority_score", sorted in priority order
        }

    Purpose:
        Ensures client’s goals are ranked systematically for allocation planning,
        considering both subjective importance (weights) and timing (target year).
    """
    goalS=state['goals']
    client_agE=state['client_data']['client_data']['client_age']         #client age is compulsory data
    financial_infO=state['financial_overview']                                           

    def llm_with_tool(state: GoalState):

        # goals=goalS
        # client_age=client_agE
        # financial_info=financial_infO
        messages=state['messages']

        system_text= """
                    You are an expert financial planning assistant. Your task is to prioritize a list of financial goals using a structured process and tools.

                    ## Process:
                    1. **Analyze Each Goal**  
                    - Determine the weight for each goal using the rules provided.

                    2. **Calculate Priority Score**  
                    - For each goal, call the `calculate_priority_score` tool with the parameters:
                        - `weight` (calculated from rules)
                        - `target_year` (from the goal’s data)

                    3. **Attach Priority Score**  
                    - After computing, add a new key `"priority_score": <float>` into each goal’s dictionary.  
                    - Do not alter or rename the other fields.  
                    - Keep the original goal structure intact.

                    4. **Sort Goals**  
                    - Call the `sort_goals_by_priority` tool with the **entire list of goal dictionaries**.  
                    - The input MUST be a Python list of dictionaries, where each dictionary follows this structure:

                        ```python
                        { 'goals':
                         [
                          {
                            "goal_name": <str>,
                            "target_year": <int>,
                            "corpus_needed": <float>,
                            "corpus_gap": <float>,
                            "funded_from": <list>,
                            "surplus": <float>,   
                            "priority_score": <float> # must be added before sorting
                          },
                          ...
                         ]
                        }
                        ```
 
                    - Example valid input:
                        ```python
                        [
                        {"goal_name": "retirement", "corpus_needed": 14697458.23, "corpus_gap": 14697458.23, "target_year": 2045, "funded_from": [...], "surplus": 0, "priority_score": 9.22},
                        {"goal_name": "Aarav Mehta under_graduation", "target_year": 2030, "corpus_needed": 4599498.77, "corpus_gap": 4599498.77, "funded_from": [...], "priority_score": 8.12}
                        ]
                        ```

                    5. **Final Output**  
                    - The output must be the sorted list returned from `sort_goals_by_priority`.  
                    - Each goal must appear in its **original form plus the added `priority_score` field**.  
                    - The order must be descending by priority.
                    **DO ENSURE THAT EACH GOAL APPEAR IN IT'S ORIGINAL FORM, WITH ALL THE FIELDS AS IT IS**
                    ---

                    ## Goal Weighting Rules:
                    - **Base Weights**:
                    - `retirement`: 9  
                    - `under_graduation`: 8  
                    - `post_graduation`: 7  
                    - `residential_house` or `House Renovation`: 5  
                    - `second_property` or `Car`: 2  
                    - `others` (like `Bike`): 4  

                    - **Adjustments**:
                    - **Retirement**: +1 if client age > 45.  
                    - **Education**: assume under-graduation = 18 yrs, post-graduation = 22 yrs. If child age < 5 → subtract 2.  
                    - **Housing**: if `fixed_assets_percent` ≥ 70 → weight = 3; if 50–69% → weight = 4.  

                    ---

                    ## Client Data
                    - Goals: {goals}  
                    - Financial Info: {financial_info}  
                    - Client Age: {client_age}  

                    Now begin the prioritization process.

                    """
        
        message=[SystemMessage(content=system_text)] + messages

        result=goal_llm.invoke(message)

        return {'messages': [result]}
    
    def tools_node(state: GoalState):

        tool_calls=state['messages'][-1].tool_calls
        results=[]
        for t in tool_calls:
            print(f"calling: {t}")
            result=tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print('back to model')
        return {'messages': results}
    
    def check_tool_calling(state: GoalState):

        result=state['messages'][-1]
        return len(result.tool_calls) > 0

    graph_=StateGraph(GoalState)

    graph_.add_node('llm_with_tool', llm_with_tool)
    graph_.add_node('tools_node', tools_node)

    graph_.add_edge(START, 'llm_with_tool')
    graph_.add_conditional_edges('llm_with_tool', check_tool_calling, {True: 'tools_node', False: END})
    graph_.add_edge('tools_node', 'llm_with_tool')

    workflow=graph_.compile() 
    result=workflow.invoke({'messages': [f'Prioritize the goals, goals: {goalS}, Financial Info: {financial_infO}, client age: {client_agE}']})
    
    answer=result['messages'][-1]
    #print(f"answer: {answer}")

    response=structured_goal_llm.invoke(f'Format the goals such that they align with the schema expected. The sorted goal are: {answer}').goals
    sorted_goals=[g.dict() for g in response]

    print(f" The sorted goals are: {sorted_goals}")

    return {'sorted_goals': sorted_goals} 

################################################################################### education web scrapper ######################################################################################################
def education_fees_calculation(state: ClientState): 
    """
    Estimate education costs for each child in the client’s plan.

    For every child, the function:
      - Confirms education destination (domestic/international).
      - Scrapes/estimates tuition fees for the chosen stream.
      - Calculates average costs across top colleges.
      - Updates the plan with college list and fee estimates.

    Parameters
    ----------
    state : ClientState
        Client data containing `education_planning` details.

    Returns
    -------
    None
        Updates `state['client_data']['education_planning']` in-place.
    """
    client_name=state['client_data']['client_data']['name']
    kids_education=state['client_data']['education_planning']

    ug_graph=StateGraph(Education)
    ug_graph.add_node('_edu_plan_llm', _edu_plan_llm)
    ug_graph.add_node('action', action)
    ug_graph.add_node('structured_edu', structured_edu)
    ug_graph.add_node('college_info_scrapper', college_info_scrapper)
    ug_graph.add_edge(START, '_edu_plan_llm')
    ug_graph.add_conditional_edges('_edu_plan_llm', check_condition, {True: 'action', False: 'structured_edu'})
    ug_graph.add_edge('action', '_edu_plan_llm')
    ug_graph.add_edge('structured_edu', 'college_info_scrapper')
    ug_graph.add_edge('college_info_scrapper', END)
    ug_workflow=ug_graph.compile()

    for education_plan in kids_education: 
        
        if education_plan['graduation_destination'] is None:
            education_plan['graduation_destination']='Domestic'
        
        if education_plan['graduation_stream'] is None: 
            education_plan['graduation_stream'] = 'Engineering'

        if education_plan['graduation_destination']=='Domestic' or 'International':

            edu_type='undergraduation'

            ug_final_state=ug_workflow.invoke({'messages': [f"""Hi I am {client_name} and my child's name is {education_plan['name_of_kid']}.
                                          My preference destination for {education_plan['name_of_kid']}'s {edu_type} is {education_plan['graduation_destination']} for the stream {education_plan['graduation_stream']}.
                                          """]})
            
            education_plan['list_of_colleges']=ug_final_state['list_of_colleges']
            education_plan['current_fees_of_graduation']=ug_final_state['avg_college_fees']

            
        if education_plan['post_graduation_destination']=='Domestic' or 'International': 

            if education_plan['post_graduation_destination'] == None:
                education_plan['post_graduation_destination']='Domestic'
            
            if education_plan['post_graduation_stream'] == None:
                education_plan['post_graduation_stream'] = 'MBA'

            edu_type='postgraduation'

            pg_final_state=ug_workflow.invoke({'messages': [f"""Hi I am {client_name} and my child's name is {education_plan['name_of_kid']}.
                                          My preference destination for {education_plan['name_of_kid']}'s edu_type {edu_type} is {education_plan['post_graduation_destination']} for {education_plan['post_graduation_stream']} stream.
                                          """]})
            
            education_plan['list_of_colleges']=pg_final_state['list_of_colleges']
            education_plan['current_fees_of_post_graduation']=pg_final_state['avg_college_fees']
    
    print(f'education_plan \n : {state['client_data']['education_planning']}')


def check_for_kid(state: ClientState):

    return state['client_data']['client_data']['if_any_kids']==True

#################################################################################### liability logic #############################################################################################################

# def plan_prepayments(state: ClientState) -> dict:
#     """
#     Inputs:
#       - liabilities: list of loans with keys:
#             type, outstanding_balance, interest_rate (annual), emi_amount,
#             is_under_penalty_period (bool), time_left_to_come_out_of_penalty_period(months) [ignored here]
#       - monthly_surplus: total extra cash you *could* pay monthly across loans
#       - liquid_pool: available liquid lump-sum pool
#       - use_liquid_pool: whether to deploy some of liquid_pool as prepayment
#       - liquid_pool_percent: 0..100 percent of liquid_pool to use (evenly across eligible loans)
#     Rules:
#       - Loans are sorted by descending interest rate
#       - Extra monthly prepayment is split equally across *eligible* loans (not under penalty)
#       - Liquid pool prepayment (if enabled) is split equally across eligible loans
#       - Prepayment (monthly or lump-sum) is applied only to loans not under penalty
#       - No “snowball reallocation” when a loan closes; extra assigned to that loan is treated as freed when it closes

#     Returns:
#       {
#         "per_loan": [
#           {
#             "type": ...,
#             "baseline_months": int,
#             "baseline_close_date": date.isoformat,
#             "accelerated_months": int,
#             "accelerated_close_date": date.isoformat,
#             "interest_saved": float,
#             "monthly_extra_applied": float,  # 0 if not eligible
#             "lump_sum_applied": float,       # 0 if not eligible or pool not used
#             "freed_by_year": {year: amount}, # amount = EMI + monthly_extra_applied
#           },
#           ...
#         ],
#         "freed_timeline": {year: total_amount_freed_that_year},
#         "assumptions": {...}
#       }
#     """
#     client_data = state['client_data']
#     liabilities = client_data['liabilities']
#     monthly_surplus_init=state['client_data']['investment_details']['financial_summary'][0]['monthly_surplus']
#     threshold_monthly_surplus = monthly_surplus_init*0.5
#     if state['goal_funding']['ending_monthly_surplus'] < threshold_monthly_surplus:
#         monthly_surplus = state['goal_funding']['ending_monthly_surplus']
#     else:   
#         monthly_surplus = threshold_monthly_surplus
#     liquid_pool = state['goal_funding']['ending_liquid_pool']
#     today = date.today()
#     use_liquid_pool = True
#     liquid_pool_percent = 40

#     # 1) Sort loans by interest rate desc
#     loans = sorted(liabilities, key=lambda x: float(x["interest_rate"]), reverse=True)

#     # Determine eligible loans for prepayment (not under penalty)
#     eligible_idx = [i for i, L in enumerate(loans) if not L.get("is_under_penalty_period", False)]
#     n_eligible = len(eligible_idx)

#     # Compute per-loan monthly extra (only for eligible loans)
#     per_loan_monthly_extra = (monthly_surplus / n_eligible) if n_eligible > 0 else 0.0

#     # Compute per-loan lump sum from liquid pool (if used and eligible)
#     total_lump_to_use = (liquid_pool * (liquid_pool_percent / 100.0)) if use_liquid_pool else 0.0
#     per_loan_lump_sum = (total_lump_to_use / n_eligible) if n_eligible > 0 else 0.0

#     per_loan_results = []
#     freed_aggregate = {}

#     for idx, L in enumerate(loans):
#         P = float(L["outstanding_balance"])
#         i_annual = float(L["interest_rate"])
#         M = float(L["emi_amount"])
#         r = i_annual / 12.0

#         # Baseline months and closure date (no prepayment)
#         base_months = _months_to_close(P, r, M)
#         if base_months is None:
#             # Non-amortizing at current EMI; mark and skip computations beyond baseline structure
#             per_loan_results.append({
#                 "type": L["type"],
#                 "baseline_months": None,
#                 "baseline_close_date": None,
#                 "accelerated_months": None,
#                 "accelerated_close_date": None,
#                 "interest_saved": None,
#                 "monthly_extra_applied": 0.0,
#                 "lump_sum_applied": 0.0,
#                 "freed_by_year": {},
#                 "note": "EMI <= monthly interest; loan is not amortizing at current EMI."
#             })
#             continue

#         base_close_dt = _add_months(today, base_months)

#         # Apply prepayment only if not under penalty
#         eligible = (idx in eligible_idx)
#         E = per_loan_monthly_extra if eligible else 0.0
#         LUMP = per_loan_lump_sum if eligible else 0.0

#         # If we apply a lump sum now, reduce principal immediately
#         P_after_lump = max(0.0, P - LUMP)

#         # If lump sum wipes the loan
#         if P_after_lump <= 0:
#             acc_months = 0
#             acc_close_dt = today
#             # Interest baseline from "now" is M*base_months - P
#             base_interest_from_now = M * base_months - P
#             acc_interest_from_now = -LUMP  # paid principal; interest from now is zero
#             saved_interest = base_interest_from_now - acc_interest_from_now
#             monthly_freed_amount = M + E  # EMI + assigned extra (E may be 0 if ineligible)
#             freed = _freed_by_year(acc_close_dt, monthly_freed_amount)
#         else:
#             # With extra monthly payment (if eligible), payment becomes M + E
#             pay_with_extra = M + E

#             acc_months = _months_to_close(P_after_lump, r, pay_with_extra)
#             if acc_months is None:
#                 # If still non-amortizing even after extra, treat as baseline only
#                 acc_months = base_months
#             acc_close_dt = _add_months(today, acc_months)

#             # Interest comparison from "now"
#             # Baseline interest from now: total payments minus principal
#             base_interest_from_now = M * base_months - P
#             acc_interest_from_now = pay_with_extra * acc_months - P_after_lump
#             saved_interest = base_interest_from_now - acc_interest_from_now

#             monthly_freed_amount = M + E
#             freed = _freed_by_year(acc_close_dt, monthly_freed_amount)

#         # Merge freed into aggregate
#         for yr, amt in freed.items():
#             freed_aggregate[yr] = freed_aggregate.get(yr, 0.0) + amt

#         per_loan_results.append({
#             "type": L["type"],
#             "baseline_months": _safe(base_months),
#             "baseline_close_date": base_close_dt.isoformat(),
#             "accelerated_months": _safe(acc_months),
#             "accelerated_close_date": acc_close_dt.isoformat(),
#             "interest_saved": float(saved_interest),
#             "monthly_extra_applied": float(E),
#             "lump_sum_applied": float(LUMP),
#             "freed_by_year": {k: float(v) for k, v in freed.items()},
#         })

#     result={
#         "per_loan": per_loan_results,
#         "freed_timeline": {k: float(v) for k, v in sorted(freed_aggregate.items())},
#         "assumptions": {
#             "today": today.isoformat(),
#             "sorted_by": "interest_rate desc",
#             "monthly_surplus": float(monthly_surplus),
#             "considered_only_not_under_penalty_for_prepayment": True,
#             "monthly_extra_split_among_eligible": float(per_loan_monthly_extra),
#             "liquid_pool_total": float(liquid_pool),
#             "use_liquid_pool": bool(use_liquid_pool),
#             "liquid_pool_percent_used": float(liquid_pool_percent),
#             "per_loan_lump_sum_if_used": float(per_loan_lump_sum),
#             "no_snowball_reallocation": True
#         }
#     }
    
#     loan_prepayed_times=state['loan_prepayed_times']+1

#     return {'liability_allocation': result, 'freed_timeline': result['freed_timeline'], 
#             'used_monthly_surplus': result['assumptions']['monthly_surplus'], 'used_liquid_surplus': result['assumptions']['use_liquid_pool'], 'EMI_allocated': True, 'loan_prepayed_times': loan_prepayed_times}

def plan_prepayments(state: ClientState):
    """
    Same inputs as before.

    Important behavior changes:
    - After splitting monthly surplus and lump-sum evenly across eligible loans,
      we cap each loan's assigned amounts so that we never allocate more than the
      loan currently needs. Lump-sum is applied first; monthly extra is capped
      to the remaining principal after lump application.
    - The returned top-level dictionary now includes:
        "allocated_monthly_surplus": total monthly extra actually allocated (sum of per-loan E),
        "allocated_lump_sum": total lump-sum actually applied (sum of per-loan LUMP).
    - No redistribution of unused amounts is performed (no snowball).
    """
    client_data = state['client_data']
    liabilities = client_data['liabilities']
    monthly_surplus_init=state['client_data']['investment_details']['financial_summary'][0]['monthly_surplus']
    threshold_monthly_surplus = monthly_surplus_init*0.5 
    if state['goal_funding'][-1]['ending_monthly_surplus'] < threshold_monthly_surplus:
        monthly_surplus = state['goal_funding'][-1]['ending_monthly_surplus']*0.40                # we consider only 40% of monthly surplus.
        unused_monthly_surplus=state['goal_funding'][-1]['ending_monthly_surplus'] - monthly_surplus
    else: 
        monthly_surplus = threshold_monthly_surplus*0.40
        unused_monthly_surplus = monthly_surplus_init - monthly_surplus
    
    liquid_pool = state['goal_funding'][-1]['ending_liquid_pool'] 
    today=date.today()
    use_liquid_pool = True, 
    liquid_pool_percent = 40

    # 1) Sort loans by interest rate desc
    loans = sorted(liabilities, key=lambda x: float(x["interest_rate"]), reverse=True)

    # Determine eligible loans for prepayment (not under penalty)
    eligible_idx = [i for i, L in enumerate(loans) if not L.get("is_under_penalty_period", False)]
    n_eligible = len(eligible_idx)

    # Compute per-loan monthly extra (only for eligible loans)
    per_loan_monthly_extra = (monthly_surplus / n_eligible) if n_eligible > 0 else 0.0

    # Compute per-loan lump sum from liquid pool (if used and eligible)
    total_lump_to_use = (liquid_pool * (liquid_pool_percent / 100.0)) if use_liquid_pool else 0.0
    per_loan_lump_sum = (total_lump_to_use / n_eligible) if n_eligible > 0 else 0.0

    per_loan_results = []
    freed_aggregate = {}

    # Track actual allocated totals
    total_allocated_monthly = 0.0
    total_allocated_lump = 0.0

    for idx, L in enumerate(loans):
        P = float(L["outstanding_balance"])
        i_annual = float(L["interest_rate"])
        M = float(L["emi_amount"])
        r = i_annual / 12.0

        # Baseline months and closure date (no prepayment)
        base_months = _months_to_close(P, r, M)
        if base_months is None:
            # Non-amortizing at current EMI; mark and skip computations beyond baseline structure
            per_loan_results.append({
                "type": L["type"],
                "baseline_months": None,
                "baseline_close_date": None,
                "accelerated_months": None,
                "accelerated_close_date": None,
                "interest_saved": None,
                "monthly_extra_applied": 0.0,
                "lump_sum_applied": 0.0,
                "freed_by_year": {},
                "note": "EMI <= monthly interest; loan is not amortizing at current EMI."
            })
            continue

        base_close_dt = _add_months(today, base_months)

        # Apply prepayment only if not under penalty
        eligible = (idx in eligible_idx)

        # Assigned (pre-safety-cap) values
        assigned_monthly = per_loan_monthly_extra if eligible else 0.0
        assigned_lump = per_loan_lump_sum if eligible else 0.0

        # --- Safety caps: do not allocate more than outstanding principal ---
        # Apply lump first (one-time immediate reduction). Cap it to principal.
        LUMP = min(assigned_lump, P) if eligible else 0.0
        P_after_lump = max(0.0, P - LUMP)

        # If lump wiped the loan, no monthly extra is needed/allocated to this loan.
        if P_after_lump <= 0:
            E = 0.0
        else:
            # Cap monthly extra to remaining principal after lump.
            # This avoids assigning a monthly extra greater than the outstanding amount.
            E = min(assigned_monthly, P_after_lump) if eligible else 0.0

        # Update allocated totals
        total_allocated_lump += LUMP
        total_allocated_monthly += E

        # If lump sum wipes the loan immediately
        if P_after_lump <= 0:
            acc_months = 0
            acc_close_dt = today
            # Baseline interest from "now" is M*base_months - P
            base_interest_from_now = M * base_months - P
            # After lump we paid principal via lump; no future interest
            acc_interest_from_now = 0.0
            saved_interest = base_interest_from_now - acc_interest_from_now
            monthly_freed_amount = M + E  # E will be 0 here
            freed = _freed_by_year(acc_close_dt, monthly_freed_amount)
        else:
            # With extra monthly payment (if eligible), payment becomes M + E
            pay_with_extra = M + E

            acc_months = _months_to_close(P_after_lump, r, pay_with_extra)
            if acc_months is None:
                # If still non-amortizing even after extra, treat as baseline only
                acc_months = base_months
            acc_close_dt = _add_months(today, acc_months)

            # Interest comparison from "now"
            # Baseline interest from now: total payments minus principal
            base_interest_from_now = M * base_months - P
            # Note: if some lump was applied, that reduced principal paid by future payments
            acc_interest_from_now = pay_with_extra * acc_months - P_after_lump
            saved_interest = base_interest_from_now - acc_interest_from_now

            monthly_freed_amount = M + E
            freed = _freed_by_year(acc_close_dt, monthly_freed_amount)

        # Merge freed into aggregate
        for yr, amt in freed.items():
            freed_aggregate[yr] = freed_aggregate.get(yr, 0.0) + amt

        per_loan_results.append({
            "type": L["type"],
            "baseline_months": _safe(base_months),
            "baseline_close_date": base_close_dt.isoformat(),
            "accelerated_months": _safe(acc_months),
            "accelerated_close_date": acc_close_dt.isoformat(),
            "interest_saved": float(saved_interest),
            "monthly_extra_assigned": float(assigned_monthly),   # what was assigned before capping
            "monthly_extra_applied": float(E),                   # what was actually applied (capped)
            "lump_sum_assigned": float(assigned_lump),           # what was assigned before capping
            "lump_sum_applied": float(LUMP),                     # what was actually applied (capped)
            "freed_by_year": {k: float(v) for k, v in freed.items()},
        })

    loan_prepayed_times=state['loan_prepayed_times']+1

    result={
        "per_loan": per_loan_results,
        "freed_timeline": {k: float(v) for k, v in sorted(freed_aggregate.items())},
        "allocated_monthly_surplus": float(total_allocated_monthly),
        "allocated_lump_sum": float(total_allocated_lump),
        "assumptions": {
            "today": today.isoformat(),
            "sorted_by": "interest_rate desc",
            "monthly_surplus": float(monthly_surplus),
            "considered_only_not_under_penalty_for_prepayment": True,
            "monthly_extra_split_among_eligible": float(per_loan_monthly_extra),
            "liquid_pool_total": float(liquid_pool),
            "use_liquid_pool": bool(use_liquid_pool),
            "liquid_pool_percent_used": float(liquid_pool_percent),
            "per_loan_lump_sum_if_used": float(per_loan_lump_sum),
            "no_snowball_reallocation": True
        },
        "unused_monthly_surplus":unused_monthly_surplus
    }

    total_monthly_extra_applied = sum(loan.get('monthly_extra_applied', 0.0) for loan in result.get('per_loan', []))
    total_lump_sum_applied = sum(loan.get('lump_sum_applied', 0.0) for loan in result.get('per_loan', []))
    
    return {'liability_allocation': [result], 'freed_timeline': [result['freed_timeline']], 
            'used_monthly_surplus': [total_monthly_extra_applied], 'used_liquid_surplus': [total_lump_sum_applied], 'EMI_allocated': True, 'loan_prepayed_times': loan_prepayed_times, 'unused_monthly_surplus': [unused_monthly_surplus] }

def check_for_pre_payment(state: ClientState):

    if state['loan_prepayed_times']<4 and state['EMI_allocation']==True:
        return True
    #print(f"state['optimal_loan_allocation']['unused_monthly_surplus'] : {state['optimal_loan_allocation']['unused_monthly_surplus']}")
    
    elif len(state['liability_allocation'])>0 :
        return False
    else: 
        return "END"

###################################################################################### liability logic ###################################################################################################################

###################################################################################### freed timelines node #########################################################################################################################

def freed_emi_by_year(state: ClientState):
    """
    Returns {year: total_monthly_emi_freed} based on each loan's *natural* closure date.
    Notes:
      - Ignores prepayment/penalty windows; uses current EMI schedule.
      - Values are the *monthly EMI amounts* that become available after closure.
    """
    liabilities =  state['liabilities']
    today = date.today()
    year=today.year
    month=today.month
    day=today.day
    data_str=f"{year}-{month}-{day}"

    freed: dict[int, float] = {}
    for loan in liabilities:
        P = float(loan["outstanding_balance"])
        r_annual = float(loan["interest_rate"])
        emi = float(loan["emi_amount"])

        n_months = _months_to_close(P, r_annual, emi)
        if n_months is None:
            # Skip (or handle however you want) loans that won't amortize at current EMI
            continue

        close_dt = _add_months(today, n_months)
        freed[close_dt.year] = freed.get(close_dt.year, 0.0) + emi 

    result=present_loans_no_allocation(liabilities, date.today())

    return {'freed_timeline': [freed], 'liability_allocation': [result]}
###################################################################################### freed timelines node #########################################################################################################################
#liabilities=final_state['liabilities']
####################################################################################### no clear goal #######################################################################################################################################

def invest_monthly_surplus(state: ClientState):
    """
    Invest monthly_surplus into three components:
      - 30% -> Debt funds/BAFs for 5 years
      - 40% -> Hybrid/Equity MF for 10 years
      - 30% -> Equity MF or NPS until retirement_age

    Returns a JSON string with details and future values (assumes end-of-month SIPs).
    Uses today's date (datetime.date.today()) as start.
    """

    #monthly_surplus=state['final_unused_monthly_surplus'] state['optimal_loan_allocation']['unused_monthly_surplus']
    monthly_surplus=state['optimal_loan_allocation']['unused_monthly_surplus']
    age = state['client_data']['client_data']['client_age']
    retirement_age = state['client_data']['client_data']['retirement_age']

    # --- Configuration: chosen constant annual rates (modifiable) ---
    rate_debt = 0.07    # 7.0% p.a. for Debt (5-year)
    rate_hybrid = 0.11  # 11.0% p.a. for Hybrid (10-year)
    rate_equity = 0.11  # 11.0% p.a. for Equity/NPS until retirement (blended conservative)

    # --- allocations ---
    alloc = {
        "debt": 0.30,
        "hybrid": 0.40,
        "equity_nps": 0.30
    }

    today = date.today()
    current_year = today.year

    # helper: FV of monthly contributions (ordinary annuity) at monthly rate i, n months:
    def fv_monthly(contribution_monthly: float, annual_rate: float, months: int) -> float:
        if months <= 0:
            return 0.0
        monthly_i = annual_rate / 12.0
        if isclose(monthly_i, 0.0):
            return contribution_monthly * months
        # FV of ordinary annuity (payments at end of period)
        return contribution_monthly * ((1 + monthly_i)**months - 1) / monthly_i

    # Determine durations in months & maturity years
    years_debt = 5
    months_debt = years_debt * 12
    maturity_debt_year = current_year + years_debt

    years_hybrid = 10
    months_hybrid = years_hybrid * 12
    maturity_hybrid_year = current_year + years_hybrid

    years_equity = retirement_age - age
    if years_equity < 0:
        # Already at/after retirement: treat as zero-duration
        years_equity = 0
    months_equity = years_equity * 12
    maturity_equity_year = current_year + years_equity

    # monthly contributions
    c_debt = monthly_surplus * alloc["debt"]
    c_hybrid = monthly_surplus * alloc["hybrid"]
    c_equity = monthly_surplus * alloc["equity_nps"]

    # compute FVs at their maturation points
    fv_debt = fv_monthly(c_debt, rate_debt, months_debt)
    fv_hybrid = fv_monthly(c_hybrid, rate_hybrid, months_hybrid)
    fv_equity = fv_monthly(c_equity, rate_equity, months_equity)

    # invested amounts (sum of contributions) for transparency
    invested_debt = c_debt * months_debt
    invested_hybrid = c_hybrid * months_hybrid
    invested_equity = c_equity * months_equity

    # totals
    total_invested = invested_debt + invested_hybrid + invested_equity
    total_fv = fv_debt + fv_hybrid + fv_equity

    result = {
        "input": {
            "leftover_monthly_surplus": monthly_surplus,
            "age": age,
            "retirement_age": retirement_age,
            "start_date": today.isoformat()
        },
        "assumptions": {
            "rate_debt_p_a": rate_debt,
            "rate_hybrid_p_a": rate_hybrid,
            "rate_equity_nps_p_a": rate_equity,
            "allocation": alloc,
            "compounding": "monthly (SIP, payments at month end)"
        },
        "components": {
            "debt": {
                "monthly_contribution": round(c_debt, 2),
                "investment_duration_years": years_debt,
                "maturity_year": maturity_debt_year,
                "months": months_debt,
                "total_contributed": round(invested_debt, 2),
                "future_value_at_maturity": round(fv_debt, 2)
            },
            "hybrid": {
                "monthly_contribution": round(c_hybrid, 2),
                "investment_duration_years": years_hybrid,
                "maturity_year": maturity_hybrid_year,
                "months": months_hybrid,
                "total_contributed": round(invested_hybrid, 2),
                "future_value_at_maturity": round(fv_hybrid, 2)
            },
            "equity_nps": {
                "monthly_contribution": round(c_equity, 2),
                "investment_duration_years": years_equity,
                "maturity_year": maturity_equity_year,
                "months": months_equity,
                "total_contributed": round(invested_equity, 2),
                "future_value_at_maturity": round(fv_equity, 2)
            }
        },
        "summary": {
            "total_contributed": round(total_invested, 2),
            "total_future_value": round(total_fv, 2)
        }
    }

    return {'extra_monthly_surplus_investment_plan': result}

######################################################################################## no clear goal ###############################################################################################################################################

#################################################### optimal strategy ###############################################################################################

def choose_optimal_strategy(state: ClientState):

    for index, goals in enumerate(state['goal_funding']):
        for goal in goals['goals']:
            for allocation in goal['funded_from']:
                if allocation['type']=='postponed' or allocation['type']=='unfunded':
                    optimal_goal_allocation=state['goal_funding'][max(index-1, 0)]
                    optimal_loan_allocation=state['liability_allocation'][max(index-1, 0)]
                    unused_monthly_surplus=state['liability_allocation'][max(index, 0)]['unused_monthly_surplus']

                    print(f"{state['unused_monthly_surplus']} : state['unused_monthly_surplus']")
                    print(f"unused_monthly_surplus: {unused_monthly_surplus}")

                    return {'optimal_goal_allocation': optimal_goal_allocation, 'optimal_loan_allocation': optimal_loan_allocation, 'final_unused_monthly_surplus':unused_monthly_surplus}  
                    # state['optimal_goal_allocation']=optimal_goal_allocation
                    # state['optimal_loan_allocation']=optimal_loan_allocation
                    #state['final_unused_monthly_surplus']=float(unused_monthly_surplus)
                    optimal=True
    
    optimal_goal_allocation=state['goal_funding'][-1]
    optimal_loan_allocation=state['liability_allocation'][-1]
    unused_monthly_surplus=state['liability_allocation'][-1]['unused_monthly_surplus']
        
    print(f"unused_monthly_surplus: {unused_monthly_surplus}")
    print(f"{state['unused_monthly_surplus']} : state['unused_monthly_surplus']")
    # state['optimal_goal_allocation']=optimal_goal_allocation
    # state['optimal_loan_allocation']=optimal_loan_allocation
    return {'optimal_goal_allocation': optimal_goal_allocation, 'optimal_loan_allocation': optimal_loan_allocation, 'final_unused_monthly_surplus':unused_monthly_surplus}

######################################################################################## choose optimal strategy ####################################################################################################

####################################### no allocation loan #######################################

from math import log, ceil, isfinite
from datetime import date, timedelta
import json

TODAY = date.today()

def months_to_close(principal: float, annual_rate: float, payment: float) -> int | None:
    """Return number of months to close an amortizing loan with fixed payment.
    If payment <= interest-only payment, returns None."""
    if principal <= 0:
        return 0
    r_month = annual_rate / 12.0
    if payment <= principal * r_month:
        return None
    # n = -log(1 - r_month*principal/payment) / log(1 + r_month)
    # rearranged safe formula:
    try:
        n = - (log(1 - r_month * principal / payment) / log(1 + r_month))
    except ValueError:
        return None
    if not isfinite(n):
        return None
    return ceil(n)

def close_date_from_months(start_date: date, months: int) -> str:
    # add months approximately by advancing month/year
    y = start_date.year
    m = start_date.month + months
    y += (m - 1) // 12
    m = ((m - 1) % 12) + 1
    d = min(start_date.day, 28)  # keep safe day to avoid month-end issues; user had day 12 in examples
    return date(y, m, start_date.day).isoformat()

def present_loans_no_allocation(loans: list, today: date = TODAY, monthly_surplus: float = 0.0):
    per_loan = []
    for ln in loans:
        principal = float(ln.get('outstanding_balance', 0))
        rate = float(ln.get('interest_rate', 0))
        emi = float(ln.get('emi_amount', 0))
        baseline_months = months_to_close(principal, rate, emi)
        baseline_close_date = None
        if baseline_months is not None:
            baseline_close_date = close_date_from_months(today, baseline_months)
        # Build loan dict with allocation fields zero/null per user's request
        loan_entry = {
            'type': ln.get('type'),
            'baseline_months': baseline_months,
            'baseline_close_date': baseline_close_date,
            'accelerated_months': 0,
            'accelerated_close_date': None,
            'interest_saved': 0.0,
            'monthly_extra_assigned': 0.0,
            'monthly_extra_applied': 0.0,
            'lump_sum_assigned': 0.0,
            'lump_sum_applied': 0.0,
            'freed_by_year': {}  # empty dict (no freed amounts)
        }
        per_loan.append(loan_entry)
    result = {
        'per_loan': per_loan,
        'freed_timeline': {},
        'allocated_monthly_surplus': 0.0,
        'allocated_lump_sum': 0.0,
        'assumptions': {
            'today': today.isoformat(),
            'sorted_by': 'interest_rate desc',
            'monthly_surplus': monthly_surplus,
            'considered_only_not_under_penalty_for_prepayment': True,
            'monthly_extra_split_among_eligible': 0.0,
            'liquid_pool_total': None,
            'use_liquid_pool': False,
            'liquid_pool_percent_used': 0.0,
            'per_loan_lump_sum_if_used': 0.0,
            'no_snowball_reallocation': True
        },
        'unused_monthly_surplus': monthly_surplus
    }
    return result
################################## no allocation loan #########################################

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
graph.add_node('calculate_asset_percentages_and_ratios', calculate_asset_percentages_and_ratios)
graph.add_node('risk_appetite_assessment', risk_appetite_assessment)
graph.add_node('goal_prioritization', goal_prioritization)
graph.add_node('education_fees_calculation',education_fees_calculation)
graph.add_node('plan_prepayments', plan_prepayments)
graph.add_node('freed_emi_by_year',freed_emi_by_year)
graph.add_node('invest_monthly_surplus', invest_monthly_surplus)
graph.add_node('choose_optimal_strategy', choose_optimal_strategy)

# add edges
graph.add_edge(START, 'calculate_age') 
graph.add_conditional_edges('calculate_age', check_for_kid, {True: 'education_fees_calculation', False: 'goals_future_value' })
graph.add_edge('education_fees_calculation', 'calculate_education_funding')            
graph.add_edge('calculate_education_funding', 'goals_future_value')
graph.add_edge('goals_future_value', 'calculate_retirement_corpus')
graph.add_edge('calculate_retirement_corpus', 'calculate_all_retirement_investments')     
graph.add_edge('calculate_all_retirement_investments', 'retirement_goal')
graph.add_edge('retirement_goal', 'asset_basket_classification') 
graph.add_edge('asset_basket_classification', 'risk_appetite_assessment')
graph.add_edge('risk_appetite_assessment', 'calculate_total_asset_value')
graph.add_edge('calculate_total_asset_value', 'calculate_fixed_assets_value')
graph.add_edge('calculate_fixed_assets_value', 'calculate_asset_percentages_and_ratios')
graph.add_edge('calculate_asset_percentages_and_ratios', 'goal_prioritization')
graph.add_edge('goal_prioritization','freed_emi_by_year') # updating freed_timeline, liability_allocation
graph.add_edge('freed_emi_by_year' ,'plan_goals') # only update the goal_allocation
graph.add_conditional_edges('plan_goals', check_for_pre_payment, {True: 'plan_prepayments', False : 'choose_optimal_strategy', 'END': END}) # 
graph.add_edge('choose_optimal_strategy', 'invest_monthly_surplus')
graph.add_edge('plan_prepayments', 'plan_goals') 
graph.add_edge('invest_monthly_surplus', END)
#graph.add_edge('plan_goals', END) 

# define workflow
workflow=graph.compile()

client_data= {
  "client_data": {
    "name": "Rahul Mehta",
    "date_of_birth": "1985-06-15",
    #"employment_type": "salaried", 
    "spouse_name": "Priya Mehta",
    "spouse_dob": "1987-09-25",          # will be added
    "if_any_kids": True,
    "children":[
    {"child_name": "Aarav Mehta",
    "child_dob": "2012-05-10"},
    {"child_name": "Raghav Mehta",
    "child_dob": "2013-05-10"}              
    ], 
    "retirement_age":60  # autocalculated if not mentioned then assume 55
  },
  "investment_details":   
  { 
    "financial_summary": [{
      "monthly_salary": 400000,
      "monthly_expenses_excl_emis": 90000,
      "other_income(rental/interest/other)": 35000,         # autocalculated
      "lump_sum_available": 800000,
      "miscellaneous_kids_education_expenses_monthly": 5000,
      "annual_vacation_expenses": 200000, 
      "emergency_fund_maintained": 300000,  # will be added 
    }], 
    "real_estate_investment": [
      {
        #"property_name": "Mumbai Apartment",
        #"type_of_property": "residential",
        "current_market_value": 15000000,
        #"year_of_purchase": 2016,
        #"purchase_price": 10000000,
        "rental_income": 35000, 
        #"loan_on_property": False
      } 
    ], 
    
    "retirement_investments": {
    
    "ulip":[
        { 
            "name_of_ulip":"retirement",
            "commencement_date_of_ulip_policy_1": "25-04-2025",  
            "monthly_premium_amount": 4000, 
            "final_year_of_premium_to_be_paid": 2035,
            "expected_rate_of_return":0.09,     #autocalculated
            "term": 12, 
            "maturity_year": 2037, 
            "maturity_value": 600000 
        } 
    ], 
    "epf": [{
      "current_value": 1500000,
      "employee_employer_contribution_monthly": 15000,
      "interest_rate": 0.085,             # autocalculated
      #"maturity_date": "2045-06-15",       
      #"maturity_year": 2045             
      #"pre_mature_withdrawal": "Upon unemployment for more than 2 months or upon attaining the age of 58 years or upon the death of the member, appointed nominee gets the benefit."
    }],
    "ppf": [{
      "current_value": 500000,
      #"year_of_account_start": 2018,
      "annual_contribution": 150000,
      "interest_rate": 0.075,             # autocalculated
      #"lock_in_end_year": 2033,         
      #"pre_mature_withdrawal": "Allowed after completion of the 5th year of account opening."
    }],
    "nps": [{
      "scheme_name": "retirement",
      "maturity_year": 2045, 
      "monthly_contribution": "monthly",    #autocalculated    
      "monthly_contribution": 6000,                
      "expected_corpus_growth_rate": 0.10,   #autocalculated
      "annuity_allocation_pct": 0.4,         # autocalculated
      "expected_annuity_rate": 0.065,        # autocalculated
      "current_value": 800000
    }, 
    {
      "scheme_name": "retirement",
      "maturity_year": 2045, 
      "monthly_contribution": "monthly",    #autocalculated    
      "monthly_contribution": 5000,                
      "expected_corpus_growth_rate": 0.10,   #autocalculated
      "annuity_allocation_pct": 0.4,         # autocalculated
      "expected_annuity_rate": 0.065,        # autocalculated
      "current_value": 600000
    }]
    }, 
    "bonds": [ 
      {
        "name_of_bond":"python_bond",
        #"purchase_date": "2022-05-01",
        "investment_amount": 400000, 
        "contribution_per_annum": 4000,
        #"current_value": 500000,
        #"purchase_price": 98000, 
        #"face_value": 100000,
        #"coupon_payments": "annually",
        #"coupon_rate": 0.07,
        "maturity_date": "2027-05-01",
        "interest_rate": 0.07   
        #"years_to_maturity": 5,
        #"sale_price_before_maturity": None,
        #"reinvestment_rate_for_coupons": 0.06,
        #"holding_period_months": 60
      }
    ],    
    "mutual_funds": [
      { 
        #"investment_mode": "SIP",
        #"start_date": "2020-01-01",
        "current_value": 0,
        "expected_annual_return": 0.12,  # autocalulated
        #"maturity_date": "2040-01-01",   
        "sip_amount": 15000 
        #"sip_frequency": "monthly"
      },
      { 
        #"investment_mode": "Lumpsum",
        #"start_date": "2019-06-01",
        "current_value": 800000,
        "expected_annual_return": 0.11,  # autocalculated
        #"maturity_date": "2030-06-01",
        "sip_amount": 0,
        #"sip_frequency": None  
      }
    ], 
    "direct_equity": [{
      "portfolio_value": 1200000,   
      #"risk_appetite": "High",
      #"average_annual_returns_rate": 0.14
    }], 
    "reits": [
      {
        #"reit_name": "Embassy Office Parks REIT",
        #"invested_amount": 500000,
        "current_value": 550000,
        #"investment_start_date": "2021-08-01",
        #"returns_received_dividend": 45000,
        #"dividends_paid_on_avg": 4,
        #"avg_dividends_pay": 11250
      }
    ], 
    "pms_aif": [
      {
        #"provider_name": "Motilal Oswal",
        #"product_name": "PMS",
        #"invested_amount": 1000000,
        #"start_date": "2022-02-01",
        #"lock_in_period_years": 3,
        "current_value": 1150000,
        # "performance": {
        #   "one_month": 0.015,
        #   "three_month": 0.04,
        #   "one_year": 0.12
        # }
      } 
    ],
    "esops": [{
      #"company_name": "TechNova Ltd",
      "vested_esops_value": 300000,
      "unvested_esops_value": 200000
    }],
    # "ncd_govt": [
    #   {
    #     "name_of_instrument": "NABARD Tax-Free Bonds",
    #     "credit_rating": "AAA",
    #     "coupon_rate": 0.065,
    #     "coupon_frequency": "annual",
    #     "face_value": 1000,
    #     "purchase_price": 980,
    #     "number_of_units_hold": 500,
    #     "invested_amount": 490000,
    #     "purchase_date": "2021-03-01",
    #     "maturity_date": "2031-03-01",
    #     "call_option": None,
    #     "put_option": None,
    #     "reinvestment_rate": 0.06
    #   },
    #   {
    #     "name_of_instrument": "NABARD Tax-Free Bonds",
    #     "credit_rating": "AAA",
    #     "coupon_rate": 0.065,
    #     "coupon_frequency": "annual",
    #     "face_value": 1000,
    #     "purchase_price": 980,
    #     "number_of_units_hold": 500,
    #     "invested_amount": 490000,
    #     "purchase_date": "2021-03-01",
    #     "maturity_date": "2031-03-01",
    #     "call_option": None,
    #     "put_option": None,
    #     "reinvestment_rate": 0.06
    #   }
    # ],
    "fixed_deposits": [ 
      {
        "name_of_bank": "state_bank_of_india",
        "principal_amount": 300000, 
        "interest_rate": 0.065,
        #"start_date": "2022-07-01",
        "maturity_date": "07-2035" # mm/yyyy 
        #'asset_tag':'liquid'
      }
    ]
  }, 
  "financial_goals": [
    {
      "goal_name": "Car",
      "capital_required_today": 250000,
      "target_year": 2045,
      "amount_saved_for_goal": 40000
    },
    {
      "goal_name": "Bike",
      "capital_required_today": 150000,
      "target_year": 2030,
      "amount_saved_for_goal": 50000
    },
    {
      "goal_name": "House Renovation",
      "capital_required_today": 150000,
      "target_year": 2028,
      "amount_saved_for_goal": 38000
    }
  ],
  "liabilities": [
        {
        "type": "Home loan",
        #"original_amount": 8200000,                
        "outstanding_balance": 495000,           
        "interest_rate": 0.0725,                  
        #"interest_type": "fixed",
        "emi_amount": 8500,                      
        #"emi_frequency": "monthly",
        #"start_date": "2019-03-01",
        #"end_date": "2039-03-01",
        "is_under_penalty_period": False,
        "time_left_to_come_out_of_penalty_period(months)": 5 #months
    }, 
    {
        "type": "Car loan",
        #"original_amount": 750000,               
        "outstanding_balance": 410000,           
        "interest_rate": 0.083,                   
        #"interest_type": "fixed",
        "emi_amount": 6414, 
        #"emi_frequency": "monthly",
        #"start_date": "2020-07-01",
        #"end_date": "2040-07-01",
        "is_under_penalty_period": False,
        "time_left_to_come_out_of_penalty_period(months)": 5 #months
    },
    {
        "type": "Personal Loan",
        #"original_amount": 900000,               
        "outstanding_balance": 720000,          
        "interest_rate": 0.089,                   
        #"interest_type": "fixed",
        "emi_amount": 8040,
        #"emi_frequency": "monthly",
        #"start_date": "2022-11-01",
        #"end_date": "2042-11-01",
        "is_under_penalty_period": False,
        "time_left_to_come_out_of_penalty_period(months)": 5 #months
    }
  ],
  "education_planning": [
    {
      "name_of_kid": "Aarav Mehta",
      "graduation_stream": "Medical",
      "graduation_destination": "Domestic", #enum(International/Domestic)
      #"current_fees_of_graduation": 0, # auto calculated
      "fund_allocated_for_graduation": 0,  
      "post_graduation_stream": "M.tech",  
      "post_graduation_destination": 'International',   #enum(International/Domestic)
      #"current_fees_of_post_graduation": 80000, # auto calculated
      "fund_allocated_for_post_graduation": 0
    #   "scheme_for_education": [{
    #   "scheme_name": "Aarav Engineering HDFC Child Gift Fund",
    #   "start_date": "2020-04-01",
    #   "end_date": "2029-04-01",
    #   "monthly_investment": 10000,
    #   "interest_rate": 0.12
    #   }]
      },
      {                                                                                                                                                                             
      "name_of_kid": "Raghav Mehta",
      "graduation_stream": "Engineering",
      "graduation_destination": 'Domestic',  #enum(International/domestic)
      #"current_fees_of_graduation": 0,  # auto calculated
      "fund_allocated_for_graduation": 0,
      "post_graduation_stream": "M.tech",
      "post_graduation_destination": "International",         #enum(international/domestic)
      #"current_fees_of_post_graduation": 90000,   # auto calculated 
      "fund_allocated_for_post_graduation": 0 
    #   "scheme_for_education": [{
    #   "scheme_name": "Raghav Engineering HDFC Child Gift Fund",
    #   "start_date": "2022-04-01",
    #   "end_date": "2035-04-01",
    #   "monthly_investment": 10000,
    #   "interest_rate": 0.12
    #   }]
    }]
  , 
  "life_insurance": [ 
    {
      "insurance_nam]e": "LIC Jeevan Anand",
      "current_value": 800000,
      "premium_payment_frequency": "annual",
      "premium_fee": 50000,
      "start_date": "2015-03-01",
      "maturity_date": "2035-03-01",
      "maturity_value": 1500000,
      "surrender_value": 600000,
      "total_premium_paid": 500000,
      "penalty_for_withdrawal": True
    } 
  ] 
} 

# execute the workflow
initial_state={'client_data': client_data, 'EMI_allocated': False, 'loan_prepayed_times': 0, 'used_monthly_surplus':[0]} 
#print("hello") 

final_state=workflow.invoke(initial_state)
print(json.dumps(final_state['liability_allocation'], indent=2))
final_state['client_data']['client_data'] 
# final_state['sorted_goals'] 
# final_state['liability_allocation'][1]
# final_state['liabilities'] 
# final_state['unused_monthly_surplus'] 
# # final_state['goal_funding'][0].keys()
# final_state['goal_funding'][0]['goals'][2]
# final_state['optimal_goal_allocation']
# len(final_state['goal_funding'])
# len(final_state['liability_allocation'])
# final_state['extra_monthly_surplus_investment_plan']
# final_state['optimal_loan_allocation']
# optimal_goal_allocation : dict
# optimal_loan_allocation : dict
# final_unused_monthly_surplus : float 

# final_state['loan_prepayed_times']
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
# final_state['financial_overview']
# final_state['client_data']['client_data']
# final_state['risk_appetite']
# #years_to_retire=final_state['client_data']['client_data']['retirement_age']-final_state['client_data']['client_data']['client_age']
# final_state['client_data']['investment_details']['financial_summary']
# final_state['client_data']['education_planning']


_goals=[{'goals': [{'goal_name': 'Retirement', 'target_year': 2045, 'corpus_needed': 14697458.23, 'corpus_gap': 0.0, 'funded_from': [{'type': 'sip_from_surplus', 'monthly': 24787.12737678599, 'from_year': 2025, 'to_year': 2045, 'months': 240, 'fv_contribution': 14697458.23}]}, {'goal_name': 'Bike', 'target_year': 2030, 'corpus_needed': 1926812.87, 'corpus_gap': 0.0, 'funded_from': [{'type': 'sip_from_surplus', 'monthly': 26049.73344911619, 'from_year': 2025, 'to_year': 2030, 'months': 60, 'fv_contribution': 1926812.87}]}, {'goal_name': 'House Renovation', 'target_year': 2028, 'corpus_needed': 128074.4, 'corpus_gap': 0.0, 'funded_from': [{'type': 'sip_from_surplus', 'monthly': 3138.6326536293186, 'from_year': 2025, 'to_year': 2028, 'months': 36, 'fv_contribution': 128074.39999999998}]}, {'goal_name': 'Car', 'target_year': 2045, 'corpus_needed': 7748738.68, 'corpus_gap': 0.0, 'funded_from': [{'type': 'freed_sip', 'monthly': 3138.6326536293186, 'from_year': 2028, 'to_year': 2045, 'months': 204, 'fv_contribution': 1364287.9487050094}, {'type': 'freed_sip', 'monthly': 18327.945309303308, 'from_year': 2030, 'to_year': 2045, 'months': 180, 'fv_contribution': 6384450.731294991}]}], 'ending_monthly_surplus': 269357.8398538019, 'ending_liquid_pool': 2550000.0, 'ending_freed_sip_schedule': {2045: 46253.70533971862, 2030: 7721.788139812881}},
{'goals': [{'goal_name': 'Retirement', 'target_year': 2045, 'corpus_needed': 14697458.23, 'corpus_gap': 0.0, 'funded_from': [{'type': 'sip_from_partial_surplus', 'monthly': 23333.33333333333, 'from_year': 2025, 'to_year': 2045, 'months': 240, 'fv_contribution': 13835435.095819503}, {'type': 'lumpsum_from_liquid', 'principal_used_today': 184945.51817925213, 'from_year': 2025, 'to_year': 2045, 'years': 20, 'fv_contribution': 862023.1341804974}]}, {'goal_name': 'Bike', 'target_year': 2030, 'corpus_needed': 1926812.87, 'corpus_gap': 0.0, 'funded_from': [{'type': 'lumpsum_from_liquid', 'principal_used_today': 1311356.463150381, 'from_year': 2025, 'to_year': 2030, 'years': 5, 'fv_contribution': 1926812.8700000003}]}, {'goal_name': 'House Renovation', 'target_year': 2028, 'corpus_needed': 128074.4, 'corpus_gap': 0.0, 'funded_from': [{'type': 'lumpsum_from_liquid', 'principal_used_today': 101669.58796931361, 'from_year': 2025, 'to_year': 2028, 'years': 3, 'fv_contribution': 128074.4}]}, {'goal_name': 'Car', 'target_year': 2045, 'corpus_needed': 7748738.68, 'corpus_gap': 0.0, 'funded_from': [{'type': 'postponed', 'postponed_years': 5, 'from_year': 2045, 'to_year': 2050}, {'type': 'freed_sip', 'monthly': 23333.33333333333, 'from_year': 2045, 'to_year': 2050, 'months': 60, 'fv_contribution': 1725889.7122493212}, {'type': 'lumpsum_from_liquid', 'principal_used_today': 879443.7878778575, 'from_year': 2025, 'to_year': 2050, 'years': 25, 'fv_contribution': 6022848.967750679}]}], 'ending_monthly_surplus': 0.0, 'ending_liquid_pool': 72584.64282319555, 'ending_freed_sip_schedule': {2050: 23333.33333333333}}]

#_goals[1]

for index, goals in enumerate(_goals):
    for goal in goals['goals']:
        for allocation in goal['funded_from']:
            if allocation['type']=='postponed' or allocation['type']=='unfunded':
                optimal_allocation=_goals[max(index-1, 0)]













