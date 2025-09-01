print("hello")
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
      "monthly_salary": 180000,
      "monthly_expenses_excl_emis": 90000,
      "other_income(rental/interest/other)": 35000,         # autocalculated
      "lump_sum_available": 800000,
      "miscellaneous_kids_education_expenses_monthly": 5000,
      "annual_vacation_expenses": 200000, 
      "emergency_fund_maintained": 300000,  # will be added 
      "monthly_surplus": 105000          # autocalculated
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
        "current_value": None,
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
        "sip_amount": None,
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
      "capital_required_today": 2500000,
      "target_year": 2045,
      "amount_saved_for_goal": 400000
    },
    {
      "goal_name": "Bike",
      "capital_required_today": 250000,
      "target_year": 2030,
      "amount_saved_for_goal": 50000
    },
    {
      "goal_name": "House Renovation",
      "capital_required_today": 1500000,
      "target_year": 2028,
      "amount_saved_for_goal": 200000
    }
  ],
  "liabilities": [
        {
        "type": "Home loan",
        #"original_amount": 8200000,                
        "outstanding_balance": 4950000,           
        "interest_rate": 0.0725,                  
        #"interest_type": "fixed",
        "emi_amount": 64811,                      
        #"emi_frequency": "monthly",
        #"start_date": "2019-03-01",
        #"end_date": "2039-03-01",
        "is_under_penalty_period": True,
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
        "is_under_penalty_period": True,
        "time_left_to_come_out_of_penalty_period(months)": 5 #months
    }
  ],
  "education_planning": [
    {
      "name_of_kid": "Aarav Mehta",
      "graduation_stream": "Engineering",
      "graduation_destination": "US", #enum(International/Domestic)
      "current_fees_of_graduation": 5000000, # auto calculated
      "fund_allocated_for_graduation": None,  
      "post_graduation_stream": "MBA",  
      "post_graduation_destination": "UK",   #enum(International/Domestic)
      "current_fees_of_post_graduation": 3000000, # auto calculated
      "fund_allocated_for_post_graduation": 500000 ,
      "scheme_for_education": [{
      "scheme_name": "Aarav Engineering HDFC Child Gift Fund",
      "start_date": "2020-04-01",
      "end_date": "2029-04-01",
      "monthly_investment": 10000,
      "interest_rate": 0.12
      }]},
      {
      "name_of_kid": "Raghav Mehta",
      "graduation_stream": "Engineering",
      "graduation_destination": "UK",  #enum(International/domestic)
      "current_fees_of_graduation": 4000000,  # auto calculated
      "fund_allocated_for_graduation": 1000000,
      "post_graduation_stream": "MBA",
      "post_graduation_destination": "UK",         #enum(international/domestic)
      "current_fees_of_post_graduation": 2000000,   # auto calculated 
      "fund_allocated_for_post_graduation": 400000 ,
      "scheme_for_education": [{
      "scheme_name": "Raghav Engineering HDFC Child Gift Fund",
      "start_date": "2022-04-01",
      "end_date": "2035-04-01",
      "monthly_investment": 10000,
      "interest_rate": 0.12
      }]
    }]
  ,
  "life_insurance": [
    {
      "insurance_name": "LIC Jeevan Anand",
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


from datetime import datetime
from datetime import date

def calculate_age(client_data):
    """ 
     calculate_age(): calculates current age of individuals mentioned in the json object.
     input argument: client data in json format
     output: client data with populated current ages of all the mentioned individuals
    """
    
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
            
    return client_data

def future_value(current_value:float, number_of_years:int, annual_growth_rate:float=0.09):
    return current_value*(1+annual_growth_rate)**(number_of_years)

def calculate_required_sip(target_value, annual_rate, years):
        """Calculates the monthly SIP required to reach a target future value."""
        if years <= 0 or target_value <= 0:
            return 0.0
        monthly_rate = annual_rate / 12
        months = int(years * 12)
        if monthly_rate == 0:
            return target_value / months
        return (target_value * monthly_rate) / (((1 + monthly_rate) ** months - 1))

def goals_future_value(client_data):
    """
    function: 
    input argument: 
    output argument: 
    """
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
                print(f"considering the surplus collected from previous goals: {surplus}, the new gap is: {goal_gap}")
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

    return client_data

client_data1=calculate_age(client_data)
client_data2=goals_future_value(client_data1)
client_data2['financial_goals']
    
client_data2['education_planning']  

def calculate_education_funding(client_data):
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
    
    return client_data

client_data3 = calculate_education_funding(client_data2)
client_data3['education_planning_summary']

def calculate_retirement_corpus(client_data, life_expectancy=85, inflation_rate=0.06):
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
    
    return retirement_plan

retirement_analysis = calculate_retirement_corpus(client_data3) 



from datetime import datetime
from collections import defaultdict
from math import ceil

current_date=datetime.today()
current_year=current_date.year

# ----------  FV helpers (same formulas you already use) ----------
def ulip_future_value(pmt, start, end, r_annual):
    start, end = [datetime.strptime(d, "%d-%m-%Y") for d in (start, end)]
    n = (end.year - start.year) * 12 + (end.month - start.month)
    if n <= 0:                                  # safety
        return 0.0
    r_m = r_annual / 12
    fv = pmt * (((1 + r_m) ** n - 1) / r_m) * (1 + r_m)
    return round(fv, 2)

def epf_future_value(cur_val, m_contrib, r, years_left):
    if years_left <= 0:
        return cur_val
    fv_cur = cur_val * (1 + r) ** years_left
    
    annual_c = m_contrib * 12
    fv_c   = sum(annual_c * (1 + r) ** (years_left - i - 1)           # end-of-year flow
                 for i in range(years_left))
    return round(fv_cur + fv_c, 2)

def ppf_future_value(cur_val, annual_c, r, n):
    if n <= 0:
        return cur_val
    fv_c   = annual_c * (((1 + r) ** n - 1) / r) * (1 + r)            # start-of-year flow
    fv_cur = cur_val * (1 + r) ** n
    return round(fv_cur + fv_c, 2)

def nps_future_value(cur_val, m_contrib, r_annual, months_left):
    if months_left <= 0:
        return cur_val
    r_m = r_annual / 12
    fv_c   = m_contrib * (((1 + r_m) ** months_left - 1) / r_m) * (1 + r_m)
    fv_cur = cur_val * (1 + r_m) ** months_left
    return round(fv_cur + fv_c, 2)

# ----------  master calculator (UPDATED for new field names) ----------
def calculate_all_retirement_investments(retirement_investments,
                                         current_age,
                                         retirement_age=60,
                                         today=datetime.today()):
    """
    Accepts ANY number of schemes per category and returns:
      1. detailed per-scheme output
      2. per-category totals
      3. grand total
    """
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

    return {
        "schemes"         : dict(results),
        "category_totals" : category_totals,
        "grand_total"     : round(grand_total, 2)
    }

# Suppose Rahul is 40 now
output = calculate_all_retirement_investments(client_data3['investment_details']['retirement_investments']  , current_age=client_data3['client_data']['client_age'] )
print(output["category_totals"])
print("Grand total corpus:", output["grand_total"])

def retirement_goal(required_retirement_corpus: float, estimated_retirement_corpus: float, sip_annual_rate: float=0.09, retirement_age: int=60):

    current_date=date.today()
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

    return result

retirement_goals=retirement_goal(retirement_analysis['recommendation']['recommended_corpus'], output["grand_total"])

education_planning=[]
for goal in client_data3['education_planning_summary']:
    education_planning.append({'goal_name': goal['name'] + " " + goal['type'], 'target_year': goal['target_year'], 'corpus_needed': goal['final_gap'], 'corpus_gap': goal['final_gap'], 'funded_from':[] })
education_goals=education_planning

financial_goals=[]
for goal in client_data3['financial_goals']:
    financial_goals.append({'goal_name': goal['goal_name'] , 'target_year': goal['target_year'], 'corpus_needed': goal['goal_gap'], 'corpus_gap': goal['goal_gap'], 'funded_from':[] })
other_goals=financial_goals

total_goals=[retirement_goals]+education_goals+other_goals

######################################################################################## asset classification #################################################################################################

def asset_basket_classification(client_data):
    
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
            
    return retirement_assets, liquid_assets, fixed_assets

retirement_bucket, liquid_bucket, fixed_bucket=asset_basket_classification(client_data3)

######################################################################################## asset classification #################################################################################################

######################################################################################### liquid and fixed assets pool ####################################################################################################

def calculate_total_asset_value(assets_list):
    """
    Calculates the total value of all assets from a given list of
    asset dictionaries. It assumes all provided assets are liquid.

    Args:
        assets_list (list): A list of dictionaries, where each dictionary
                            represents an asset.

    Returns:
        dict: A dictionary containing the 'total_asset_value'.
    """
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
        "total_asset_value": total_asset_value
    }

# Calculate the total value
liquid_basket_value = calculate_total_asset_value(liquid_bucket)
liquid_pool=liquid_basket_value['total_asset_value']



def calculate_fixed_assets_value(assets_list):
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
        "total_fixed_asset_value": total_fixed_asset_value
    }

# Example usage with your provided data:
fixed_assets = [
    {'real_estate_investment': {'current_market_value': 15000000, 'rental_income': 35000, 'asset_tag': 'fixed_asset', 'asset_id': 1}},
    {'bonds': {'name_of_bond': 'python_bond', 'investment_amount': 400000, 'contribution_per_annum': 4000, 'maturity_date': '2027-05-01', 'interest_rate': 0.07, 'asset_tag': 'fixed_asset', 'asset_id': 11}},
    {'pms_aif': {'current_value': 1150000, 'asset_tag': 'fixed_asset', 'asset_id': 16}},
    {'esops': {'vested_esops_value': 300000, 'unvested_esops_value': 200000, 'asset_tag': 'fixed_asset', 'asset_id': 17}},
    {'fixed_deposits': {'name_of_bank': 'state_bank_of_india', 'principal_amount': 300000, 'interest_rate': 0.065, 'maturity_date': '07-2035', 'asset_tag': 'fixed_asset', 'asset_id': 18}}
]

# Calculate the total value
calculated_values = calculate_fixed_assets_value(fixed_bucket)
fixed_pool=calculated_values['total_fixed_asset_value']

########################################################################################### liquid and fixed assets pool #########################################################################################################














