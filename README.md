# Financial Planning AI Agent

This project implements an AI-powered financial planning agent designed to analyze a user's financial data, assess their risk appetite, plan for various financial goals including education and retirement, and provide a comprehensive overview of their asset allocation. The agent utilizes a Langchain-based graph structure to orchestrate various financial calculations and assessments.

## Project Overview

The core of this project is an intelligent agent that processes client financial data (provided in JSON format) through a series of interconnected steps. Each step represents a specific financial calculation or analytical task. The agent aims to:

1.  **Calculate Ages**: Determine the current ages of all individuals mentioned in the client's data.
2.  **Goals Future Value**: Calculate the future value of financial goals and identify any funding gaps or surpluses.
3.  **Education Funding**: Plan and allocate funds for children's education, considering existing schemes and future costs.
4.  **Retirement Corpus Calculation**: Determine the required retirement corpus using multiple methodologies and evaluate existing retirement investments.
5.  **Asset Classification**: Categorize and value client assets into liquid, fixed, and retirement categories.
6.  **Goal Funding Allocation**: Strategically allocate available funds (surplus, freed SIPs, liquid assets) to achieve financial goals, with an option to postpone goals if necessary.
7.  **Asset Percentage Allocation**: Provide a detailed breakdown of asset allocation by category and within each category.
8.  **Risk Appetite Assessment**: Evaluate the client's risk appetite based on their asset holdings and years to retirement.

## How to Run

To run this financial planning agent, follow these steps:

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/aakash-haldankar-quantal/langgraph_armstrong.git
    cd https://github.com/aakash-haldankar-quantal/langgraph_armstrong.git
    ```

2.  **Set up Environment Variables**:
    Create a `.env` file in the project root directory and add your API keys for Azure OpenAI and Groq:
    ```
    AZURE_API_KEY=your_azure_api_key
    AZURE_API_BASE=your_azure_api_base
    AZURE_API_VERSION=your_azure_api_version
    AZURE_DEPLOYMENT_NAME=your_azure_deployment_name
    GROQ_API_KEY=your_groq_api_key
    ```

3.  **Install Dependencies**:
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    (Ensure `requirements.txt` lists `langchain`, `langgraph`, `langchain-openai`, `langchain-groq`, `pydantic`, `python-dotenv`, etc.)

4.  **Prepare Client Data**:
    Ensure your client financial data is correctly formatted in the `client_data_updated.py` file, which is imported by the main script.

5.  **Execute the Script**:
    Run the main Python script:
    ```bash
    python app.py
    ```

The script will execute the financial planning workflow, and the `final_state` variable will hold all the calculated and processed information about the client's financial situation and goal planning. You can then inspect `final_state` to view the outputs of each step.
