from openai import AzureOpenAI
import re

from openai import AzureOpenAI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import Sequence, TypedDict, Annotated, Literal, Optional
import operator
from langchain_core.messages import BaseMessage
from functools import partial
import requests
import yaml
# from langgraph.checkpoint.memory import MemorySaver
import warnings
warnings.filterwarnings("ignore")

import os
# from dotenv import load_dotenv
import PIL.Image
import matplotlib.pyplot as plt

from langchain_openai import AzureChatOpenAI

import sqlite3
from langchain import OpenAI
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI
from langchain.agents.agent import AgentExecutor,AgentOutputParser

from txt2sqlagent import TextToSQLAgent
from analystagent import AnalystAgent
from rolebasedagent import RoleBasedAgent


# load_dotenv(dotenv_path='./.env')
try:
    client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
                )

    client_1 = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_KEY"),
                api_version="2024-05-01-preview",
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
                )
    
except EnvironmentError as e:
    print(f"Error initializing AzureOpenAI: {e}")
    raise

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    role: str


agent_descriptions = {
    "Text-to-SQL Agent": "This agent is equipped with direct access to a database. "
                         "It can interpret user queries related to database tables, generate the necessary "
                         "SQL queries, execute them, and retrieve the required data from the database. "
                         "This agent must be called first if data is required and is not provided",

    "Analyst Agent": "This agent is powered by a code interpreter and specializes in data analysis and visualization. "
                     "It can analyze datasets, generate visualizations such as plots, and perform exploratory data analysis (EDA). "
                      "It can also perform mathematical calculations. "
                     "The output can be either textual (e.g., statistical summaries) or graphical (e.g., charts or plots). "
                     "Both types of output indicate the task is complete.",

"Role-based Agent": "This agent can take on different roles based on role provided by the supervisor and should only be called when absolutely necessary. " 
                        "There should be strictly only one call per user query of role-based Agent "
                        "It can act as a Query Transformation Agent when the user's query is complex and contains multiple subqueries that need to be handled individually. " 
                        "This role has access to the database schema used by the Text-to-SQL Agent and assist the Text-to-SQL Agent with simpler subqueries. "
                        "This role should only be invoked if the query contains multiple components that require decomposition. the Next Agent should always be the text-to-SQL agent " 
                        "The supervisor agent must ensure that the Role-based Agent is only and rarely be selected when no simpler alternative is available "
                        "When selecting the Role-based Agent, the supervisor must specify the role i.e. Query Transformation Agent."                                        
    }

agents = list(agent_descriptions.keys())
options = ['FINISH'] + agents
role_options = ['Query Transformation']
agent_descriptions_text = "\n".join([f"{agent}: {description}" for agent, description in agent_descriptions.items()])

class routeResponse(BaseModel):
    next: Literal[tuple(options)]
    role: Optional[Literal[tuple(role_options)]] = None

    
system_prompt = (
    "You are a supervisor responsible for managing the workflow between"
    " the following agents:" + ', '.join(agents) +". Below are the descriptions of these agents:"
    "\n\n" + agent_descriptions_text + "\n\n"
    "Your task is to handle the user's request by routing it to the appropriate agent."
    " If the 'Role-based Agent' is selected, you must also specify the role: " + ','.join(role_options) + ". "
    "Once an agent completes its task and responds, you will evaluate the result."

    "The Data Analyst Agent may return either textual outputs (e.g., data summaries, insights) or graphical outputs (e.g., plots, charts). "
    "In both cases, treat the response as a complete result and take further decisions accordingly." 
    "There could be a follow up questions on the response of Analyst Agent so you need to send the request back to Analyst Agent in case of follow up questions."

    "\n\n IMPORTANT INSTRUCTION: "

    "- You MUST NOT invoke the same agent more than twice. STRICT LIMIT of two invocations per agent. "
    "- After you receive output from any agent twice, you MUST use the result and proceed with the next most appropriate agent. "
    "- Do NOT call the same agent beyond two attempts. "
    "FAILURE TO FOLLOW THIS RULE WILL RESULT IN ERRORS IN THE PROCESS.\n\n"
    
    " If additional steps are needed, route the next task to the appropriate"
    " agent. Continue this process until all tasks related to the user's"
    " request are completed. When all tasks are done, respond with 'FINISH'"
    " to conclude the conversation."

    " Follow these steps:"
    " 1. Analyze the user's request and identify the first task."
    " 2. Route the task to the appropriate agent."
    " 3. If the Role-based Agent is selected, specify the appropriate role: " + ', '.join(role_options) +
    " 3. Wait for the agent's response and evaluate the result."
    " 4. If further tasks are required, route the next task to the appropriate agent."
    " 5. Repeat this process until all tasks are completed."
    " 6. Once all tasks are completed, respond with 'FINISH' to end the conversation."
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Based on the conversation above, decide the next step:"
            " Who should act next? Select one from: {options}"
            " If you select 'Role-based Agent', specify the role: " + ', '.join(role_options) + ". "
            " Respond with 'next: <option>' where <option> is one of the following: {options}. Example: next: 'FINISH'"
        ),
    ]
).partial(options=str(options), agents=", ".join(agents))

try:
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"), 
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        )

except EnvironmentError as e:
    print(f"Error initializing AzureChatOpenAI: {e}")
    raise

def supervisor_agent(state):   
    supervisor = prompt | llm.with_structured_output(routeResponse)
    response = supervisor.invoke(state)
    return response


graph = StateGraph(AgentState)

text_to_sql_agent = TextToSQLAgent()
analyst_agent = AnalystAgent()
role_based_agent = RoleBasedAgent()

graph.add_node("Supervisor", supervisor_agent)
graph.add_node("Text-to-SQL Agent", text_to_sql_agent.fetch)
graph.add_node("Analyst Agent", analyst_agent.analyze)
graph.add_node("Role-based Agent", role_based_agent.perform_task)

conditional_map = {
    'Text-to-SQL Agent': 'Text-to-SQL Agent', 
    'Analyst Agent': 'Analyst Agent', 
    'Role-based Agent': 'Role-based Agent', 
    'FINISH': END
    }

graph.add_edge(START, "Supervisor")
for agent in agents:
    graph.add_edge(agent, "Supervisor")


graph.add_conditional_edges("Supervisor", lambda x: x["next"], conditional_map)

# memory = MemorySaver()
# app = graph.compile(checkpointer=memory)
app = graph.compile()

