from langchain_openai import AzureChatOpenAI
import os
os.getenv("LOGGING_LEVEL")

import sqlite3
# from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits.sql.base import create_sql_agent
# from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
# from langchain.sql_database import SQLDatabase
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage
import warnings
warnings.filterwarnings("ignore")

# from dotenv import load_dotenv
# load_dotenv(dotenv_path='./.env')

class TextToSQLAgent():

    def __init__(self) -> None:
        #load llm
        self.llm = AzureChatOpenAI(azure_deployment= os.getenv("AZURE_OPENAI_DEPLOYMENT"),azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                            openai_api_key=os.getenv("AZURE_OPENAI_KEY"), openai_api_type="azure",temperature=0.9)

        #load_db
        parent_path = os.path.dirname(os.path.abspath(__file__))
        sqlite_db_path = os.path.join(parent_path, "northwind.db")
        sqlite_uri = f"sqlite:///{sqlite_db_path}"


        self.db = SQLDatabase.from_uri(sqlite_uri)
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        self.sql_agent = self.create_agent()

    def create_agent(self):

        agent = create_sql_agent(
            llm=self.llm,
            toolkit=self.toolkit,
            verbose=False,
            # agent_type="zero-shot-react-description",
            agent_type="openai-tools",
            handle_parsing_errors=False,
            agent_executor_kwargs={"return_intermediate_steps": True, "handle_parsing_errors":True})

        return agent

    def fetch(self,state):
        try:
            a = len(state["messages"])
            msgs = ""
            for i in range(a):
                msgs += str(i+1) + ") " + (str(state["messages"][i])) + " "
            prompt = f"""
                You are provided with the entire chat history: {msgs}

                Your task is to focus on the most recent user query and work specifically on that. 
                Use the previous messages as context if they are relevant, but prioritize the most recent user query in your response.
                """
            result = self.sql_agent.invoke(prompt)

            if 'intermediate_steps' in result:
                intermediate_response = result['intermediate_steps'][-1][1]
            else:
                intermediate_response = ""
            final_output = result['output']   
            response = intermediate_response + final_output  + " sender: Text-to-SQL Agent"
            state = {
                "messages": [AIMessage(content=response)],
            }

            return state
        except KeyError as e:
            print(f"KeyError: {str(e)} - Check the result structure.")
            raise
        except sqlite3.OperationalError as e:
            print(f"SQLite Error: {str(e)} - Check your SQL query and database schema.")
            raise
        except Exception as e:
            print(f"Error during fetch: {str(e)}")
            raise