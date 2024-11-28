from openai import AzureOpenAI
from langchain_core.messages import AIMessage
import yaml
import warnings
warnings.filterwarnings("ignore")

import os
# from dotenv import load_dotenv
# load_dotenv(dotenv_path='./.env')

class RoleBasedAgent:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            )
        dir = os.path.dirname(os.path.abspath(__file__))
        self.prompts_file = os.path.join(dir, "prompts.yaml")
        self.prompts = self._load_prompts()
        
        self.prompts = self._load_prompts()

    def _load_prompts(self):
    
        if not os.path.exists(self.prompts_file):
            raise FileNotFoundError(f"Prompts file not found at {self.prompts_file}")
        
        with open(self.prompts_file, "r") as file:
            prompts = yaml.safe_load(file)
        
        return prompts

    def _get_prompt_for_role(self, role):

        role_prompt = self.prompts.get("role_prompts", {}).get(role)
        
        if not role_prompt:
            raise ValueError(f"Invalid role '{role}'. No corresponding prompt found.")
        
        return role_prompt["prompt"]

    def perform_task(self, state):

        role = state['role']
        role_prompt = self._get_prompt_for_role(role)

        a = len(state["messages"])
        msgs = ""
        for i in range(a):
            msgs += str(i + 1) + ") " + (str(state["messages"][i])) + " "
        
        p = f"""
                You are provided with the entire chat history: {msgs}

                Your task is to focus on the most recent user query and work specifically on that. 
                Use the previous messages as context if they are relevant, but prioritize the most recent user query in your response.
                """

        prompt = role_prompt.format(messages=p)

        response = self.client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[{"role": "user", "content": prompt}]
        )

        response_content = response.choices[0].message.content
        response_content += f" sender: {role} Agent"

        state = {
            "messages": [AIMessage(content=response_content)],
        }

        return state