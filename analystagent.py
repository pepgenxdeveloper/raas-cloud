from openai import AzureOpenAI
from langchain_core.messages import AIMessage
import requests
import os
import warnings
warnings.filterwarnings("ignore")
import PIL.Image
import matplotlib.pyplot as plt
import json

# from dotenv import load_dotenv
# load_dotenv(dotenv_path='./.env')


class AnalystAgent:
    def __init__(self):
        self.client = AzureOpenAI(
                                    api_key=os.getenv("AZURE_OPENAI_KEY"),
                                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                                    api_version="2024-05-01-preview",
                                )

    def analyze(self, state):
        a = len(state["messages"])
        msgs = ""
        for i in range(a):
            msgs += str(i+1) + ") " + (str(state["messages"][i])) + " "
        

        assistant_id= "asst_MvRMP9S57pDQcuhNBJUeG7fo"
        thread_id= "thread_jXrPIHwfTVnGZcMrWjRzdxCS"
        
        attachments = []
            # print("I am sending this attachment: ", attachments)


        self.client.beta.assistants.retrieve(assistant_id= assistant_id)

        self.client.beta.threads.messages.create(
                    thread_id=thread_id,
                    role="user",
                    content=msgs,
                    attachments = attachments
                    )
        run = self.client.beta.threads.runs.create_and_poll(
                    thread_id=thread_id,
                    assistant_id=assistant_id,
                )
        

        if run.status == 'completed':
            messages = self.client.beta.threads.messages.list(thread_id=thread_id)
            output = messages
        elif run.status == 'requires_action':
            output = run.status
            pass
        else:
            output = run.status
        response_data = output.model_dump_json()
        response_data = json.loads(response_data)
        final_response = ""
        if "content" in response_data['data'][0]:
            for i in response_data['data'][0]['content']:
                if "image_file" in i:
                    image_id = i['image_file']['file_id']
                    final_response += image_id + " ; "

                if "text" in i:
                    text = i['text']['value']
                    final_response += text

        else:
            final_response = "Unable to parse response: 'content' key not found."
            
        final_response += " sender: Analyst Agent"
        state = {
            "messages": [AIMessage(content=final_response)],
        } 
        
        return state