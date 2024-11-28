from openai import AzureOpenAI
from langchain_core.messages import AIMessage
import requests
import os
import warnings
warnings.filterwarnings("ignore")
import PIL.Image
import matplotlib.pyplot as plt

# from dotenv import load_dotenv
# load_dotenv(dotenv_path='./.env')


class AnalystAgent:
    def __init__(self):
        self.endpoint = "https://pepgenx-code-interpreter.azurewebsites.net/load_and_run"
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
        
        payload = {
            "api_key": os.getenv("AZURE_OPENAI_KEY"),
            "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "api_version": "2024-05-01-preview",
            "model": "pepgenx-agentic-gpt4o",
            "role": "user",
            "content": msgs,
            "assistant_id": "asst_MvRMP9S57pDQcuhNBJUeG7fo",
            "thread_id": "thread_jXrPIHwfTVnGZcMrWjRzdxCS"
            }
        
        response = requests.post(self.endpoint, json=payload)
        response.raise_for_status() 
        response_data = response.json()
        final_response = ""
        for i in response_data['data'][0]['content']:
            if "image_file" in i:
                image_id = i['image_file']['file_id']
                final_response += image_id + " ; "
                image_data = self.client.files.content(image_id)

                image_data_bytes = image_data.read()
                with open("./my-image.png", "wb") as file:
                    file.write(image_data_bytes)
                img = PIL.Image.open('my-image.png')
                plt.imshow(img)
                plt.axis('off')
                plt.show()

            if "text" in i:
                text = i['text']['value']
                final_response += text
        final_response += " sender: Analyst Agent"
        state = {
            "messages": [AIMessage(content=final_response)],
        } 
        
        return state