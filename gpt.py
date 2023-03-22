import os
import openai


class ChatApp:
    def __init__(self, system_text, api_path="api_key.txt", model="gpt-3.5-turbo-0301", sequence=False):
        self.messages = [{"role": "system", "content": system_text}]
        self.sequence = sequence
        self.max_tokens = 20 if sequence else 2

        with open(api_path, "r") as file:
            openai.api_key = file.readline().strip()

        self.model = model

    def chat(self, message):
        self.messages.append({"role": "user", "content": message})
        response = openai.ChatCompletion.create(
            model=self.model, messages=self.messages, max_tokens=self.max_tokens, temperature=0.5
        )

        self.messages.append(
            {"role": "assistant", "content": response["choices"][0]["message"]["content"]}
        )
        # if self.sequence:
        #     return re
        # else: 
        return response["choices"][0]["message"]["content"].strip()
