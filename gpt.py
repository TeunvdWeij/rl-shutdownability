import os
import openai


class ChatApp:
    def __init__(self, system_text, api_path="api_key.txt", model="gpt-3.5-turbo-0301"):
        self.messages = [{"role": "system", "content": system_text}]

        with open(api_path, "r") as file:
            openai.api_key = file.readline().strip()

        self.model = model

    def chat(self, message):
        self.messages.append({"role": "user", "content": message})
        response = openai.ChatCompletion.create(
            model=self.model, messages=self.messages, max_tokens=2, temperature=0
        )

        self.messages.append(
            {"role": "assistant", "content": response["choices"][0]["message"]["content"]}
        )
        return response["choices"][0]["message"]["content"].strip()
