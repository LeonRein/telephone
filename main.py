from assistant import Assistant
from tts import TTS
import asyncio
from asyncio import Task

def read_system_prompt():
    with open("system_prompt.txt", "r") as file:
        return file.read().strip()
    
def chat(user_input: str, assistant: Assistant):
    for response in assistant.chat(user_input):
        print(response, end="", flush=True)
    print()  # For newline after the assistant's response
    print("You: ", end="", flush=True)


def main():
    system_prompt = read_system_prompt()
    assistant = Assistant(system_prompt)
    assistant.start()

    tts = TTS()
    tts.start()

    assistant.on_partial_response(lambda text: print(text, end="", flush=True))
    assistant.on_sentence_response(lambda text: tts.speak(text))
    assistant.on_finish(lambda: print("\nYou: ", end="", flush=True))

    assistant.chat(None)
    while True:
        user_input: str = input()
        assistant.abort()
        tts.abort()
        assistant.chat(user_input)

if __name__ == "__main__":
    main()
