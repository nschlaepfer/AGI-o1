import gradio as gr
import requests
import json

API_URL = "http://localhost:5000"  # Adjust this to your Flask server's address

def chat(message, history):
    response = requests.post(f"{API_URL}/chat", json={"message": message})
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return f"Error: {response.status_code} - {response.text}"

def edit_message(index, new_content):
    response = requests.post(f"{API_URL}/edit_message", json={"index": index, "new_content": new_content})
    if response.status_code == 200:
        return response.json()["message"]
    else:
        return f"Error: {response.status_code} - {response.text}"

def remove_message(index):
    response = requests.post(f"{API_URL}/remove_message", json={"index": index})
    if response.status_code == 200:
        return response.json()["message"]
    else:
        return f"Error: {response.status_code} - {response.text}"

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    edit_index = gr.Number(label="Message Index to Edit", precision=0)
    edit_content = gr.Textbox(label="New Content")
    edit_button = gr.Button("Edit Message")

    remove_index = gr.Number(label="Message Index to Remove", precision=0)
    remove_button = gr.Button("Remove Message")

    msg.submit(chat, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

    edit_button.click(
        edit_message,
        inputs=[edit_index, edit_content],
        outputs=gr.Textbox(label="Edit Result")
    )

    remove_button.click(
        remove_message,
        inputs=[remove_index],
        outputs=gr.Textbox(label="Remove Result")
    )

if __name__ == "__main__":
    demo.launch()