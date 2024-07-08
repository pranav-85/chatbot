import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage     
from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()

api_key = "api-key"

if api_key is None:
    st.error("API key is not set.")
else:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.set_page_config(page_title="Chat Bot", page_icon="🤖")

    st.title("Chat Bot")

    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

    def format_history(chat_history):
        formatted_history = []
        for msg in chat_history:
            if isinstance(msg, HumanMessage):
                formatted_history.append({"role": "user", "parts": [{"text": msg.content}]})
            elif isinstance(msg, AIMessage):
                formatted_history.append({"role": "model", "parts": [{"text": msg.content}]})
        return formatted_history

    def get_response(query, chat_history):
        formatted_history = format_history(chat_history)

        # Create a chat session with the current chat history
        chat = model.start_chat(history=formatted_history)

        # Send the user's query and get the response
        result = chat.send_message(query)

        # Add the human query and AI response to the chat history
        chat_history.append(HumanMessage(query))
        chat_history.append(AIMessage(result.text))

        return result.text, chat_history

    # Display the chat history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)

    user_query = st.chat_input("Message Bot")

    if user_query:
        response, updated_history = get_response(user_query, st.session_state.chat_history)

        # Display the new user query and AI response
        with st.chat_message("Human"):
            st.markdown(user_query)
        with st.chat_message("AI"):
            st.markdown(response)

        st.session_state.chat_history = updated_history
