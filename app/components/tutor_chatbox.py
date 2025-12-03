"""
Tutor chatbox component for Streamlit dashboard.
Interactive chat interface with adaptive LLM tutor.
"""

import streamlit as st
import requests
from typing import Dict, Optional


class TutorChatbox:
    """Tutor chatbox component"""
    
    @staticmethod
    def display(api_url: str):
        """
        Display tutor chatbox interface.
        
        Args:
            api_url: FastAPI backend URL
        """
        st.subheader("💬 Chat with Tutor")
        
        # Initialize chat history in session state
        if "tutor_chat_history" not in st.session_state:
            st.session_state.tutor_chat_history = []
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.tutor_chat_history:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.write(message["content"])
                else:
                    with st.chat_message("assistant"):
                        st.write(message["content"])
        
        # Chat input
        user_input = st.chat_input("Type your message here...")
        
        if user_input:
            # Add user message to history
            st.session_state.tutor_chat_history.append({
                "role": "user",
                "content": user_input
            })
            
            # Get emotional context (if available)
            context = {}
            if st.session_state.get("emotion_history"):
                latest_emotions = st.session_state.emotion_history[-1].get("emotions", {})
                context["emotions"] = latest_emotions
            
            if st.session_state.get("engagement_history"):
                latest_engagement = st.session_state.engagement_history[-1]
                context["engagement"] = latest_engagement.get("engagement", 50.0)
                context["confusion"] = latest_engagement.get("confusion", 0.0)
            
            # Get tutor response from API
            try:
                response = requests.post(
                    f"{api_url}/tutor/chat",
                    params={"message": user_input},
                    json={"context": context},
                    timeout=10
                )
                
                if response.status_code == 200:
                    tutor_response = response.json().get("response", "I'm here to help!")
                else:
                    tutor_response = "Sorry, I encountered an error. Please try again."
            
            except Exception as e:
                tutor_response = f"Error connecting to tutor: {e}"
            
            # Add tutor response to history
            st.session_state.tutor_chat_history.append({
                "role": "assistant",
                "content": tutor_response
            })
            
            # Rerun to update chat display
            st.rerun()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.tutor_chat_history = []
            st.rerun()

