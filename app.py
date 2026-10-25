
import streamlit as st
from main import answer_customer

# Streamlit page configuration
st.set_page_config(
    page_title="AI Customer Support",
    page_icon="🤖",
    layout="centered"
)

# Page header
st.title("🤖 AI Customer Support Agent")
st.caption("Powered by CrewAI + RAG + ChromaDB")

st.write(
    "Welcome to ABC Electronics Customer Support! "
    "Ask questions about returns, refunds, shipping, "
    "and warranties."
)

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! How can I help you today?"
            )
        }
    ]

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Customer input
question = st.chat_input(
    "Ask your customer support question..."
)

if question:
    # Display customer message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Execute CrewAI agent
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                response = answer_customer(question)
            except Exception:
                response = (
                    "Sorry, I couldn't process your "
                    "request right now. Please try again "
                    "or contact customer support."
                )

        st.markdown(response)

    # Save agent response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

# Sidebar
with st.sidebar:
    st.header("Customer Support")

    st.write("Available topics:")
    st.write("• Return policy")
    st.write("• Refund processing")
    st.write("• Shipping information")
    st.write("• Product warranty")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
