import streamlit as st
import pinecone
from openai import OpenAI
import os

# Initialize OpenAI and Pinecone
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
pc = pinecone.Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index = pc.Index("gujtaxlaw")

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def get_relevant_context(query, k=3):
    query_embedding = get_embedding(query)
    results = index.query(vector=query_embedding, top_k=k, include_metadata=True)
    contexts = [match.metadata.get('text', '') for match in results.matches]
    return " ".join(contexts)

def get_chatgpt_response(query, context):
    system_prompt = """You are an expert tax consultant specializing in Gujarat tax laws and regulations.
    Your responses must ALWAYS be in Gujarati language, regardless of the input language.

    Follow these guidelines for your responses:
    1. Always structure your response in clear sections using Gujarati headings
    2. Provide detailed explanations with relevant tax provisions and rules
    3. Include practical examples where applicable
    4. If specific numbers or calculations are involved, show them clearly
    5. End with any important cautionary notes or deadlines if relevant
    6. If you're not completely sure about something, clearly state that in Gujarati

    Even if the user asks in English, your response should be detailed and well-structured in Gujarati only.
    Use the following context to provide accurate information: {context}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
    ]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7,
        max_tokens=1000
    )
    return response.choices[0].message.content

# Streamlit UI remains the same
st.title("ગુજરાતી ટેક્સ સહાયક | Gujarati Tax Assistant")
st.write("Ask your tax-related questions in English or Gujarati (ગુજરાતી અથવા અંગ્રેજીમાં તમારા ટેક્સ સંબંધિત પ્રશ્નો પૂછો)")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Enter your question here... | અહીં તમારો પ્રશ્ન લખો..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get relevant context and generate response
    context = get_relevant_context(prompt)
    response = get_chatgpt_response(prompt, context)

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with instructions
with st.sidebar:
    st.markdown("""
    ### Instructions | સૂચનાઓ

    #### For English Users:
    1. You can ask questions in English
    2. Responses will always be in Gujarati
    3. Ask clear and specific questions
    4. The response will include detailed explanations

    #### ગુજરાતી વપરાશકર્તાઓ માટે:
    1. તમે ગુજરાતીમાં પ્રશ્નો પૂછી શકો છો
    2. જવાબો હંમેશા ગુજરાતીમાં આપવામાં આવશે
    3. સ્પષ્ટ અને ચોક્કસ પ્રશ્નો પૂછો
    4. જવાબમાં વિગતવાર સમજૂતી આપવામાં આવશે
    """)

    # Add a note about response format
    st.markdown("""
    ### Response Format | જવાબ ફોર્મેટ

    Responses will include | જવાબોમાં સમાવેશ થશે:
    - Detailed explanation | વિગતવાર સમજૂતી
    - Relevant rules | સંબંધિત નિયમો
    - Examples | ઉદાહરણો
    - Important notes | મહત્વપૂર્ણ નોંધ
    """)
