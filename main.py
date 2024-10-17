## RAG Q&A Conversation With PDF Including Chat History
# https://python.langchain.com/v0.2/docs/tutorials/qa_chat_history/

import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from src.chains import create_vector_embedding, load_document, load_url
from src.chains import clear_session_state_documents_vectors, create_vector_db, trimmer
from src.prompts import contextualize_q_prompt, qa_prompt

st.markdown(
    """
    <style>
        .st-emotion-cache-janbn0 {
            flex-direction: row-reverse;
            text-align: right;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

def main():

    ## set up Streamlit 
    st.title("RAG Chatbot with chat history")
    st.write("Upload pdf/url and chat with their content - Groq and Llama3")

    with st.sidebar:
        st.header('1. Input Parameters')
        ## Input the Groq API Key
        api_key=st.text_input("Enter your Groq API key:",type="password",placeholder="gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxx")

   
    # try:
    if True:
        ## Check if groq api key is provided
        if api_key:
            a=0
            llm=ChatGroq(groq_api_key=api_key,model_name="Llama3-8b-8192")

            ## chat interface
            with st.sidebar:
                session_id=st.text_input("Session ID",value="default_session")

            ## statefully manage chat history
            if 'store' not in st.session_state:
                st.session_state.store={}

            with st.sidebar:
                st.subheader("2. Upload PDF or enter URL")
                upload_type = st.radio("Select input type", ("PDF", "URL"))

                if upload_type == "PDF":
                    # clear_session_state_documents_vectors()
            
                    uploaded_files=st.file_uploader("Choose single/multiple .pdf files",type="pdf",accept_multiple_files=True)
                    # if uploaded_files:
                    if st.button("Fetch content"):
                        # with st.spinner("Loading URL content..."):                
                        clear_session_state_documents_vectors(st)
                        st.session_state.documents=load_document(uploaded_files=uploaded_files)
                        create_vector_db(st)
                        a=1

                elif upload_type == "URL":
                    # clear_documents_vectors_session_state()
                    upload_url = st.text_input("Enter URL", "")
                    if st.button("Fetch content"):
                        # with st.spinner("Loading URL content..."):
                        clear_session_state_documents_vectors(st)
                        st.session_state.documents=load_url(upload_url)
                        create_vector_db(st)
                        a=1

            if "vectors" in st.session_state:

                retriever = st.session_state.vectors.as_retriever() 

                history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

                #  question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
                question_answer_chain = qa_prompt | llm | StrOutputParser()

                rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

                def get_session_history(session:str)->BaseChatMessageHistory:
                    if session_id not in st.session_state.store:
                        st.session_state.store[session_id]=ChatMessageHistory()
                    return st.session_state.store[session_id]

                conversational_rag_chain=RunnableWithMessageHistory(
                    rag_chain,get_session_history,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer"
                    )

                session_history=get_session_history(session_id)
                session_history.messages=trimmer.invoke(session_history.messages)

                if len(session_history.messages) == 0:
                    session_history.add_ai_message("How can I help you?")
                    print("No msg")
                for msg in session_history.messages:
                    st.chat_message(msg.type).write(msg.content)

                if a==1:
                    m="New data loaded. You can now ask questions regarding it."
                    st.chat_message("assistant").info(m)
                    session_history.add_ai_message(m)                    
                    a=0

                user_input = st.chat_input("Your question:")
                if user_input:
                    # session_history=get_session_history(session_id)
                    response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={
                    "configurable": {"session_id":session_id}
                    }, # constructs a key "abc123" in `store`.
                    )

                    st.chat_message("human").write(user_input)

                    # Display assistant response in chat message container
                    with st.chat_message("assistant"):
                        st.markdown(response['answer'])                    

        else:
            with st.sidebar:
                st.warning("Please enter the Groq API Key")
    
    # except Exception as e:
    #     st.error(f"An Error Occurred: {e}")
    #     print("------error--------------")
    #     print(e)

if __name__ == "__main__":
    main()