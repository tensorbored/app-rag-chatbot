# RAG Chatbot App with Memory

## Description
This Streamlit app is designed to chat with users based on uploaded pdf/url using RAG (Retrieval Augmented Generation) approach and refined query to keep track of context.

## Run App Online
Link: https://chat-with-docs-chatbot.streamlit.app/

## App Live Testing
![gif.gif](assets/app-rag-chatbot-gif.gif)

## Screenshots
![img.png](assets/app-rag-chatbot-screenshot.png)

- **Refined Query** In the 2nd Question *what was the update before that* we are using `create_history_aware_retriever` to create a new refined standalone version of user question which reference context in the chat history to fetch data from vector database.
- Hence using refined question, relevant embedding was retrieved from vector database and LLM was able to answer regarding the previous update version.  

## Set-up
1. To get started, first create an API_KEY from here: https://console.groq.com/keys. Then update the `GROQ_API_KEY` in the app text input with newly generated API_KEY. 

2. To get started, first install the dependencies using:
    ```commandline
     pip install -r requirements.txt
    ```
   
3. Run the streamlit app:
   ```commandline
   streamlit run main.py
   ```

**Contact**
For any questions or issues, please contact Prateek @ [s.prateek3080@gmail.com].
   
**Additional Terms:**
This software is licensed under the MIT License. However, commercial use of this software is strictly prohibited without prior written permission from the author. Attribution must be given in all copies or substantial portions of the software.