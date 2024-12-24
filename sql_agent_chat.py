import os
import streamlit as st

from langchain import hub
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from st_callable_util import get_streamlit_cb


### Page initialization ###
st.set_page_config(page_title="SQL Agent Chat", page_icon="🗁⌨")
st.title("🗁⌨ SQl DB Agent")


### GLOBAL VARIABLES ###
DATABASE_USER = 'serveradmin'
USER_PASSWORD = 'TestUnilever1!'
DATABASE_ENDPOINT = 'prdunileverdatabase.mysql.database.azure.com'
DATABASE_SCHEMA = 'transactional_data'

AZURE_DEPLOYMENT = 'gpt-4o-mini'
AZURE_OPENAI_API_VERSION = '2024-08-01-preview'
AZURE_OPENAI_API_KEY = 'DJSInhXbnuV08lkSHlCihDnQMG2VkZMCdTiCP2l3H4xcushxL88mJQQJ99ALACYeBjFXJ3w3AAABACOGZqoB'
AZURE_ENDPOINT = 'https://unilever-poc-openai.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-08-01-preview'

mysql_uri = "mysql+mysqlconnector://{}:{}@{}:3306/{}".format(DATABASE_USER, 
                                                             USER_PASSWORD, 
                                                             DATABASE_ENDPOINT, 
                                                             DATABASE_SCHEMA)


### CONFIGURATIONS ###
db = SQLDatabase.from_uri(mysql_uri)

llm = AzureChatOpenAI(azure_deployment=AZURE_DEPLOYMENT,  
                      api_version=AZURE_OPENAI_API_VERSION,   
                      api_key=AZURE_OPENAI_API_KEY,    
                      azure_endpoint=AZURE_ENDPOINT)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

system_message = (hub
                 .pull("langchain-ai/sql-agent-system-prompt")
                 .format(dialect=db.dialect, top_k=5))

agent_executor = create_react_agent(llm, 
                                    tools, 
                                    state_modifier=system_message)

intro_msg = "Hi! I'm your SQL Assistent. What would you like to know?"

### Chat Initialization ###

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [AIMessage(content=intro_msg)]

for msg in st.session_state.messages:
    if type(msg) == AIMessage:
        st.chat_message("assistant").write(msg.content)
    if type(msg) == HumanMessage:
        st.chat_message("user").write(msg.content)


if user_query := st.chat_input(placeholder="Ask me anything related to a SQL Database!"):
    st.session_state.messages.append(HumanMessage(content=user_query))
    st.chat_message("user").write(user_query)

    with st.chat_message("assitent"):
        msg_placeholder = st.empty()  # Placeholder for visually updating AI's response after events end
        # create a new placeholder for streaming messages and other events, and give it context
        st_callback = get_streamlit_cb(st.empty())
        response = agent_executor.invoke({"messages": [{"role": "user", "content": user_query}]}, config={"callbacks": [st_callback]})
        last_msg = response["messages"][-1].content
        st.session_state.messages.append(AIMessage(content=last_msg))  # Add that last message to the st_message_state
        msg_placeholder.write(last_msg) # visually refresh the complete response after the callback container

### bacth inference ###
#if user_query:
#    st.session_state.messages.append({"role": "user", "content": user_query})
#    st.chat_message("user").write(user_query)

    
    #with st.chat_message("assistant"):
    #    #st_cb = StreamlitCallbackHandler(st.container())
    #    output = agent_executor.invoke({"messages": [{"role": "user", "content": user_query}]}) #,callbacks=[st_cb]
    #    response = output['messages'][-1].content  
    #    st.session_state.messages.append({"role": "assistant", "content": response})
    #    st.write(response)