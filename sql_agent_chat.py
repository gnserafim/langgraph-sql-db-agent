import os
import streamlit as st

from dotenv import load_dotenv
from langchain import hub
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

load_dotenv()

### Page initialization ###
st.set_page_config(page_title="SQL Agent Chat", page_icon="🗁⌨")
st.title("🗁⌨ SQl DB Agent")
st.write("---")
st.write("")

### Functions ###
def build_react_sql_agent(sql_database: SQLDatabase,
                          temperature: float=0.0) -> CompiledStateGraph:
    
    llm = AzureChatOpenAI(temperature=temperature)

    toolkit = SQLDatabaseToolkit(db=sql_database, llm=llm)
    tools = toolkit.get_tools()
    dialect = toolkit.dialect

    system_message = (hub
                     .pull("langchain-ai/sql-agent-system-prompt")
                     .format(dialect=dialect, top_k=5)
                     )

    return create_react_agent(llm, 
                              tools, 
                              state_modifier=system_message,
                              checkpointer=MemorySaver())

def run_agent(user_query: str,
              agent: CompiledStateGraph, 
              stream_mode: str) -> dict:

    config = {"configurable": {"thread_id": "only_one_thread"}}

    return agent.invoke({"messages": [{"role": "user",
                                       'content': user_query}]},
                        stream_mode=stream_mode,
                        config=config)


@st.cache_resource(show_spinner = False)
def initial_loading(database_uri):
    with st.spinner("Loading Database and initializing agent..."):
        db = SQLDatabase.from_uri(database_uri)
        agent = build_react_sql_agent(db)

    return agent

### Objects initialization ###
DATABASE_USER = os.getenv("DATABASE_USER")
USER_PASSWORD = os.getenv("USER_PASSWORD")
DATABASE_ENDPOINT = os.getenv("DATABASE_ENDPOINT")
DATABASE_SCHEMA = os.getenv("DATABASE_SCHEMA")

mysql_uri = "mysql+mysqlconnector://{}:{}@{}:3306/{}".format(DATABASE_USER,  
                                                             USER_PASSWORD,  
                                                             DATABASE_ENDPOINT, 
                                                             DATABASE_SCHEMA)   


agent = initial_loading(mysql_uri)
intro_msg = "Hi! I'm your SQL Assistent. What would you like to know?"

### Chat Initialization ###
if "messages" not in st.session_state:
    st.session_state["messages"] = [AIMessage(content=intro_msg)]

for msg in st.session_state.messages:
    if type(msg) == AIMessage:
        st.chat_message("assistant").write(msg.content)
    if type(msg) == HumanMessage:
        st.chat_message("user").write(msg.content)

### bacth inference ###
if user_query := st.chat_input(placeholder="Ask me anything related to a SQL Database!"):
   st.session_state.messages.append(HumanMessage(content=user_query))
   st.chat_message("user").write(user_query)
   
   with st.chat_message("assistant"):
        with st.spinner("Search for information..."):
            output = run_agent(user_query, agent, 'values')
            response = output['messages'][-1].content  
            st.write(response)
            st.session_state.messages.append(AIMessage(content=response))