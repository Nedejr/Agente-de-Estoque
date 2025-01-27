import os
import streamlit as st
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv('OPEN_AI_KEY')


st.set_page_config(
    page_title='Estoque GPT',
    page_icon='🖥️',

)
st.header('Assistente de Estoque')

model_options = ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'gpt-4o-mini', 'gpt-4o']

selected_model = st.sidebar.selectbox(
    label='Selecione o modelo LLM',
    options=model_options
)

st.sidebar.markdown('### Sobre')
st.sidebar.markdown('Este agente consulta um banco de dados de estoque utilizando GPT')

st.write('Faça perguntas sobre o estoque de produtos, preços e reposições.')
user_question = st.text_input('O que deseja saber sobre o estoque ?')


model = ChatOpenAI(model=selected_model, api_key=api_key )
db = SQLDatabase.from_uri('sqlite:///estoque.db')
toolkit = SQLDatabaseToolkit(
    db=db, 
    llm=model
)

prompt = '''
Use as ferramentas necessárias para responder pperguntas relacionadas ao estoque de produtos. Você fornecerá insights sobre os produtos, preços, reposição
de estoque e relatórios conforme solicitado pelo usuário. A resposta final deve ter uma formatação amigável de visualização para o usuário.
Sempre responde em português brasileiro.
Pergunta: {q}
'''

prompt_template = PromptTemplate.from_template(prompt)

system_message = hub.pull('hwchase17/react')

agent = create_react_agent(
    llm = model,
    tools=toolkit.get_tools(),
    prompt=system_message
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=toolkit.get_tools(),
    verbose=True,
    handle_parsing_errors=True

)


if st.button('Consultar'):
    if user_question:
        with st.spinner('Consultando o banco de dados:'):
            formatted_prompt = prompt_template.format(q=user_question)
            output = agent_executor.invoke(
                {'input': formatted_prompt}
            )
            st.markdown(output.get('output'))
    else:
        st.warning('Por favor insira sua pergunta.')
