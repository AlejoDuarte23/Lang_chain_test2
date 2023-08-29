
import chainlit as cl
import os
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI


os.environ["OPENAI_API_KEY"] = ""

template = """You are a nice chatbot having a conversation with a human, asnwers the question step by step.

Previous conversation:
{chat_history}

New human question: {question}
Response:"""

@cl.on_chat_start
def main():
    prompt = PromptTemplate.from_template(template)

    memory = ConversationBufferMemory(memory_key="chat_history")

    llm_chain = LLMChain(prompt=prompt, llm=OpenAI(model_name="gpt-4", temperature=0,verbose=True),memory=memory,verbose=True)
    cl.user_session.set("llm_chain", llm_chain)



@cl.on_message
async def main(message: str):
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain
    res = await cl.make_async(llm_chain)(
        message, callbacks=[cl.LangchainCallbackHandler()]
    )

    await cl.Message(content=res["text"]).send()
    return llm_chain