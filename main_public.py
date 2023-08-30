import json
import os
from typing import Optional

import openai
import tiktoken
import streamlit as st
from langchain import LLMChain, PromptTemplate
from langchain.chains import TransformChain

from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import ZapierNLARunAction
from langchain.utilities import ZapierNLAWrapper
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from pypdf import PdfReader
from streamlit.runtime.uploaded_file_manager import UploadedFile

os.environ["OPENAI_API_KEY"] = # YOUR OPENAI_KEY
os.environ["ZAPIER_NLA_API_KEY"] = # YOUR ZAPIER_NLA_KEY

RESUME_GROUP = 2
COMPARE_RESUME_GROUP = 3


@st.cache_data
def load_db_chain(pdfs: list[UploadedFile]) -> tuple[list[FAISS], list[str]]:
    """create database for uploaded_resumes"""
    # load the pdf documents
    docs = []
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            sub_text = page.extract_text()
            cleaned_sub_text = ""
            for i in range(len(sub_text)):
                if sub_text[i] in {'{', '}'}:
                    cleaned_sub_text += ' '
                else:
                    cleaned_sub_text += sub_text[i]
            text += cleaned_sub_text
            # text = text.replace('\n', ' ')
        docs.append(text)

    # text splitting
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        length_function=len
    )
    embedding = OpenAIEmbeddings()
    dbs, names = [], []
    for doc in docs:
        dbs.append(FAISS.from_texts(texts=splitter.split_text(doc), embedding=embedding))
        template = f"Name the candidate with the following piece of information. Dashes act as delimitors. \n" \
                   f"----------------------------------\n" \
                   f"{doc[:2000]}\n----------------------------------\n"
        prompt = PromptTemplate(template=template, input_variables=[])
        names.append(LLMChain(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", verbose=True), prompt=prompt).predict().strip())

    return dbs, names


def compare_resumes(llm_model: ChatOpenAI, databases: list[FAISS], names: list[str], criteria: Optional[str],
                    index: tuple[int, int]) -> list[str]:
    """chain-of-thoughts to guide the chatbot to compare resumes (qualifications of different candidates)
    and return response"""
    work_prompt = f"Compare the {index[1] - index[0]} candidates' work experience IN AROUND 50 WORDS. " \
                  f"Refer to each candidate with their names. " \
                  f"COMPARISON IN 50 WORDS:"
    work_experience = chatbot_predict(llm_model, databases, work_prompt, names, index)

    education_prompt = f"Compare the {index[1] - index[0]} candidates' education history IN AROUND 50 WORDS. " \
                       f"Refer to each candidate with their names. \n" \
                       f"COMPARISON IN 50 WORDS:"
    education_compare = chatbot_predict(llm_model, databases, education_prompt, names, index)

    conclusion_prompt = "WORK EXPERIENCE:\n {work} \n----------------------\n" \
                        "EDUCATION HISTORY:\n {education} \n----------------------\n" \
                        "USER'S QUESTION:\n"
    if criteria:
        conclusion_prompt += f"Draw a conclusion on the qualifications of these {index[1] - index[0]} candidates based on " \
                             f"the following criteria:\n {criteria}"
    else:
        conclusion_prompt += f"Draw a conclusion on your comparison, " \
                             f"including each of the expertise of the {index[1] - index[0]} candidates. \n"
    conclusion_prompt += "State which candidate is stronger if a fair comparison can be conducted. \n"
    if criteria:
        conclusion_prompt += "If none of the candidates really fulfils the criteria, say none of them are suitable for " \
                             "the position.\n\n"
    conclusion_prompt += "CONCLUSION IN 1-2 SENTENCES:"
    final_prompt = PromptTemplate(template=conclusion_prompt, input_variables=["work", "education"])
    conclusion_chain = LLMChain(llm=llm_model, prompt=final_prompt)
    conclusion = conclusion_chain.predict(work=work_experience, education=education_compare, verbose=True)
    st.session_state.memory.save_context({"input": conclusion_prompt}, {"output": conclusion})

    return [work_experience, education_compare, conclusion]


def chatbot_predict(llm_model: ChatOpenAI, databases: list[FAISS], query: str, candidate_names: list[str],
                    index: tuple[int, int]) -> str:
    """custom make prompt to make sure context across all resumes are retrieved separately"""
    # define conversational chain
    # prompt_template = "Chat history:\n {chat_history}\n-----------------------------------------------------"
    prompt_template = """Use the following pieces of context to answer the users question. Dashes are used as separators.
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
-----------------------------------------------------\n"""
    for i in range(index[0], index[1]):
        matching_texts = databases[i].similarity_search(query)
        prompt_template += f"This is a resume: {candidate_names[i]}\n"
        for matching_text in matching_texts:
            prompt_template += matching_text.page_content + '\n'
        prompt_template += "-----------------------------------------------------\n"
    prompt_template += """THIS IS THE USER'S QUESTION: \n {query}"""
    prompt_template += "-----------------------------------------------------\n"
    # st.write(st.session_state.memory.load_memory_variables({}))
    final_prompt = PromptTemplate(template=prompt_template, input_variables=["query"])
    chain = LLMChain(llm=llm_model, prompt=final_prompt, verbose=True)
    return chain.predict(query=query)


def llm_memory_creation() -> ChatOpenAI:
    """define llm and create memory"""
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(
            memory_key='chat_history',
            return_messages=True,
            k=5
        )
    # print("memory resetted")
    return llm


def generalize_responses(llm_model: ChatOpenAI, todo: str, responses: list[str]) -> str:
    """Generate a summary by combining the responses for diff sets of candidates"""
    prompt_template = f"These are responses generated for the same query but for different sets of resumes.\n" \
                      f"--------------------------\n"
    for i in range(len(responses)):
        prompt_template += f"Resumes set {i + 1}:\n{responses[i]}\n\n"
    prompt_template += "--------------------------\nTO-DO:\n{todo}"
    final_prompt = PromptTemplate(template=prompt_template, input_variables=["todo"])
    chain = LLMChain(llm=llm_model, prompt=final_prompt, verbose=True)
    return chain.predict(todo=todo)


def ask(llm_model: ChatOpenAI, databases: list[FAISS], inputs: str, cand_names: list[str]) -> str:
    """Call this to make a query on all resumes."""
    chatbot_responses, resume_count = [], 0
    while resume_count < len(cand_names):
        if len(cand_names) - resume_count == RESUME_GROUP + 1:
            chatbot_responses.append(chatbot_predict(llm_model, databases, inputs, cand_names, (resume_count, len(cand_names))))
            resume_count += RESUME_GROUP + 1
        else:
            upper_boun = min(len(cand_names), RESUME_GROUP + resume_count)
            chatbot_responses.append(
                chatbot_predict(llm_model, databases, inputs, cand_names, (resume_count, upper_boun)))
            resume_count += upper_boun - resume_count
    return generalize_responses(llm_model, "Combine the above responses without letting the user know "
                                           "there are multiple resume sets. When no useful "
                                           "information can be found in any of the resume sets, "
                                           "don't mention them. Your ultimate goal is to answer "
                                           f"this question: {inputs}", chatbot_responses)


def nla_gcal(inputs):
    """Create Google calendar event inviting the interviewees"""
    action = next((a for a in actions if a["description"].startswith("Google Calendar:")), None)
    temp = {"calendar_invite": ZapierNLARunAction(action_id=action["id"], zapier_description=action["description"],
                                                  params_schema=action["params"], verbose=True).run(inputs["instructions"])}
    return temp


def create_draft(inputs):
    """Create gmail draft inviting the interviewees"""
    action = next((a for a in actions if a["description"].startswith("Gmail:")), None)
    return {"gmail_data": ZapierNLARunAction(action_id=action["id"], zapier_description=action["description"],
                                             params_schema=action["params"]).run(inputs["instructions"])}


if __name__ == '__main__':
    # built front end of chatbot
    files = st.file_uploader("Upload a resume", type='pdf', accept_multiple_files=True)
    if files:
        model = llm_memory_creation()
        docs_vector_dbs, names = load_db_chain(files)

        usr_input = st.text_input("Your query on the resumes uploaded:")
        ask_but = st.button("Ask!")
        st.write("Click this button for comprehensive qualification anaylysis for multiple individuals: ")
        compare_criteria = st.text_input("State the criteria of comparison between candidates. "
                                         "You can leave it blank if you like. "
                                         "Please don't hit enter after inputting. Click the Compare resumes button.")
        compare_but = st.button("Compare resumes!!")
        email_recipients = st.text_input("Name the candidates you want to send interview invites to: "
                                         "(Please seperate two candidates with a comma) ")
        interview_date = st.text_input("Specify the date and time you want to hold an interview with these candidates")
        send_but = st.button("Send!")
        if compare_but:
            if len(names) > COMPARE_RESUME_GROUP + 1:
                res_lists, count = [], 0
                while count < len(names):
                    if len(names) - count == COMPARE_RESUME_GROUP + 1:
                        res_lists.append(compare_resumes(model, docs_vector_dbs, names, compare_criteria,
                                                        (count, count + COMPARE_RESUME_GROUP + 1)))
                        count += COMPARE_RESUME_GROUP + 1
                    else:
                        index_upper_boun = min(len(names), COMPARE_RESUME_GROUP + count)
                        res_lists.append(compare_resumes(model, docs_vector_dbs, names, compare_criteria,
                                                         (count, index_upper_boun)))
                        count += index_upper_boun - count
                res = generalize_responses(model, "Combine the following comparison conclusions and list the "
                                                  "stronger candidates without letting the user know there "
                                                  "are multiple resume sets",
                                           [res_list[2] for res_list in res_lists])
                work, edu = "", ""
                for res_list in res_lists:
                    work += res_list[0] + " "
                    edu += res_list[1] + " "
                st.write("Working Experience:")
                st.write(work)
                st.write("Education History:")
                st.write(edu)
                st.write("Conclusion drawn:")
                st.write(res)
            else:
                res_list = compare_resumes(model, docs_vector_dbs, names, compare_criteria, (0, len(names)))
                st.write("Working Experience:")
                st.write(res_list[0])
                st.write("Education History:")
                st.write(res_list[1])
                st.write("Conclusion drawn:")
                st.write(res_list[2])
        elif usr_input and ask_but:
            if len(names) > RESUME_GROUP + 1:
                res = ask(model, docs_vector_dbs, usr_input, names)
            else:
                res = chatbot_predict(model, docs_vector_dbs, usr_input, names, (0, len(names)))
            st.write(res)
        elif email_recipients and send_but and interview_date:
            recipients_list = [recipient.strip() for recipient in email_recipients.split(',')]
            email_addresses = {}
            for i in range(len(recipients_list)):
                found = False
                for j in range(len(names)):
                    if recipients_list[i] in names[j]:
                        email_addresses[recipients_list[i]] = chatbot_predict(model, docs_vector_dbs,
                                                                              f"Give me the email address of {recipients_list[i]}",
                                                                              names, (j, j + 1))
                        found = True
                        break
                if not found:
                    st.write(f"{recipients_list[i]}'s email address is not found")
            print(email_addresses)
            if email_addresses != {}:
                recipients_list = [re_name for re_name in email_addresses]
                actions = ZapierNLAWrapper().list()
                gcal_chain = TransformChain(input_variables=["instructions"], output_variables=["calendar_invite"],
                                            transform=nla_gcal, verbose=True)
                draft_chain = TransformChain(input_variables=["instructions"], output_variables=["gmail_data"],
                                             transform=create_draft, verbose=True)
                calendar_event_obj = json.loads(gcal_chain.run(f"Create a Google calendar event for an interview scheduled "
                                                    f"at {interview_date} inviting {str(recipients_list)}"
                                                    f"\n{email_addresses}"))
                st.write("The Google calendar event:")
                st.write(f'Title: {calendar_event_obj["summary"]}\n\nAttendees: {calendar_event_obj["attendee_emails"]}\n\n'
                         f'Date: {calendar_event_obj["end__date_pretty"]}\n\nStart time: {calendar_event_obj["start__time_pretty"]}\n'
                         f'\nEnd time: {calendar_event_obj["end__time_pretty"]}')
                email_template = f'Dear students,\n\nI am pleased to inform you that you have been selected ' \
                                 f'to join our interview. The date and time of our interview is {interview_date}.' \
                                 f'\n\nYours sincerely,\nPhantom65536\n'
                email_draft_instructions = f"Draft a complete formal email to {str(recipients_list)} " \
                                           f"inviting them to participate in an interview.\n" \
                                           f'{email_addresses}\n---------------------------------\n' \
                                           f'FOLLOW THIS TEMPLATE and insert line breaks if necessary.:' \
                                           f'\n{email_template}'
                email_draft_obj = draft_chain.run(email_draft_instructions)
                st.write("The email sent:")
                st.write(f"Recipients: {calendar_event_obj['attendee_emails']}\n\nBody: \n\n{email_template}")
