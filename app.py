import streamlit as st
import json, os
import spacy
#Import langchain dependencies
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

os.environ['OPENAI_API_KEY'] = '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'

# Load JSON Dataset
with open('./custom_dataset_2.json', 'r') as f:
    patient_data = json.load(f)

# Extract Conclusion from Data
conclusion = patient_data.get("Conclusion", "Other")
# Extract Patient Info
patient_info = patient_data.get("PatientInfo", {})

# List of Questions
questions = list(patient_info.keys())

# Function to get the corresponding answer
def get_matched_answer(selected_question):
    for question, answer in patient_info.items():
        if selected_question.lower() in question.lower():
            return answer
    return None

# Prompt template
question_template = PromptTemplate(
    input_variables=['selected_question', 'user_answer', 'matched_answer', 'conclusion'],
    template='You are an intelligent chatbot. Find whether the user answer "{user_answer}" is similar to the matched answer "{matched_answer}" for the question "{selected_question}". If you find it to be similar, then write the conclusion: "{conclusion}". Else write I do not know now.'
)

# Llms
llm = OpenAI(temperature=0.1)
question_chain = LLMChain(llm=llm, prompt=question_template, verbose=True, output_key='answer')

# Streamlit UI
st.title('🤖 Prompting a PEFT LLM')

# Assistant UI
with st.chat_message('assistant'):
    st.write('Please select from the below list of questions to answer!')
    selected_question = st.selectbox("Select a question to answer:", questions)

# User UI
with st.chat_message('user'):
    st.write("Answer the selected question:")
    user_answer = st.text_input("Your answer:", "")
    matched_answer = get_matched_answer(selected_question)
    if user_answer and matched_answer:
        # Verify the input variables passed to the prompt template
        st.write("Input variables for prompt template:", {
            'selected_question': selected_question,
            'user_answer': user_answer,
            'matched_answer': matched_answer,
            'conclusion': conclusion
        })
        # Use the prompt template to generate the prompt
        prompt = question_template.format( user_answer=user_answer,
                                           patient_data=patient_data,
                                           matched_answer=matched_answer,
                                           selected_question=selected_question,
                                           conclusion=conclusion
                                           )
        st.write("Generated prompt:", prompt)

        # Use the prompt to interact with the LLM and get the response
        # Invoke the LangChain with explicit input keys
        response = question_chain.run({
            'selected_question': selected_question,
            'user_answer': user_answer,
            'matched_answer': matched_answer,
            'conclusion': conclusion,
            'prompt': prompt
        })

        # Display the response
        st.write("Response:", response)