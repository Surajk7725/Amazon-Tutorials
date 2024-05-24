import streamlit as st
import boto3
import json

with st.sidebar:
    aws_access_key_id = st.text_input("AWS Access Key Id", placeholder="access key", type="password")
    aws_secret_access_key = st.text_input("AWS Secret Access Key", placeholder="secret", type="password")

boto_session = boto3.session.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key)

st.title("💬 Chatbot")
st.caption("🚀 A streamlit chatbot powered by AWS Bedrock Claude LLM")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if customer_input := st.chat_input():

    if not aws_access_key_id and not aws_secret_access_key:
        st.info("Access Key Id or Secret Access Key are not provided yet !")
        st.stop()

    client = boto_session.client(
        service_name='bedrock-runtime',
        region_name="us-east-1"
    )
    st.session_state.messages.append({"role": "user", "content": customer_input})
    st.chat_message("user").write(customer_input)
    prompt = f"\n\nHuman:{customer_input}\n\nAssistant:"

    body = json.dumps({
        "prompt": prompt,
        "max_gen_len": 512,
        "temperature": 0.5,
        "top_p": 0.9
    })

    response = client.invoke_model(
        body=body,
        modelId="meta.llama2-70b-chat-v1",
        accept='application/json',
        contentType='application/json'
    )

    msg = json.loads(response['body'].read())

    # Extract and print the response text
    response_text = msg["generation"]
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.chat_message("assistant").write(response_text)
