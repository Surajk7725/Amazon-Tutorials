import streamlit as st
import boto3
import json
import time
from typing import List


# Function to calculate metrics (accuracy, throughput, latency, input/output tokens, robustness)
def calculate_metrics(prompt: str, response_text: str, start_time: float, end_time: float):
    latency = end_time - start_time
    input_tokens = len(prompt.split())
    output_tokens = len(response_text.split())

    # Calculate throughput
    throughput = output_tokens / latency if latency > 0 else "N/A"

    return {
        "throughput": throughput,
        "latency": latency,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens
    }
st.set_page_config(
    page_title="Amazon Titan Text Premier V1",
    page_icon="🤖",
    layout="centered"
)
with st.sidebar:
    aws_access_key_id = st.text_input("AWS Access Key Id", placeholder="access key", type="password")
    aws_secret_access_key = st.text_input("AWS Secret Access Key", placeholder="secret", type="password")

boto_session = boto3.session.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key)

st.title("💬 Amazon ChatBot")
st.caption("🚀 A streamlit chatbot powered by AWS Bedrock Amazon Titan Text Premier-v1")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if customer_input := st.chat_input():

    if not aws_access_key_id and not aws_secret_access_key:
        st.info("Access Key Id or Secret Access Key are not provided yet!")
        st.stop()

    client = boto_session.client(
        service_name='bedrock-runtime',
        region_name="us-east-1"
    )
    st.session_state.messages.append({"role": "user", "content": customer_input})
    st.chat_message("user").write(customer_input)
    prompt = f"\n\nHuman:{customer_input}\n\nAssistant:"

    body = json.dumps({
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 1000,
            "temperature": 0.7,
            "topP": 0.9,
            "stopSequences": []
        }
    })

    start_time = time.time()
    response = client.invoke_model(
        body=body,
        modelId="amazon.titan-text-premier-v1:0",
        accept='application/json',
        contentType='application/json'
    )
    end_time = time.time()

    msg = json.loads(response['body'].read().decode('utf-8'))
    response_text = msg['results'][0]['outputText']

    metrics = calculate_metrics(prompt, response_text, start_time, end_time)

    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.chat_message("assistant").write(response_text)

    # Display metrics in the sidebar
    with st.sidebar:
        st.write("Evaluation Metrics")
        st.write(f"**Throughput**: {metrics['throughput']:.6f} tokens/second")
        st.write(f"**Latency**: {metrics['latency']:.6f} seconds")
        st.write(f"**Input Tokens**: {metrics['input_tokens']}")
        st.write(f"**Output Tokens**: {metrics['output_tokens']}")







