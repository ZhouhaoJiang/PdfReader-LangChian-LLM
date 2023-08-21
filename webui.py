import base64
import json
import os
import random
import time
import requests
import config
import gradio as gr


# 重置会话
def clear():
    return '', None


def create_base(kb_name, filname, pdf_file):
    """
    创建知识库
    :param kb_name:
    :param filname:
    :param pdf_file:
    :return:
    """
    # 二进制转base64
    pdf_file = base64.b64encode(pdf_file)
    pdf_file = pdf_file.decode("utf-8")

    params = {
        "kb_name": kb_name,
        "file_name": filname,
        "pdf_file": pdf_file,
    }
    gr.Info("知识库创建中...")
    response = requests.post("http://127.0.0.1:9999/create_knowledge_base", json=params)
    if response.status_code == 200:
        print("Connection established. Receiving data...")
        gr.Info("知识库创建成功")
        return kb_name
    else:
        print("Failed to connect. Status code:", response.status_code)
        gr.Error("知识库创建失败")
        return None


# 请求会话api
def request_chatglm(kb_name, query, chat_history, chat_type):
    """
    请求会话api
    :param chat_type:
    :param kb_name:
    :param query:
    :param chat_history:
    :return:
    """
    print(query)
    if chat_type == "知识库对话":
        params = {
            "kb_name": kb_name,
            "query": query,
        }
        print("params:", params)
        response = requests.post("http://127.0.0.1:9999/PdfReader", json=params, stream=True)
        if response.status_code == 200:
            print("Connection established. Receiving data...")
            chat_history = [[query, ""]]
            # chat_history.append([query, ""])
            for line in response.iter_lines(decode_unicode=True):
                try:
                    data_dict = json.loads(line[6:])
                    chat_response = data_dict["response"]
                    print("chat_response:", chat_response)
                    print("chat_history:", chat_history)
                    chat_history[-1][1] = chat_response
                    yield chat_history
                except Exception as e:
                    print("Exception in handle_sse_response:", e)
                    # yield None
        else:
            print("Failed to connect. Status code:", response.status_code)
            gr.Error("Failed to connect")
    else:
        params = {
            "query": query,
        }
        response = requests.post("http://127.0.0.1:9999/Stream_chat", json=params, stream=True)
        if response.status_code == 200:
            print("Connection established. Receiving data...")
            print("chat_history:", chat_history)
            chat_history.append([query, ""])
            for line in response.iter_lines(decode_unicode=True):
                try:
                    data_dict = json.loads(line[6:])
                    chat_response = data_dict["response"]
                    # print("chat_response:", chat_response)
                    # print("chat_history:", chat_history)
                    chat_history[-1][1] = chat_response
                    yield chat_history
                except Exception as e:
                    print("Exception in handle_sse_response:", e)
                    # yield None
        else:
            print("Failed to connect. Status code:", response.status_code)
            gr.Error("Failed to connect. Status code:", response.status_code)


with gr.Blocks(title="PdfReader") as webui:
    gr.Markdown(
        """
            <center><h1>PdfReader With LangChain</hq><center>
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            embedding_model = gr.Dropdown(label="Embedding Model", choices=["text2vec"], value="text2vec")
            llm_model = gr.Dropdown(label="LLM Model", choices=["chatglm2-6b-int4"], value="chatglm2-6b-int4")

            chat_type = gr.Radio(label="对话方式", choices=["知识库对话", "模型对话"], value="模型对话")

            # kb_name = gr.Radio(choices=[name for name in os.listdir(config.KNOWLEDGE_FILE_PATH) if
            #                             os.path.isdir(os.path.join(config.KNOWLEDGE_FILE_PATH, name))],
            #                    label="知识库", value="无", live=True)
            # kb_submit = gr.Button(value="加载知识库", variant="primary")

            pdf_file = gr.File(label="PDF File", file_types=["pdf"], type="binary")
            writer_kb_name = gr.Textbox(label="填写知识库名称")
            create_button = gr.Button(value="构造知识库", variant="primary")

        with gr.Column(scale=3):
            chatbot = gr.Chatbot().style(height=500)
            query = gr.Textbox()
            with gr.Row():
                submit = gr.Button(value="发送", variant="primary")
                clear_btn = gr.Button(value="清空", variant="secondary")

    create_button.click(create_base, [writer_kb_name, writer_kb_name, pdf_file], outputs=[writer_kb_name])
    submit.click(request_chatglm, inputs=[writer_kb_name, query, chatbot, chat_type], outputs=[chatbot])
    clear_btn.click(clear, outputs=[query, chatbot])
    # submit.click(test, inputs=[kb_name, query, chatbot], outputs=[chatbot])

if __name__ == '__main__':
    webui.launch(inline=False, share=True, debug=True, server_name="0.0.0.0", server_port=7788,
                 enable_queue=True, show_error=True)
