# -*- coding: utf-8 -*-
import base64
import json
import logging
import os
import sys
import time
from typing import List, Tuple

import requests
from fastapi import Body
from pydantic import BaseModel

import config
import fastapi
import uvicorn
import logging
import sys
import torch
from sse_starlette import EventSourceResponse, ServerSentEvent
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from server.create_knowledge_base import create_base
from transformers import AutoTokenizer, AutoModel
from starlette.middleware.cors import CORSMiddleware


# get-logger用于记录日志
def getLogger(name, file_name, use_formatter=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s    %(message)s')
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    if file_name:
        handler = logging.FileHandler(file_name, encoding='utf8')
        handler.setLevel(logging.INFO)
        if use_formatter:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
            handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


class ChatGLM():
    def __init__(self) -> None:
        logger.info("Start initialize model...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_PATH, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(config.LLM_MODEL_PATH, trust_remote_code=True).cuda()
        # 多显卡支持，使用下面两行代替上面一行，将num_gpus改为你实际的显卡数量
        # from utils import load_model_on_gpus
        # self.model = load_model_on_gpus("THUDM/chatglm2-6b", num_gpus=2)
        self.model.eval()
        logger.info("Model initialization finished.")

    def clear(self) -> None:
        if torch.cuda.is_available():
            with torch.cuda.device:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    def answer(self, query: str, history, prompt):
        print("answer:", query, history, prompt)
        response, history = self.model.chat(self.tokenizer, query, prompt, history=history, max_length=8192)
        history = [list(h) for h in history]
        return response, history

    def stream(self, query, history, page, prompt):
        if query is None or history is None:
            yield {"query": "", "response": "", "history": [], "finished": True}
        size = 0
        response = ""
        for response, history in self.model.eval().stream_chat(
                self.tokenizer,
                query=query,
                history=history,
                prompt=prompt,
                max_length=8192,
                top_p=0.9,
                temperature=0.9,
                past_key_values=None,
                return_past_key_values=False):
            this_response = response[size:]
            history = [list(h) for h in history]
            size = len(response)
            yield {"delta": this_response, "response": response, "finished": False}
        logger.info("Answer - {}".format(response))
        yield {"response": response, "query": query, "page": page, "delta": "[EOS]", "history": history,
               "finished": True}
        # yield {"query": query, "delta": "[EOS]", "response": response, "page": page, "finished": True}


logger = getLogger('ChatGLM', 'chatlog.log')
MAX_HISTORY = 3  # 最大历史记录数


def start_server():
    env = os.environ
    app = fastapi.FastAPI()
    bot = ChatGLM()

    # 配置 CORS 中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 允许所有来源，可以根据需求进行配置
        allow_credentials=True,
        allow_methods=["*"],  # 允许所有请求方法
        allow_headers=["*"],  # 允许所有请求头
    )

    class CreateKnowledgeBaseRequest(BaseModel):
        kb_name: str = fastapi.Form(..., description="知识库名称")
        file_name: str = fastapi.Form(..., description="文件名称")
        pdf_file: str = fastapi.Form(..., description="base64编码的pdf文件内容")

    @app.post("/create_knowledge_base", tags=["Knowledge Base Management"], summary="创建知识库")
    async def create_knowledgebase(
            kbRequest: CreateKnowledgeBaseRequest = Body(..., context_type="application/json",
                                                         description="创建知识库")):
        """
        创建知识库
        :param kbRequest:
        :param file_name:
        :param kb_name:
        :param pdf_file:
        :return:
        """
        kb_name = kbRequest.kb_name
        file_name = kbRequest.file_name
        pdf_file = kbRequest.pdf_file

        # 保存文件至pdf_file/conversation_id
        file_path = os.path.join(config.PDF_FILE_PATH, kb_name)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_path = os.path.join(file_path, f"{file_name}.pdf")
        # 写入base64文件
        with open(file_path, "wb") as f:
            f.write(base64.b64decode(pdf_file))

        # 创建知识库
        return create_base(file_path, kb_name)

    class SteamChatRequest(BaseModel):
        query: str = fastapi.Form(..., description="对话问题")
        history: List[Tuple[str, str]] = None

    @app.post("/Stream_chat", tags=["Chat"], summary="与llm模型对话")
    async def stream_chat(
            chatRequest: SteamChatRequest = Body(..., context_type="application/json", description="对话请求")):
        """
        与ChatGlm对话
        :param chatRequest:
        :return:
        """
        query = chatRequest.query
        history = json.dumps(chatRequest.history)
        history = json.loads(history)

        def decorate(generator):
            print("generator", generator)
            for item in generator:
                yield ServerSentEvent(json.dumps(item, ensure_ascii=False), event='delta')

        try:
            text = query
            # ori_history = history
            page = []
            prompt = "你是一个助手，负责用中文聊天和解决用户的问题。"
            logger.info("Query - {}".format(text))
            # if len(ori_history) > 0:
            #     logger.info("History - {}".format(ori_history))
            # history = ori_history[-MAX_HISTORY:]
            # history = [tuple(h) for h in history]
            # return EventSourceResponse(decorate(bot.stream(text, history)))

            return EventSourceResponse(decorate(bot.stream(query=query, history=history, page=page, prompt=prompt)))
        except Exception as e:
            logger.error(f"error: {e}")
            return EventSourceResponse(decorate(bot.stream(None, None, None, None)))

    class PdfReaderRequest(BaseModel):
        kb_name: str = fastapi.Form(..., description="知识库名称")
        query: str = fastapi.Form(..., description="对话问题")
        history: List[Tuple[str, str]] = None

    @app.post("/PdfReader", tags=["Chat"], summary="与llm模型对话PDF")
    async def chatpdf(
            PdfRequest: PdfReaderRequest = Body(..., context_type="application/json", description="对话请求")):
        """
        与ChatGlm对话PDF
        :param PdfRequest:
        :param kb_name:
        :param history: [[问题,回答],[问题,回答],......] 空为[]
        :param query: 对话问题
        :return:
        """
        kb_name = PdfRequest.kb_name
        query = PdfRequest.query
        history = json.dumps(PdfRequest.history)
        history = json.loads(history)
        print("history:", history)

        persist_directory = os.path.join(config.KNOWLEDGE_FILE_PATH, kb_name)
        print(persist_directory)

        # 联网搜索
        # try:
        #     from search_engine_parser.core.engines.bing import Search as BingSearch
        #     bsearch = BingSearch()
        #     search_args = (query, 1)
        #     results = await bsearch.async_search(*search_args)
        #     web_content = results["description"][:5]
        #     logger.info("Web_Search - {}".format(web_content))
        # except Exception as e:
        #     logger.error("Web_Search - {}".format(e))
        #     web_content = ""
        web_content = ""

        # 从目录加载向量
        logger.info("Start load vector database... %s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        embedding = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_PATH)
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        print(vectordb._collection.count())
        logger.info("Load database building finished. %s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        # docs = vectordb.similarity_search(query, k=3)
        docs = vectordb.similarity_search(query, k=5)

        page = list(set([docs.metadata['page'] for docs in docs]))
        page.sort()

        context = [docs.page_content for docs in docs]
        prompt = f"已知PDF内容：\n{context}\n根据已知信息回答问题：\n{query}\n网络检索内容：\n{web_content}"

        def decorate(generator):
            print("generator", generator)
            for item in generator:
                yield ServerSentEvent(json.dumps(item, ensure_ascii=False), event='delta')

        try:
            text = query
            query = f"内容为：{context}\n根据已知信息回答问题：{query}"
            # 使用Langchain处理PDF去除历史记录
            history = []
            # 给历史记录加上问题
            # new_message = [f"这是我提供的文章内容{context}", "收到"]
            # history.append(new_message)
            # history = json.dumps(history, ensure_ascii=False)
            ori_history = history
            logger.info("Query - {}".format(text))
            if len(ori_history) > 0:
                logger.info("History - {}".format(ori_history))
            history = ori_history[-MAX_HISTORY:]
            history = [tuple(h) for h in history]
            # return EventSourceResponse(decorate(bot.stream(text, history)))

            return EventSourceResponse(decorate(bot.stream(query, history, page=page, prompt=prompt)))
        except Exception as e:
            logger.error(f"error: {e}")
            return EventSourceResponse(decorate(bot.stream(None, None, None, None)))

    @app.get("/free_gc", tags=["GPU"], summary="释放GPU缓存")
    def free_gpu_cache():
        try:
            bot.clear()
            return {"success": True}
        except Exception as e:
            logger.error(f"error: {e}")
            return {"success": False}

    host = env.get("HOST") if env.get("HOST") is not None else "0.0.0.0"
    port = int(env.get("PORT")) if env.get("PORT") is not None else 9999
    # uvicorn.run(app=app, host=host, port=port, reload=True)
    uvicorn.run(app=app, host=host, port=port)


if __name__ == '__main__':
    start_server()
