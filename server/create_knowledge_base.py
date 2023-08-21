import os
import time

from langchain.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import config


def create_base(file_path, kb_name):
    """
    创建PDF文件向量库
    :param kb_name:
    :param file_path: 文件路径
    :return:
    """
    try:
        print(f'file: {file_path}')
        print("Start building vector database... %s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        # loader = UnstructuredFileLoader(file_path, model="element")
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        # print(f'docs: {docs}')

        # 文本分割
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
        docs = text_splitter.split_documents(docs)

        # 向量化
        embedding = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_PATH)
        # 构造向量库+conversation_id
        persist_directory = os.path.join(config.KNOWLEDGE_FILE_PATH, kb_name)

        # 创建向量数据库
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embedding,
            persist_directory=persist_directory
        )
        print("vectordb:", vectordb._collection.count())
        vectordb.persist()
        print("Vector database building finished. %s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        return {"status": 200, "message": "success"}

    except Exception as e:
        return {"status": 500, "message": str(e)}
