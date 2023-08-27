import os

# 当前目录
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# LLM_MODEL_PATH = r"D:\Project\ChatPdf\chatglm2-6b-int4"  
LLM_MODEL_PATH = "THUDM/chatglm2-6b-int4" # 对话模型

# EMBEDDING_MODEL_PATH = r'D:\Project\ChatPdf\text2vec-base-chinese' 
EMBEDDING_MODEL_PATH = 'shibing624/text2vec-base-multilingual'  # 检索模型文件 or huggingface远程仓库

PDF_FILE_PATH = r"D:\Project\Langchain-LLM-PdfReader\pdf_file"
KNOWLEDGE_FILE_PATH = r"D:\Project\Langchain-LLM-PdfReader\knowledge_base"
