from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import sys
import pandas as pd
import os
from ctransformers import AutoModelForCausalLM,AutoConfig
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # 데이터 파일 경로 설정 (파일 경로는 실제 환경에 맞게 수정해야 합니다)
# data_file_path = './pdfs/23.09.csv'
# # CSV 파일을 데이터 프레임으로 읽어오기
# df = pd.read_csv(data_file_path, encoding='latin1')

# # 데이터 프레임 확인
# print(df.head())  # 데이터 프레임의 처음 몇 행을 출력하여 데이터 구조를 확인합니다.

DB_FAISS_PATH = "vectorstore/db_faiss"
loader = CSVLoader(
                    file_path="./pdfs/2023.csv", 
                    encoding='utf-8', 
                    # csv_args={'delimiter': ',', 'fieldnames': ['task_result_id','task_id','process_type','activity_type','batch_date','work_date','work_time','delivery_grp_daily_id','delivery_grp_daily_name','bp_cd','bp_nm','sl_cd,sl_nm,item_cd','item_nm','item_unit','lot_no','plan_qty','complete_qty','worker']},
                    
                    )
# loader = CSVLoader(file_path="./pdfs/2023.csv", encoding='latin1', csv_args={'delimiter': ','})
# loader = CSVLoader(file_path="2019.csv", encoding="utf-8", csv_args={'delimiter': ','})
data = loader.load()
# print(data)

# Split the text into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 100,
    chunk_overlap  = 100,
    length_function = len,
    add_start_index = True,)

text_chunks = text_splitter.split_documents(data)

print(len(text_chunks))
print(text_chunks[111])
print(text_chunks[112])
print(text_chunks[113])
print(text_chunks[114])
# print(text_chunks[5])
# print(text_chunks[6])
# print(text_chunks[7])
# print(text_chunks[8])
# print(text_chunks[9])
# print(text_chunks[10])
# print(text_chunks[11])
# Download Sentence Transformers Embedding From Hugging Face
embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
# embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-mpnet-base-v2')
# embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

# COnverting the text Chunks into embeddings and saving the embeddings into FAISS Knowledge Base
docsearch = FAISS.from_documents(text_chunks, embeddings)

docsearch.save_local(DB_FAISS_PATH)


# query = "What is the mean of all task work time?"

#docs = docsearch.similarity_search(query, k=3)

#print("Result", docs)

# llm = CTransformers(model="/dli/task/llama.cpp/models/Llama-2-7B-GGUF/llama-2-7b.Q4_0.gguf",
#                     temperature=0.1,
#                     model_type="llama",
#                     max_tokens=512,
#                     top_p=1,
#                     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
#                     verbose=True, # Verbose is required to pass to the callback manager
#                     )

# config = AutoConfig.from_pretrained("TheBloke/Llama-2-7b-Chat-GGUF")
# # Explicitly set the max_seq_len
# config.max_seq_len = 2048
# config.max_answer_len= 2048
config = {'max_new_tokens': 1024, 'repetition_penalty': 1.1}
llm = CTransformers(model="/dli/task/llama.cpp/models/Llama-2-13B-chat-GGUF/llama-2-13b-chat.Q5_K_M.gguf",
                    # n_ctx =2048,
                    temperature=0.1,
                    model_type="llama",
                    max_tokens=1024,
                    n_gpu_layers = 1024,  # Change this value based on your model and your GPU VRAM pool.
                    n_batch = 1024,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
                    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                    verbose=True, # Verbose is required to pass to the callback manager
                    config=config,
                    )

# llm = CTransformers(model="/dli/task/llama.cpp/models/vicuna-13B-v1.5-16K-GGUF/vicuna-13b-v1.5-16k.Q2_K.gguf",
#                     temperature=0.1,
#                     model_type="llama",
#                     max_tokens=512,
#                     n_gpu_layers = 512,  # Change this value based on your model and your GPU VRAM pool.
#                     n_batch = 512,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
#                     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
#                     verbose=True, # Verbose is required to pass to the callback manager
#                     )

qa = ConversationalRetrievalChain.from_llm(llm, retriever=docsearch.as_retriever())

while True:
    chat_history = []
    #query = "What is the value of  GDP per capita of Finland provided in the data?"
    query = input(f"Input Prompt: ")
    if query == 'exit':
        print('Exiting')
        sys.exit()
    if query == '':
        continue
    result = qa({"question":query, "chat_history":chat_history})
    print("Response: ", result['answer'])
    
# while True:
#     #query = "What is the value of  GDP per capita of Finland provided in the data?"
#     query = input(f"Input Prompt: ")
#     if query == 'exit':
#         print('Exiting')
#         sys.exit()
#     if query == '':
#         continue
#     result = qa({"question":query})
#     print("Response: ", result['answer'])
