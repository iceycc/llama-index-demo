from llama_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, \
    ServiceContext
from langchain import OpenAI
import gradio as gr
import sys
import os

os.environ["OPENAI_API_KEY"] = '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'
data_directory_path = './data'
index_cache_path = 'cache.json'


# 构建索引
def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 2000
    max_chunk_overlap = 20
    chunk_size_limit = 500

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=num_outputs))
    # 按最大token数500来把原文档切分为多个小的chunk
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=chunk_size_limit)
    # 读取directory_path文件夹下的文档
    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
    # 保存索引
    index.save_to_disk(index_cache_path)
    return index


def chatbot(input_text):
    # 加载索引
    index = GPTSimpleVectorIndex.load_from_disk(index_cache_path)
    response = index.query(input_text, response_mode="compact")
    return response.response


if __name__ == "__main__":
    # 使用gradio创建可交互ui
    iface = gr.Interface(fn=chatbot,
                         inputs=gr.inputs.Textbox(lines=7, label="Enter your text"),
                         outputs="text",
                         title="Text AI Chatbot")
    index = construct_index(data_directory_path)
    iface.launch(share=True)