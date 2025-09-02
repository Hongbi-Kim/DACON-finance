import pandas as pd
import os

from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings

model_name = "nlpai-lab/KURE-v1"

hf_embeddings = HuggingFaceEndpointEmbeddings(
    model=model_name,
    task="feature-extraction",
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
)

client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_api_key"],
)

collections = client.get_collections()

vector_store = QdrantVectorStore(
    client = client,
    collection_name = "rag-finance",
    embedding = hf_embeddings
)

collection_info = client.get_collection("rag-finance")
print(f"포인터 개수: {collection_info.points_count}")
print(f"컬렉션 상태: {collection_info.status}")

test = pd.read_csv("/Users/hongbikim/Dev/dacon_finance/open/test.csv")


from pydantic import BaseModel, Field
# JSON 스키마 정의
class Answer(BaseModel):
    answer: int = Field(description="정답 번호")
    justification: str = Field(description="정답에 대한 근거")

class Answer_write(BaseModel):
    answer: str = Field(description="정답")
    justification: str = Field(description="정답에 대한 근거")
    
# Ollama 모델을 불러옵니다.
llm = ChatOllama(model="standard_lee/kanana-nano-2.1b-instruct:latest",
                 temperature=0.01,
                 num_predict = 256)
# llm = ChatOllama(model="qwen3:4b-instruct",
#                  temperature=0.01,
#                  num_predict = 256)

llm2 = ChatOllama(model="standard_lee/kanana-nano-2.1b-instruct:latest",
                 temperature=0.01,
                 num_predict = 1024)

prompt_number = """Your task is to choose correct answer the question based on the context below.
Context: {context}
Question: {query}

If the given context is not relevant to the question, do not refer to it and answer based on your knowledge.

Return your response in the following JSON format:
{{
    "answer": <정답_번호>,
    "evidence": "<정답에_대한_간단한_근거>"
}}

The correct answer number must be returned based on evidence.
Make sure to return only valid JSON without any additional text.
"""

prompt_write = """You are a financial expert.
Your task is to answer the question based on the context below.

Context: {context}
Question: {query}

Write based on the given context.
If the given context is not relevant to the question, do not refer to it and write based on your knowledge.

Be specific and write clearly without repeating.
Return the answer.
"""

answer_list = []
reference_list = []
for i in range(len(test)):
    print(f"=== {i+1} / {len(test)} ===")
    query = test['Question'][i]
    print(query)

    results = vector_store.similarity_search(
        query = query,
        k = 10,
    )
    reference_list.append(results)
    context = []
    for result in results:
        context.append(result.page_content)
    context = "\n".join(context)

    if "\n" in query:
        prompt_template = prompt_number
        json_parser = JsonOutputParser(pydantic_object=Answer)
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm | json_parser

    else:
        prompt_template = prompt_write
        json_parser = StrOutputParser()
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm2 | json_parser


    # 간결성을 위해 응답은 터미널에 출력됩니다.
    answer = chain.invoke({
            "context": context,
            "query": query
        })
    print(answer)
    answer_list.append(answer)

sub_answer = []
for e, aa in enumerate(answer_list):
    if type(aa) == str:
        sub_answer.append(aa)
        # print(aa)
    else:
        try:
            sub_answer.append(int(aa['answer']))
            # print(int(aa['answer']))
        except Exception as e:
            sub_answer.append(1)
            print(e)

submit = pd.read_csv("/Users/hongbikim/Dev/dacon_finance/open/sample_submission.csv")
submit['Answer'] = sub_answer

submit.to_csv("../open/submit_kanana_2b_0829.csv", index=False)