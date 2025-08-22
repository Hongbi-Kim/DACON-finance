from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Ollama 모델을 불러옵니다.
llm = ChatOllama(model="exaone3.5:latest")

# 프롬프트
prompt = ChatPromptTemplate.from_template("{topic} 에 대하여 간략히 설명해 줘.")

# 체인 생성
chain = prompt | llm | StrOutputParser()

# 간결성을 위해 응답은 터미널에 출력됩니다.
answer = chain.invoke({"topic": "간편송금"})

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("kakaocorp/kanana-1.5-8b-instruct-2505")
model = AutoModelForCausalLM.from_pretrained("kakaocorp/kanana-1.5-8b-instruct-2505")

prompt = f"""Your task is to answer the user's question based on the provided context.

Here is the 

"""

messages = [
    {"role": "user", "content": query},
    {"role": "assistant", "content": prompt},
]

inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))