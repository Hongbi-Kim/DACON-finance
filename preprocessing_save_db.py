import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain.schema import Document
import uuid
import re
import pandas as pd

from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from dotenv import load_dotenv
load_dotenv()

files = os.listdir("/Users/hongbikim/Dev/dacon_finance/docs")

##############################################################################
chunk_size=512
model_name = "nlpai-lab/KURE-v1"
##############################################################################

# Qdrant 클라이언트 설정
hf_embeddings = HuggingFaceEndpointEmbeddings(
    model=model_name,
    task="feature-extraction",
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
)


client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_api_key"],
)


def reset_collection(collection_name="rag-finance"):
    """컬렉션 초기화"""
    try:
        # 기존 컬렉션 삭제
        client.delete_collection(collection_name)
        print(f"✅ 컬렉션 '{collection_name}' 삭제 완료")
    except Exception as e:
        print(f"⚠️  컬렉션 삭제 중 오류: {e}")
    
    # 임베딩 차원 확인
    sample_embedding = hf_embeddings.embed_query("test")
    embedding_dim = len(sample_embedding)
    
    # 새 컬렉션 생성
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=embedding_dim,
            distance=Distance.COSINE
        )
    )
    print(f"✅ 새 컬렉션 '{collection_name}' 생성 완료")
    
    # 새 벡터스토어 반환
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=hf_embeddings
    )

vector_store = reset_collection()

collection_info = client.get_collection("rag-finance")
print(f"포인터 개수: {collection_info.points_count}")
print(f"컬렉션 상태: {collection_info.status}")

# 텍스트 전처리 함수
def clean_text(text):
    text = text.replace("&lt", "").replace("&amp", "&").replace("&quot", '"').replace("&apos", "'").replace("&nbsp", " ").replace("&gt;", ">").replace("&gt", ">").replace("&lt;", "<").replace("&lt", "<").replace("&#39;", "'").replace("&#39", "'").replace("&#34;", '"').replace("&#34", '"').replace("&gt;", ">").replace("&gt", ">").replace("&lt;", "<").replace("&lt", "<").replace("&nbsp;", " ").replace("&nbsp", " ").replace("&amp;", "&").replace("&amp", "&").replace("&quot;", '"').replace("&quot", '"').replace("&apos;", "'").replace("&apos", "'").replace("&#39;", "'").replace("&#39", "'").replace("&#34;", '"').replace("&#34", '"')
    text = re.sub(r"\s{2,}", " ", text)
    # text = re.sub(r".{2,}", ".", text)
    new_text = []
    for t in text.split("\n"):
        if "삭제" not in t:
            new_text.append(t)
    new_text = "\n".join(new_text)
    PAIR_PATTERN = re.compile(r"<[^<>]*>|\[[^\[\]]*\]")
    PAIR_PATTERN.sub("", new_text)
    return PAIR_PATTERN.sub("", new_text)

    
# 법률문서 구분자 (우선순위 순)
legal_separators = [
    "\n제\d+조",      # 조문
    "\n제\d+장",      # 장
    "\n제\d+절",      # 절
    "\n①",           # 항
    "\n②", "\n③", "\n④", "\n⑤", "\n⑥", "\n⑦", "\n⑧", "\n⑨", "\n⑩",
    "\n1.", "\n2.", "\n3.", "\n4.", "\n5.",  # 숫자 항목
    "\n가.", "\n나.", "\n다.", "\n라.", "\n마.",  # 한글 항목
    "\n\n",          # 단락
    "\n",            # 줄바꿈
    " ",             # 공백
]

base_splitter = RecursiveCharacterTextSplitter(
    separators=legal_separators,
    chunk_size=chunk_size,
    length_function=len,
)

def extract_legal_structure(text):
    """텍스트에서 법률 구조 정보 추출"""
    structure = {}
    
    # 조문 찾기
    article_match = re.search(r'제(\d+)조', text)
    if article_match:
        structure['article'] = f"제{article_match.group(1)}조"
    
    # 장 찾기
    chapter_match = re.search(r'제(\d+)장', text)
    if chapter_match:
        structure['chapter'] = f"제{chapter_match.group(1)}장"
    
    # 절 찾기
    section_match = re.search(r'제(\d+)절', text)
    if section_match:
        structure['section'] = f"제{section_match.group(1)}절"
    
    # 항 찾기
    paragraph_matches = re.findall(r'[①②③④⑤⑥⑦⑧⑨⑩]', text)
    if paragraph_matches:
        structure['paragraph'] = paragraph_matches
    
    return structure

def _merge_short_chunks(splits, page_num, min_length=200):
    """500글자 이하의 짧은 청크들을 합치기"""
    if not splits:
        return []
    
    merged_splits = []
    current_chunk = None
    
    for split in splits:
        if current_chunk is None:
            current_chunk = split
        else:
            if len(current_chunk.page_content) < min_length:
                # 내용 합치기
                current_chunk.page_content += "\n\n" + split.page_content
                
                # 메타데이터 업데이트 (법률 구조 정보는 첫 번째 청크 기준)
                current_chunk.metadata.update({
                    # 'merged_chunks': current_chunk.metadata.get('merged_chunks', 1) + 1,
                    'chunk_length': len(current_chunk.page_content),
                    # 'is_merged': True
                })
                
                # 추가 법률 구조 정보가 있으면 업데이트
                if split.metadata.get('legal_article') and not current_chunk.metadata.get('legal_article'):
                    current_chunk.metadata['legal_article'] = split.metadata.get('legal_article')
                if split.metadata.get('legal_chapter') and not current_chunk.metadata.get('legal_chapter'):
                    current_chunk.metadata['legal_chapter'] = split.metadata.get('legal_chapter')
                if split.metadata.get('legal_section') and not current_chunk.metadata.get('legal_section'):
                    current_chunk.metadata['legal_section'] = split.metadata.get('legal_section')
                
                # # 항 정보는 누적
                # current_paragraphs = current_chunk.metadata.get('legal_paragraph', [])
                # new_paragraphs = split.metadata.get('legal_paragraph', [])
                # if isinstance(current_paragraphs, list) and isinstance(new_paragraphs, list):
                #     current_chunk.metadata['legal_paragraph'] = list(set(current_paragraphs + new_paragraphs))
                
            else:
                # 현재 청크가 충분히 크면 저장하고 새로운 청크 시작
                current_chunk.metadata['chunk_length'] = len(current_chunk.page_content)
                merged_splits.append(current_chunk)
                current_chunk = split
    
    # 마지막 청크 처리
    if current_chunk is not None:
        current_chunk.metadata['chunk_length'] = len(current_chunk.page_content)
        merged_splits.append(current_chunk)
    
    # 최종 청크 인덱스 재정렬
    for i, split in enumerate(merged_splits):
        # split.metadata['final_chunk_index'] = i
        split.metadata['page_chunk_id'] = f"page_{page_num}_chunk_{i}"
    
    return merged_splits

all_splits = []
error_files = []
for idx, file in enumerate(files):
    if file.endswith(".pdf") ==  False:
        continue
    FILE_PATH = "/Users/hongbikim/Dev/dacon_finance/docs_none/" + file
    file_id = uuid.uuid4().hex

    # PyMuPDF 로더 인스턴스 생성
    loader = PyMuPDFLoader(FILE_PATH)

    # 문서 로드
    docs = loader.load()

    for doc in docs:
        # 페이지별로 처리
        page_num = doc.metadata.get('page', 0)
        content = clean_text(doc.page_content)
        
        # 조문/장절 정보 추출
        legal_structure = extract_legal_structure(content)
        
        # 텍스트 분할
        splits = base_splitter.split_text(content)
        
        # 임시로 분할된 청크들 생성
        temp_splits = []
        for i, split_content in enumerate(splits):
            chunk_structure = extract_legal_structure(split_content)
            
            split_doc = Document(
                page_content=split_content,
                metadata={
                    # **doc.metadata,
                    # 'chunk_index': i,
                    'file_name': FILE_PATH.split("/")[-1],
                    'file_id': file_id,
                    'page_chunk_id': f"page_{page_num}_chunk_{i}",
                    # 'legal_article': chunk_structure.get('article'),
                    # 'legal_chapter': chunk_structure.get('chapter'),
                    # 'legal_section': chunk_structure.get('section'),
                    # 'legal_paragraph': chunk_structure.get('paragraph'),
                }
            )
            temp_splits.append(split_doc)
        
        # 짧은 청크들 합치기
        merged_splits = _merge_short_chunks(temp_splits, page_num)
        try:
            print(f"[{idx}/{len(files)}]")
            vector_store.add_documents(documents=merged_splits)
        except Exception as e:
            print(f"Error adding documents for file {i}th {file}: {e}")
            error_files.append(file)
            continue
        all_splits.extend(merged_splits)

print(len(files))
print(len(all_splits))

