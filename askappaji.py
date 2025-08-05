from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
import requests
import os
from getpass import getpass
from pinecone import Pinecone
from pinecone import ServerlessSpec
from torch import cuda
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chat_models import init_chat_model
import pdb
import time
import transformers
from torch import cuda, bfloat16
from langchain_community.llms import HuggingFacePipeline
from datasets import load_dataset

# """DATA PREPROCESSING"""

# batch_size = 32
# data = load_dataset("ss6638/AppajiSpeeches",data_dir="data", split="train",)
# data = data.to_pandas()
# print('datalen',len(data))
# pineconde_docs=[]
# for i in range(0, len(data), batch_size):
#     i_end = min(len(data), i+batch_size)
#     batch = data.iloc[i:i_end]
#     ids = [f"{x['discourse_date']}-{str(i)}" for i, x in batch.iterrows()]
#     pineconde_docs.extend([x['discourse_chunk'] for i, x in batch.iterrows()])
# pinecone_docs = pineconde_docs


"""SETTING UP PINECONE"""
PINECONE_API_KEY = '4b46f5d6-ba2e-4fc7-950d-c64897e4ed02' #os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = 'appaji-speeches-v2'
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        # dimension of the vector embeddings produced by OpenAI's text-embedding-3-small
        dimension=384,
        metric="cosine",
        # parameters for the free tier index
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# Initialize index client
index = pc.Index(name=index_name)
index.describe_index_stats()


"""EMBEDDING DOCUMENTS"""
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

def clean_url_for_title(url):
    # do url_title, chunk_num to enable subscriptable hashing and replacement
    # grabs the end of the url minus .md
    return url.split("/")[-1].replace(".md", "")
def generate_ids(doc_chunk):
    # Here, we follow a schema that puts the document name, and the chunk number together, like doc1#chunk1
    title = clean_url_for_title(doc_chunk.metadata['source'])
    chunk_num = doc_chunk.metadata['chunk_num']
    feature = doc_chunk.metadata['feature'] if 'feature' in doc_chunk.metadata else "na"
    return f"release_{title}#feature_{feature}#chunk_num{chunk_num}"

ids = [generate_ids(doc) for doc in pinecone_docs]
vector_store.add_documents(documents=pinecone_docs, ids=ids)


"""LLM SETUP"""
model_dir = '/local/zemel/weights/llama-7b' # Meta-Llama-3-70B-Instruct,gemma-2-9b-it
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)
hf_auth = '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'
model_config = transformers.AutoConfig.from_pretrained(model_dir,use_auth_token=hf_auth)
model = transformers.AutoModelForCausalLM.from_pretrained(model_dir,trust_remote_code=True,config=model_config,quantization_config=bnb_config,use_auth_token=hf_auth,)
model.to("cuda")
model.eval()
tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir,use_auth_token=hf_auth,)
generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    temperature=0.3,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=1024,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)
llm = HuggingFacePipeline(pipeline=generate_text)







"""USING LLM"""
query = "Tell me about version 7.0 of the Pinecone Python SDK"

retrieved_docs = vector_store.similarity_search(query, k=5)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

prompt = f'''You are an assistant that answers questions exclusively about the 
Pinecone SDK release notes:

Here's a question: {query}

Here's some context from the release notes:

{docs_content}


Question: {query}

Answer:
'''

# This will take a few seconds to run, due to the generation of the response from OpenAI
answer = llm.invoke(prompt)
pdb.set_trace()
for num, d in enumerate(retrieved_docs):
    print(f"Doc number: {num+1}")
    print(d.page_content)
    print("Metadata:")
    print(d.metadata)
    print("-"*100)
