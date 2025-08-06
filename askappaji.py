from pinecone import Pinecone
from pinecone import ServerlessSpec
from torch import cuda
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import pdb
import time
import transformers
from torch import cuda, bfloat16
from langchain_community.llms import HuggingFacePipeline
from datasets import load_dataset
import gradio as gr

# """DATA PREPROCESSING"""

batch_size = 32
data = load_dataset("ss6638/AppajiSpeeches",data_dir="data", split="train",)
data = data.to_pandas()
print('datalen',len(data))
appaji_speeches_texts_data=[]
appaji_speeches_metadata=[]
for i in range(0, len(data), batch_size):
    i_end = min(len(data), i+batch_size)
    batch = data.iloc[i:i_end]
    texts = [f"{x['discourse_chunk']}-{str(i)}" for i, x in batch.iterrows()]
    metadatas = [{"date": x['discourse_date'], "link": x['discourse_link']} for i, x in batch.iterrows()]
    appaji_speeches_texts_data.extend(texts)
    appaji_speeches_metadata.extend(metadatas)


"""SETTING UP PINECONE"""
PINECONE_API_KEY = '4b46f5d6-ba2e-4fc7-950d-c64897e4ed02' #os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = 'appaji-speeches-v3'
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

# Check if index already has documents
index_stats = index.describe_index_stats()
existing_vectors = index_stats.get('total_vector_count', 0)

if existing_vectors > 0:
    print(f"Index already contains {existing_vectors} vectors")
    
    # Check for duplicates based on metadata
    existing_count = 0
    new_texts = []
    new_metadatas = []
    
    for text, metadata in zip(appaji_speeches_texts_data, appaji_speeches_metadata):
        # Create a unique ID based on date and link
        doc_id = f"{metadata.get('date', '')}_{metadata.get('link', '')}".replace(" ", "_").replace("/", "_")[:512]
        
        # Check if this ID exists in the index
        try:
            fetch_response = index.fetch(ids=[doc_id])
            if doc_id in fetch_response.get('vectors', {}):
                existing_count += 1
                continue
        except:
            pass
        
        # Add text hash to metadata for future reference
        enhanced_metadata = metadata.copy()
        enhanced_metadata['text_hash'] = str(hash(text))[:20]  # Store hash for comparison
        
        new_texts.append(text)
        new_metadatas.append(enhanced_metadata)
    
    print(f"Found {existing_count} existing documents, adding {len(new_texts)} new documents")
    
    if new_texts:
        # Use custom IDs based on metadata for easier duplicate detection
        ids = [f"{meta.get('date', '')}_{meta.get('link', '')}".replace(" ", "_").replace("/", "_")[:512] for meta in new_metadatas]
        texts = vector_store.add_texts(new_texts, metadatas=new_metadatas, ids=ids)
    else:
        print("No new documents to add")
else:
    print(f"Index is empty, adding all {len(appaji_speeches_texts_data)} documents")
    # For first-time addition, create IDs based on metadata
    ids = [f"{meta.get('date', '')}_{meta.get('link', '')}".replace(" ", "_").replace("/", "_")[:512] for meta in appaji_speeches_metadata]
    # Add text hash to metadata for future duplicate detection
    enhanced_metadatas = []
    for text, metadata in zip(appaji_speeches_texts_data, appaji_speeches_metadata):
        enhanced_metadata = metadata.copy()
        enhanced_metadata['text_hash'] = str(hash(text))[:20]
        enhanced_metadatas.append(enhanced_metadata)
    texts = vector_store.add_texts(appaji_speeches_texts_data, metadatas=enhanced_metadatas, ids=ids)


"""LLM SETUP"""
model_dir = '/local/zemel/weights/Llama-2-7b-chat-hf' # Meta-Llama-3-70B-Instruct,gemma-2-9b-it
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



"""ASK APPAJI FUNCTION"""
def ask_appaji(query):
    retrieved_docs = vector_store.similarity_search(query, k=5)
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    # Llama-2-chat format with system prompt
    system_prompt = "You are a helpful assistant that answers questions exclusively based on the context provided from Appaji's speeches. Be concise and accurate. If the context doesn't contain relevant information, say 'I don't have enough information from Appaji's speeches to answer this question.'"
    
    prompt = f'''<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

Here is possibly relevant context from the speeches:
{docs_content}

Question: {query} [/INST]'''
    print('starting llm invocation')
    response = llm.invoke(prompt)
    print('got response from llm')
    # Extract answer from Llama-2 format response
    if "[/INST]" in response:
        answer = response.split("[/INST]")[-1].strip()
    else:
        answer = response.strip()
    answer += "\n\nSources:\n\n"
    for num, d in enumerate(retrieved_docs):
        answer+=f"Source {num+1}: " + d.page_content + f"[{d.metadata['link']}]" + "\n\n" 
    return answer

demo = gr.Interface(
    fn=ask_appaji,
    inputs=gr.Textbox(label="Question"),
    outputs=gr.Textbox(label="Answer"),
    title="Ask Appaji",
    description="Ask questions about Appaji's speeches"
)
demo.launch(share=True)