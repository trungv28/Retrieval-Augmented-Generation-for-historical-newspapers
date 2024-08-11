# Create sample data
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import load_dataset
from tqdm import tqdm

dataset = load_dataset("miracl/miracl-corpus", "fr")
model_name = "intfloat/e5-small"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

def transfer(start, end):
    docs = []
    ids = []
    for idx in range(start, end):
        doc = Document(page_content=dataset[idx]['text'],
                    metadata={
                        'title': dataset[idx]['title']
                    })
        docs.append(doc)
        ids.append(dataset[idx]['docid'])
    return docs, ids
docs, ids = transfer(0, 10000)
persist_directory = "corpus_db"

vectordb = Chroma.from_documents(documents=docs, embedding=hf, ids = ids, persist_directory=persist_directory) 

vectordb.persist()
def batch_process(batch_size):
    for i in tqdm(range(10000, 1000000, batch_size)):
        docs, ids = transfer(i, i+batch_size)
        vectordb.add_documents(documents = docs, ids = ids)
batch_size = 10000
batch_process(batch_size)

title_list = list(set([doc['title'] for doc in vectordb.get()['metadatas']]))
docs = []
for title in title_list:
    doc = Document(page_content=title)
    docs.append(doc)
persist_directory = "title_db"
titledb = Chroma.from_documents(documents=docs, embedding=hf, persist_directory=persist_directory)