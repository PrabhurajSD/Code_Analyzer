# Install all required libraries
get_ipython().system('!pip install openai langchain tiktoken deeplake') # run in cmd


import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake


os.environ['OPENAI_API_KEY'] = 'enter your OpenAi API key'
embeddings = OpenAIEmbeddings()

#login into deep lake activeloop
# run in cmd if require
get_ipython().system('activeloop login -t **enter deeplake token**) 

#run in cmd
get_ipython().system('pip install GitPython')
from git import Repo



!mkdir test_repo

repo_path = "/content/test_repo"
git_Url = "paste the link of repo you want to Analyze"


repo = Repo.clone_from(git_Url, to_path=repo_path)

import os
from langchain.document_loaders import TextLoader
root_dir = repo_path
docs = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        try:
            loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
            docs.extend(loader.load_and_split())
        except Exception as e:
            pass

from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter (chunk_size=1000, chunk_overlap=0) 
texts = text_splitter.split_documents(docs)

db = DeepLake.from_documents(texts, embeddings, dataset_path="hub://enter your org /code_Reader_1")


db = DeepLake(dataset_path="hub://prabhuraj/code_Reader_1", read_only=True, embedding_function= embeddings)

retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['fetch_k'] = 100
retriever.search_kwargs['maximal_marginal_relevance'] = True 
retriever.search_kwargs['k'] = 20


def filter(x):
    # filter based on source code
    if 'com.google' in x['text'].data()['value']: 
        return False
    
    
    # filter based on path e.g. extension
    metadata = x['metadata'].data()['value']
    return 'scala' in metadata['source'] or 'py' in metadata['source']


# if we want to add filters we can do it by apply above function 
# retriever.search_kwargs['filter'] = filter

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

model = ChatOpenAI(model='gpt-3.5-turbo') 
qa = ConversationalRetrievalChain.from_llm(model,retriever=retriever)


questions = [
"Can you show me the part of the code of fine_tuned_llama_2.py that is responsible for fine-tuning?"
]
chat_history = []

for question in questions:
    result = qa({"question": question, "chat_history": chat_history})
    chat_history.append((question, result['answer']))
    print(f"-> **Question**: {question} \n")
    print(f"**Answer**: {result['answer']} \n")



