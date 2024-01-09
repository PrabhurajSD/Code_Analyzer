#!/usr/bin/env python
# coding: utf-8

# In[49]:


get_ipython().system('pip install openai langchain tiktoken deeplake')


# In[51]:


import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake

os.environ['OPENAI_API_KEY'] = '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'
embeddings = OpenAIEmbeddings()


# In[52]:


get_ipython().system('activeloop login -t eyJhbGciOiJIUzUxMiIsImlhdCI6MTcwNDgzMzMzNiwiZXhwIjoxNzM2NDU1NzMxfQ.eyJpZCI6InByYWJodXJhaiJ9.Vk_EZplPk9X8Dw1BVNvTcFfLFUod_hFSMaAq5Hw0YYQhl5FOIx91EIpxFxJusXIgT5SJxVXuD2ToXLUWA5V3ig')


# In[53]:


get_ipython().system('pip install GitPython')
from git import Repo


# In[ ]:


#!mkdir test_repo


# In[65]:


repo_path = "/content/test_repo"
git_Url = "https://github.com/PrabhurajSD/Fine_Tuned_Llama_2"


# In[66]:


repo = Repo.clone_from(git_Url, to_path=repo_path)


# In[67]:


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


# In[68]:


from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter (chunk_size=1000, chunk_overlap=0) 
texts = text_splitter.split_documents(docs)


# In[69]:


repo_name = git_url.split("/")[-1].replace(".git", "")
db = DeepLake.from_documents(texts, embeddings, dataset_path="hub://prabhuraj/code_Reader_1")


# In[70]:


db = DeepLake(dataset_path="hub://prabhuraj/code_Reader_1", read_only=True, embedding_function= embeddings)


# In[71]:


retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['fetch_k'] = 100
retriever.search_kwargs['maximal_marginal_relevance'] = True 
retriever.search_kwargs['k'] = 20


# In[72]:


def filter(x):
    # filter based on source code
    if 'com.google' in x['text'].data()['value']: 
        return False
    
    
    # filter based on path e.g. extension
    metadata = x['metadata'].data()['value']
    return 'scala' in metadata['source'] or 'py' in metadata['source']


#if we want to add filters we can do it by apply above function 
# retriever.search_kwargs['filter'] = filter


# In[73]:


from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

model = ChatOpenAI(model='gpt-3.5-turbo') # switch to 'gpt-4'
qa = ConversationalRetrievalChain.from_llm(model,retriever=retriever)


# In[81]:


questions = [
"can you show me the part of code of fine_tuned_llama_2.py which is responsible for fine tuning ?"
]
chat_history = []

for question in questions:
    result = qa({"question": question, "chat_history": chat_history})
    chat_history.append((question, result['answer']))
    print(f"-> **Question**: {question} \n")
    print(f"**Answer**: {result['answer']} \n")


# In[ ]:




