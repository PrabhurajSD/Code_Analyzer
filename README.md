# Code_Analyzer
This project enables seamless and intuitive conversations about your code repositories, offering code suggestions, improvements, and insights in a natural language format.

To use this codebase chatbot, follow these steps:
1. copy the code from Code_Analyzer.py
2. Paste it in any Python Code editor of your choice
3. Get your OpenAI API key and paste it into the below code line of Code_Analyzer.py:
   os.environ['OPENAI_API_KEY'] = 'enter your OpenAi API key'
4. Now create your deelake account.
5. Create your Data container to store embeddings.
6. Generate the Deeplake API key and copy it into the below code line of Code_Analyzer.py:
   get_ipython().system('activeloop login -t **enter deeplake token**)
7. Replace repo_path location with your preferred location where you want to clone the repo.
8. After login into activeloop, copy the link of the GitHub Repository you want to analyze into the below code line of Code_Analyzer.py:
   git_Url = "paste the link of repo you want to Analyze"
9. Replace the deeplake dataset path with your Org path
10. All things are set, Now you are ready to go just run the code
