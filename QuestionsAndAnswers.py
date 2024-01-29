import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
from langchain.llms import OpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings

# account for deprecation of LLM model
import datetime
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"

file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file, encoding='ISO-8859-1')
print("I have a CSV file")

# VectorstoreIndexCreator Configuration
# Issue: The current version of langchain raises a ValidationError due to compatibility issues with pydantic.
# Current Pydantic Version: 1.10.10. Using this specific version as a temporary solution to avoid the issue.
# Workaround: Using VectorstoreIndexCreator with the specified vectorstore_cls and loader.
# Future Update: Awaiting an update to langchain that resolves this compatibility issue with newer versions of Pydantic.
# Once updated, this section may require adjustments to align with the new version's requirements and functionalities.
# Relevant GitHub Issue: [ValidationError: 2 validation errors for PydanticOutputFunctionsParser](https://github.com/langchain-ai/langchain/issues/9815)
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

print("I have an index")

query ="Please list all your shirts with sun protection \
in a table in markdown and summarize each one."

llm_replacement_model = OpenAI(temperature=0, 
                               model='gpt-3.5-turbo-instruct')

response = index.query(query, 
                       llm = llm_replacement_model)

def format_response(md_table):
    """
    Format a regular response string into a Markdown table.
    
    Args:
    response (str): The input response string to be formatted.
    
    Returns:
    str: The formatted Markdown table.
    """
    # Split the response into lines
    lines = response.split('\n')

    # Assuming the first line contains headers separated by '|'
    headers = lines[0].split('|')
    # Calculate column widths based on headers
    col_widths = [len(header.strip()) for header in headers]

    # Process each line
    formatted_lines = [lines[0]]  # Add headers as-is
    for line in lines[1:]:
        if not line.strip():
            continue  # Skip empty lines
        cols = line.split('|')
        # Adjust col_widths if there are more columns in the data rows than in the headers
        for i, col in enumerate(cols):
            if i >= len(col_widths):
                col_widths.append(len(col.strip()))
            else:
                col_widths[i] = max(col_widths[i], len(col.strip()))

        # Create formatted line
        formatted_line = '|'.join(col.strip().ljust(col_widths[i]) for i, col in enumerate(cols))
        formatted_lines.append(formatted_line)

    # Adding the Markdown table formatting
    header_line = '|'.join('-' * width for width in col_widths)
    formatted_lines.insert(1, header_line)  # Inserting the separator line after headers

    return '\n'.join(formatted_lines)

# Formatting the response
formatted_response = format_response(response)

print(formatted_response)
print("\n")

# Embedding part of the code
print("I'm going to load the documents: \n")
docs = loader.load()
print(docs[0])
print("\n")

print("I'm going to create the embeddings: \n")
embeddings = OpenAIEmbeddings()
embed = embeddings.embed_query("Hi my name is Harrison")
print(len(embed))
print(embed[:5])
print("\n")

print("I'm going to create the DocArrayInMemorySearch database: \n")
db = DocArrayInMemorySearch.from_documents(
    docs, 
    embeddings
)
query = "Please suggest a shirt with sunblocking"
docs = db.similarity_search(query)

print("I'm going to print the results: \n")
print("Document Length: ",len(docs))
print("\n")
print("Query: ",query)
print("\n")
print("Results: ")
print(docs[0])
print("\n")

# RetrievalQA part of the code
print("I'm going to create the RetrievalQA object: \n")
retriever = db.as_retriever()
llm = ChatOpenAI(temperature = 0.0, model=llm_model)
qdocs = "".join([docs[i].page_content for i in range(len(docs))])

response = llm.call_as_llm(f"{qdocs} Question: Please list all your \
shirts with sun protection in a table in markdown and summarize each one.") 

# Formatting the response
formatted_response = format_response(response)
print(formatted_response)
print("\n")

print("I'm going to create the RetrievalQA object using the 'stuff' chain: \n")
qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)

query =  "Please list all your shirts with sun protection in a table \
in markdown and summarize each one."

response = qa_stuff.run(query)
formatted_response = format_response(response)
print(formatted_response)

