import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

import langchain
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.evaluation.qa import QAGenerateChain
from langchain.evaluation.qa import QAEvalChain

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
data = loader.load()

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

# Set retriever
llm = ChatOpenAI(temperature = 0.0, model=llm_model)

# Set QA
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=index.vectorstore.as_retriever(), 
    verbose=True,
    chain_type_kwargs = {
        "document_separator": "<<<<>>>>>"
    }
)

#Coming up with test datapoints
print("Coming up with test datapoints: \n")
print(data[10], "\n", data[11], "\n")

#Hard-coded examples
print("Hard-coded examples \n")
examples = [
    {
        "query": "Do the Cozy Comfort Pullover Set\
        have side pockets?",
        "answer": "Yes"
    },
    {
        "query": "What collection is the Ultra-Lofty \
        850 Stretch Down Hooded Jacket from?",
        "answer": "The DownTek collection"
    }
]

#LLM-Generated examples
print("LLM-Generated examples: \n")
example_gen_chain = QAGenerateChain.from_llm(ChatOpenAI(model=llm_model))

generated_examples = example_gen_chain.apply_and_parse(
    [{"doc": t} for t in data[:5]]
)

# Restructuring the examples to match the expected format
new_examples = [{'query': ex['qa_pairs']['query'], 'answer': ex['qa_pairs']['answer']} for ex in generated_examples]


print(new_examples[0])
print("Data: \n")
print(data[0])

#Adding Hard coded examples to new examples
print("Adding Hard coded examples to new examples: \n")
examples += new_examples

#Running QA
print("Running QA: \n")
print(qa.run(examples[0]["query"]))

#Setting langchain.debug to True
print("Setting langchain.debug to True For Manual Evaluation: \n")
langchain.debug = True

print("Running QA again: \n")
qa.run(examples[0]["query"])

# Turn off the debug mode
print("Turning off the debug mode: \n")
langchain.debug = False

#LLM assisted evaluation
print("LLM assisted evaluation: \n")

predictions = qa.apply(examples)
llm = ChatOpenAI(temperature=0, model=llm_model)
eval_chain = QAEvalChain.from_llm(llm)

graded_outputs = eval_chain.evaluate(examples, predictions)

for i, eg in enumerate(examples):
    print(f"Example {i}:")
    print("Question: " + predictions[i]['query'])
    print("Real Answer: " + predictions[i]['answer'])
    print("Predicted Answer: " + predictions[i]['result'])
    print("Predicted Grade: " + graded_outputs[i]['results'])
    print()

print("Graded Outputs: \n")
print(graded_outputs[0])