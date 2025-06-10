
Langchain provide us with multiple building blocks that we can use directly or configure them according to our needs.

## 1.Prompt Templates
Are responsible for formatting user input into a format that can be passed to a language model. This is to improve the output of LLM by properly formatting the query.

```python
from langchain_core.prompt import PromptTemplate,ChatPromptTemplate,MessagesPlaceHolder

# Simple prompt template: to format single input, and are generally used for simple input
prompt_template=PromptTemplate.from_template("Tell me joke about {topic}")
prompt=prompt_template.invoke({'topic':'cats'})


# ChatPrompts: use to format list of message,consist of list of template themselves
prompt_template=ChatPromptTemplate.from_messages([
	("system","Translate it to {language}"),
	("human","{query}"),
])
prompt=prompt_template.invoke({'language':'spanish',"query":"Hello,Mate"})


# MessagesPlaceHolder: This template is resp for adding list of messages in a paricular place.
prompt_template=ChatPromptTemplate.from_messages([
	("system","You are a helpful assistant."),
	MessagesPlaceHolder("msgs")
])
prompt=prompt_template.invoke({"msgs":[HumanMessage("What is the capital of France?")]})
# alternate way
prompt_template=ChatPromptTemplate.from_messages([
	("system","You are a helpful assistant."),
	("placeholder","{msgs}") # <--- this is the changed part
])
```

### Few shot Learning:
Providing examples to LLM on how to respond/behave can greatly improve its performance can be easily achieved by providing few examples, this is called few-show-prompting.

```python
form langchain_core.prompts import PromptTemplate,FewShotPromptTemplate

# config a formatter: will format few-shot examples into string
prompt_template=PromptTemplate("Question:{question}\n{answer}")

# Examples that ur LLM will use for refrence
egs=[
	 {"question":"What is the capital of India",answer:"The capital of India is Delhi"},
	 {"question":"What is general eqn of parabola",answer:"The general eqn of parabol is ax^2+bx+c=0;a!=0"},	 
]
print(prompt_template.invoke(egs[0]).to_string())

# Pass egs to FewShotPromptTemplate: takes in few egs and a formatter and formats all egs in that format
few_shot_prompt_template=FewShotPromptTemplate(
	examples=egs,
	example_prompt=prompt_template, # formatter,
	suffix="Question:{input}\n",
	input_variables=['input']
)

prompt=few_shot_prompt_template.invoke({"input":"Who is Noeter,what she is famous for"}).to_string()


# Few-shot Learning with Chat Model
from langchain_claude import Claude
from langchain_core.prompts import ChatPromptTemplate,FewShotChatPromptTemplate
from langchain_core.parser import StrOutputParser 

model=Claude(model="claude-sonnet-3.5",temprature=0.4)

formatter=ChatPromptTemplate.from_messages([
	('human':"{input}"),
	('ai':"{output}")
])

few_shot_examples=[
	{"input":"2 😀 4 😀 6", "output":"12"},
	{"input":"5*2 😀 10", "output":"20"},
]

fewshot_prompt=FewShotChatPromptTemplate(
	example_prompt=formatter,
	examples=few_shot_examples
)

final_prompt=ChatPromptTemplate.from_messages([
	("system":"You are a math wizard"),
	fewshot_prompt,
	("human":"{input}")
])

chain=final_prompt | model | StrOutputParser
chain.invoke({"input":"What is -3*2😀6"})
```

**Learn about Dynamic Few Shot prompting, that allows u to dynamically selects only few of the provided examples, based on query.**

### How to compose Prompts together
U can compose different parts of prompts together, with langchain. It provide a friendly interface for either string or chat prompts.

```python
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_core.messages import HumanMessage,AiMessage,SystemMessage

# Using Strings
prompt=PromptTemplate.from_template("Tell me a joke about {topic}"+",make it funny"+" and in {language}")

# Using Chat prompts
system_prompt=SystemMessage(content="You are a sarcastic assistant,that use puns while answering any queries")
prompt=ChatPromptTemplate.from_messages(
	system_prompt+HumanMessage(content="Hi")+AiMessage(content="What?")+{input}
)
prompt.invoke({"input":"How can i solve reimann hypothesis"})

# Using PipelinePrompt: can be usefull when u want to reuse parts of prompts,consist of 2 parts: 
		# final prompt: the final prompt to return
		# pipeline propmpts: list of tuples ,consisting string name and prompt template.Each prompt template will 
		# be formatted and then passed to future prompt template

from langchain_core.prompts import PipelinePromptTemplate,PromptTemplate
full_template="""{introduction}

{example}

{start}
"""
full_prompt=PromptTemplate.from_template(full_template)

introduction_prompt="""You are immpersonating {person}"""

example_template="""Here is an example of interaction
Q:{example_q}
A:{example_a}
"""
example_prompt=PromptTemplate.from_template(example_template)

start_prompt=PromptTemplate("""Now do this for real,\nQ:{input}\nA:""")

pipeline_prompt=PipeLinePromptTemplate(
	final_prompt=full_prompt,pipeline_prompts=[introduction_prompt,example_prompt,start_prompt]
)

pipeline_prompt.invoke({"person":"Hitler","example_q":"Who desrve to die","example_a":"Those fuckin,Jews should burn","input":"How we can become great again?"})
```

## 2.Example Selectors
If u have large no. of examples then u may need to select only few example to include in the prompt. But how to decide which one to select, example selector helps u it that. 

| Name       | Description                                                                                 |
| ---------- | ------------------------------------------------------------------------------------------- |
| Similarity | Uses semantic similarity between inputs and examples to decide which examples to choose.    |
| MMR        | Uses Max Marginal Relevance between inputs and examples to decide which examples to choose. |
| Length     | Selects examples based on how many can fit within a certain length                          |
| Ngram      | Uses ngram overlap between inputs and examples to decide which examples to choose.          |

## 3.Chat Models

Chat Models are LLMs that are finetuned to act like an assistant that can responds according to query of user.[Chat Models](https://python.langchain.com/docs/concepts/#chat-models) are newer forms of language models that take messages in and output a message.
## 4.Messages

[Messages](https://python.langchain.com/docs/concepts/#messages) are the input and output of chat models. They have some `content` and a `role`, which describes the source of the message.
## 5.Output Parsers

[Output Parsers](https://python.langchain.com/docs/concepts/#output-parsers) are responsible for taking the output of an LLM and parsing into more structured format.

## LangChain 0.2 Output Parsers Table

| Parser Name               | Description                                               | Example Usage                                                              |
| ------------------------- | --------------------------------------------------------- | -------------------------------------------------------------------------- |
| `BaseOutputParser`        | Base class for all output parsers.                        |                                                                            |
| `StructuredOutputParser`  | Parses output into a structured format (e.g., JSON, CSV). |                                                                            |
| `JsonOutputParser`        | Parses output into a JSON object.                         | `parser = JsonOutputParser()`                                              |
| `CsvOutputParser`         | Parses output into a CSV object.                          | `parser = CsvOutputParser()`                                               |
| `PythonReplOutputParser`  | Parses output from a Python REPL.                         | `parser = PythonReplOutputParser()`                                        |
| `RegexParser`             | Parses output using a regular expression.                 | `parser = RegexParser(regex='(?<=key: ).*?(?=,)')`                         |
| `HtmlParser`              | Parses output from an HTML document.                      | `parser = HtmlParser(selector='h1')`                                       |
| `PromptOutputParser`      | Parses output based on a prompt template.                 | `parser = PromptOutputParser(template='{{output}}')`                       |
| `ChatOutputParser`        | Parses output from a chat conversation.                   | `parser = ChatOutputParser()`                                              |
| `SQLDatabaseOutputParser` | Parses output from a SQL database.                        | `parser = SQLDatabaseOutputParser(database_url='sqlite:///mydatabase.db')` |
| `OpenAIOutputParser`      | Parses output from OpenAI models.                         | `parser = OpenAIOutputParser(model_name='text-davinci-003')`               |

**Note:** Newer versions of LangChain may have additional parsers.

**Example Usage:**

```python
from langchain.output_parsers import JsonOutputParser

parser = JsonOutputParser()
output = parser.parse('{"key1": "value1", "key2": "value2"}')

print(output)
# Output: {'key1': 'value1', 'key2': 'value2'}
```

**Obsidian Copilot Integration:**

Obsidian Copilot can integrate with LangChain to provide enhanced note-taking capabilities. For example, you can use LangChain's output parsers to extract structured data from text, code, or web pages and store it in your Obsidian notes.

**Example:**

You can use LangChain's `HtmlParser` to extract specific data from a web page and store it in your Obsidian note:

```python
from langchain.output_parsers import HtmlParser
from langchain.chains import  LLMChain

# Define your LLM chain
llm_chain = LLMChain(llm=your_llm, prompt=your_prompt)

# Define your HTML parser
parser = HtmlParser(selector='h1')

# Run the LLM chain and parse the output
output = parser.parse(llm_chain.run(your_input))

# Store the output in your Obsidian note
# ...
```

This is just one example of how you can use LangChain's output parsers with Obsidian Copilot. The possibilities are endless!
## 6.Document Loaders

Document loaders are essential components in LangChain that allow you to load and process various types of documents for use with your language models. They act as bridges between your data sources and your LLM, making it easier to leverage the power of AI for tasks like question answering, summarization, and text generation.

**Types of Document Loaders:**

| Loader Type           | Description                                         | Example                                                    |
| --------------------- | --------------------------------------------------- | ---------------------------------------------------------- |
| **File Loaders**      | Load documents from local files.                    | `from langchain.document_loaders import TextLoader`        |
| **Directory Loaders** | Load documents from entire directories.             | `from langchain.document_loaders import DirectoryLoader`   |
| **Web Loaders**       | Load documents from web pages.                      | `from langchain.document_loaders import WebBaseLoader`     |
| **Database Loaders**  | Load documents from databases.                      | `from langchain.document_loaders import SQLDatabaseLoader` |
| **Custom Loaders**    | Create custom loaders for specialized data formats. | `from langchain.document_loaders import BaseLoader`        |

**Example:**

Imagine you have a collection of research papers in PDF format stored on your computer. You want to use LangChain to build a question answering system that can answer questions based on the contents of these papers.

1. **Document Loader:** You would use a `DirectoryLoader` to load all the PDFs from a specific folder.
2. **Processing:** The loader would convert each PDF into a structured format, such as a list of text chunks with metadata like page number and filename.
3. **LLM Integration:** You would then feed these processed documents to a language model, which could be used to answer questions about the research papers.

**Benefits:**

* **Simplify Data Access:**  Document loaders abstract away the complexities of data loading, allowing you to focus on building your AI applications.
* **Support Various Formats:**  LangChain provides loaders for a wide range of document formats, including text, PDF, HTML, and more.
* **Customizability:** You can create custom loaders to handle specialized data formats or specific data sources.

By using document loaders, you can easily integrate diverse data sources with LangChain and unlock the full potential of language models for various tasks.
## 7.Text Splitters

Text splitters are essential components in LangChain that help you _break down large pieces of text into smaller, manageable chunks_. This is crucial for working with language models, as they often have limitations on the amount of text they can process at once.

**Types of Text Splitters:**

## LangChain Text Splitters: A Comprehensive Guide

| Name                                | Classes                                                                                                                                                                                                                                                                                 | Splits On                             | Description                                                                                                                         | Adds Metadata |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| **Recursive**                       | [RecursiveCharacterTextSplitter](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/), [RecursiveJsonSplitter](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_json_splitter/) | A list of user-defined characters     | Recursively splits text, aiming to keep related pieces of text together. This is the recommended starting point for text splitting. |               |
| **HTML**                            | [HTMLHeaderTextSplitter](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/HTML_header_metadata/), [HTMLSectionSplitter](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/HTML_section_aware_splitter/)          | HTML-specific characters              | Splits text based on HTML structure, adding metadata about the source (e.g., heading level, section).                               | ✅             |
| **Markdown**                        | [MarkdownHeaderTextSplitter](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/markdown_header_metadata/)                                                                                                                                            | Markdown-specific characters          | Splits text based on Markdown structure, adding metadata about the source (e.g., heading level).                                    | ✅             |
| **Code**                            | [many languages](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/code_splitter/)                                                                                                                                                                   | Code (Python, JS) specific characters | Splits text based on characters specific to coding languages. 15 different languages are available.                                 |               |
| **Token**                           | [many classes](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/split_by_token/)                                                                                                                                                                    | Tokens                                | Splits text based on tokens, using various tokenization methods.                                                                    |               |
| **Character**                       | [CharacterTextSplitter](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/character_text_splitter/)                                                                                                                                                  | A user-defined character              | Splits text based on a user-defined character. One of the simplest methods.                                                         |               |
| **[Experimental] Semantic Chunker** | [SemanticChunker](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/semantic-chunker/)                                                                                                                                                               | Sentences                             | First splits on sentences and then combines adjacent sentences if they are semantically similar enough.                             |               |
| **AI21 Semantic Text Splitter**     | [AI21SemanticTextSplitter](https://python.langchain.com/v0.1/docs/integrations/document_transformers/ai21_semantic_text_splitter/)                                                                                                                                                      |                                       | Identifies distinct topics that form coherent pieces of text and splits along those.                                                | ✅             |
| `Custom Splitter`                   | [langchain-text-splitters: 0.2.4 — 🦜🔗 LangChain documentation](https://python.langchain.com/v0.2/api_reference/text_splitters/index.html#langchain-text-splitters-base)                                                                                                               |                                       | Create your own text splitter with your own rules                                                                                   |               |

**Key Points:**

* **Metadata:** Some splitters add metadata to each chunk, providing context about its origin (e.g., heading level, section).
* **Context Preservation:** Recursive and semantic splitters aim to keep related pieces of text together, preserving context.
* **Customization:** You can adjust parameters like chunk size, tokenization method, and splitting criteria to suit your specific needs.

By choosing the right text splitter, you can optimize your LangChain applications for efficient processing, accurate results, and improved understanding of your data.


**Intuitive Example:**

Imagine you have a long article about machine learning, and you want to use LangChain to summarize it.

1. **Text Splitter:** You would use a `SentenceTextSplitter` to break the article into individual sentences.
2. **LLM Integration:** You would then feed these sentences to a language model, which could generate a summary of the article based on the individual sentences.

**Benefits:**

* **Manageable Chunks:** Text splitters ensure that your text is broken down into manageable chunks that can be processed by your language model.
* **Context Preservation:**  Some splitters, like sentence splitters, preserve the context of the original text by splitting at natural boundaries.
* **Customizability:** You can create custom splitters to handle specific text splitting requirements, such as splitting based on specific keywords or patterns.

By using text splitters, you can effectively manage large text inputs and ensure that your language models can process them efficiently, leading to better results and improved performance.


## 8. Embeddings:

* **What they are:** Embeddings are numerical representations of text, transforming words and sentences into vectors of numbers. These vectors capture the semantic meaning of the text, allowing for similarity comparisons.
* **How they work:**  LangChain integrates with various embedding models (e.g., OpenAI, Hugging Face) to generate these representations.
* **Why they matter:** Embeddings enable powerful search and retrieval capabilities, allowing you to find similar documents or answer questions based on semantic understanding.

## 9. Vector Stores:

* **What they are:** Vector stores are databases specifically designed to store and query high-dimensional vectors (like embeddings).
* **How they work:** They use efficient indexing techniques to quickly retrieve vectors that are similar to a given query vector.
* **Why they matter:**  Vector stores allow you to efficiently search and retrieve relevant information from large collections of embedded documents.

## 10. Retrieval-Augmented Generation (RAG):

* **What it is:** RAG is a powerful technique that combines embeddings, vector stores, and language models to generate more informative and accurate responses.
* **How it works:** 
    1. A query is converted into an embedding.
    2. The vector store is searched for similar embeddings, retrieving relevant documents.
    3. These retrieved documents are then fed to a language model, which generates a response based on the retrieved context.
* **Why it matters:** RAG enables language models to access and leverage external knowledge, making them more knowledgeable and capable of answering complex questions.

**Example:**

Imagine you want to build a chatbot that can answer questions about a company's internal documentation.

1. **Embeddings:** You would use LangChain to embed the documentation into vectors.
2. **Vector Store:** You would store these vectors in a vector store.
3. **RAG:** When a user asks a question, the chatbot would:
    * Convert the question into an embedding.
    * Search the vector store for relevant documents.
    * Use a language model to generate a response based on the retrieved documents.

This process allows the chatbot to access and understand the company's documentation, providing more accurate and contextually relevant answers to user queries.

**In short:**  Embeddings, vector stores, and RAG form a powerful combination within LangChain, enabling you to build intelligent applications that can access and understand external knowledge to generate more insightful and accurate responses.
