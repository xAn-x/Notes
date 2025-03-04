## 1st Program:
## **Installation**

```shell
pip install langchain langchain-openai langchain-google-vertextai
```

## **Simple Chatbot:**

```env

LANGCHAIN_API_KEY="..."
OPENAI_API_KEY="..."
GOOGLE_API_KEY="..."
``` 
```python
from dotenv import load_env
from langchain_openai import ChatOpenAI
from lanchain_google_vertexai import ChatVertexAI

load_env()

model=ChatOpenAI(
	model='gpt-4o-mini'
)

model=ChatVertexAI(
	model='google-gemma-1.5-flash'
)


from langchain_core.messages import HumanMessage,SystemMessage,AiMessage

from lanchain_core.parser import StrOutputParser # SimpleStringParser

output_parser=StrOutputParser()

chat_history=[
	SystemMessage(content='You are a math educator and only answer math related query,rather than providing direct answer to user query provide with some hints like what topic or methods can be used to solve the question,until user ask explicitly or is really strugling.')
]

query=input("User: ")
chat_history.append(HumanMessage(content=query))


response=model.invoke(chat_history) # AiMessage(content="",max_tokens="",...)
output=output_parser.invoke(response) # message
```

## Using LCEL (LangChain Execution Language):

LCEL is a fast and efficient way to write your Langchain queries, using it u can chain multiple steps involved in your application into a single powerful chain. Each chain takes user input and outputs the content in the way u want to format it. U can have any number of steps in b/w and can even branch or execute multiple chains in ||.

**LCEL is way to create arbitrary custom chains. It is built on _Runnable_ protocol.** It provide u with feature to invoke a chain synchronously and asynchronously and even stream the response so user don't have to wait.

```python
from dotenv import load_env
from lanchain_google_vertexai import ChatVertexAI

load_env()

model=ChatVertexAI(
	model='google-gemma-1.5-flash'
)

from langchain_core.messages import HumanMessage,SystemMessage,AiMessage
from lanchain_core.parser import StrOutputParser
from lanchain_core.prompts import ChatPromptTemplate

# Help u format your query dynamically so the model can adapt and respond accordingly
system_prompt=ChatPromptTemplate.from_messages(
	[
		('system':'Translate it to {language}'),
		('human':'{query}')
	]
)

output_parser=StrOutputParser()

# Without LCEL
prompt_result=system_prompt.invoke({'language':'French','query':'Hello'})
response=model.invoke(prompt_result)
output=output_parser(response)


# With LCEL
chain=system_prompt | model | output_parser
output=chain.invoke({'language':'French','query':'Hello'})
```

# Custom Runnable in LCEL:

```python
from lanchain_core.runnables import RunnableLambda

runnable=RunnableLambda(lambda x:str(x))
runnable.invoke(2) # '2'
runnable.stream(2) # generator-obj
await runnable.ainvoke(2) # '2'

# Can batch multiple inputs also
runnable.batch([2,3,4])


# Can chain multiple runnable
runnable1=RunnableLambda(lambda x:x['foo'])
runnable2=RunnableLambda(lambda x:[x]*2)
chain=runnable1 | runnable2  
chain.invoke({'foo':7}) # ---runnable1--> 7 ---runnable2-->[7]*2 : [7,7]


# can invoke runnables in ||
from langchain_core.runnable import RunnableParallel
chain=RunnableParallel(branch_1=runnable1,branch_2=runnable2)
chain.invoke({'foo':7}) # [{'branch_1':7},('branch_2':[{'foo':7},{'foo':7}])]


# U can merge the output of 1 runnable with its input if reqire
form langchain_core.runnables import RunnablePassThrough # merge input of runnable with its output
runnable=RunnableLambda(lambda x:x['foo']+10)
chain=RunnablePassThrough(bar=runnable)
chain.invoke({'foo':7}) # {'foo':7,'bar':10}


# Branching based on some condn
from langchain_core import RunnableBranch
system_prompt=ChatPrompTemplate.from_messages([
	("system","Classify user feedback into one the 3 classes :positive,negative,neutral"),
	("human","{feedback}")
])

positive_template=ChatPrompTemplate.from_messages([
	("system","Ask user what he likes about the product and any other feature he think i.e missing?"),
	("human","{feedback}")
])
negative_template=ChatPrompTemplate.from_messages([
	("system","Ask user the areas where the product can be improved and create a list out of it?"),
	("human","{feedback}")
])
neutral_template=ChatPrompTemplate.from_messages([
	("system","Ask,what user like and what changes he wants?"),
	("human","{feedback}")
])


branches=RunnableBranch(
	{
		lambda output: "positive" in output, # Return true if this is the branch
		positive_template | model | StrOutputParser()
	},{
		lambda output: "negative" in output,
		negative_template | model | StrOutputParser()
	},{
		lambda output: "neutral" in output,
		neutral_template | model | StrOutputParser()
	}
)

chain=system_prompt | model | StrOutputParser() | RunnableBranch 

# U can get the graph,config what to do when a run start/end read more @ their site...
```