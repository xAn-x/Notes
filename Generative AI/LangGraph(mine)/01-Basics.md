### Elements
1. **State:** A shared Data structure that holds current info or context of entire application.
   
   It's like applications memory keeping track of variables and data that nodes can access and modify.
   
2. **Nodes:** Individual func or oprns that perform specific task. Receives input (current state) -> process it -> output (updated state).

3. **Graph:** How different tasks (nodes) are connected and executed (workflow).

4. **Edges:** Connections b/w nodes determine the execution flow(which node will execute next). 

	There are also **conditional edges** that decide which node to execute based on condition.

5. **Tools:** Specialized functions or utils that nodes can utilize to preform specific task.
   
   Enhance the capabilities. **ToolNode** a special kind of node whose job is to run a tool. It connects the tool output back into state. 
 
### Messages

![[Pasted image 20250610204459.png]]

### Hello World 

```python
from langgraph import StateGraph,START,END
from typing import TypeDict

class State(TypeDict):
"Global shared state among agents"
	message:str|None

# nodes are func that process the state and return updated state
def greet(state:State)->State:
"Greet user(act as node)"
	# if u return state with a key that's not +nt initially it will add it
	return {"message":"Hello World!","additional":"Hola"}
	
builder=StateGraph(State)

# add nodes
builder.add_node("greet",greet)

# add edges
builder.add_edge(START,"greet")
builder.add_edge(greet,END)

graph=builder.compile() 
resulting_state=graph.invoke({"message":None})
# state={"message":"Hello World!","optional":"Hola"}
```

![[Pasted image 20250617201114.png]]

### Conditional Graph

```python
from langgraph import StateGraph,START,END
from typing import TypeDict,List,Literal

class State(TypeDict):
	values:List[int|float]
	opern:Literal["add","mult"]

def add_elements(state:State)->State:
	state["result"]=sum(state["values"])
	return state

def mult_elements(state:State)->State:
	state["result"]=reduce(
		lambda accum,current->accum*curr
		state["values"],1
	)
	return state

builder=StateGraph(State)

builder.add_node("add",add)
builder.add_node("mult",mult)
builder.add_node("router",lambda state:state) # passthrough state as it is

builder.add_conditional_edges(
	"router",
	lambda state: state["oprn"],
	# Output:Node
	{"add":"add","mult":"mult"}
)
builder.add_edge(START,router)
builder.add_edge(add,END)
builder.add_edge(mult,END)
```

![[Pasted image 20250617204402.png]]

### Looping Graph

```python
from langgraph import StateGraph,START,END
from typing import TypeDict,List,Literal

class State(TypeDict):
	name:str
	values:List[int]
	counter:int

def greet_user(state:State)->State:
	state["greetings"]=f"Hello,{state["name"]}!"
	state["counter"]=0
	return state

def random_number_geneator(state:State)->State:
	state["values"]=state["values"].append(random.randint(0,10))
	state["counter"]+=1
	return state


builder=StateGraph(State)

builder.add_node("greeter",greet_user)
builder.add_node("rand_gen",random_number_geneator)
builder.add_router("should_continue",lambda state:state)

builder.add_edge(START,"greeter")
builder.add_edge("greeter","random_number_geneator")
builder.add_edge("random_number_geneator","should_continue")

builder.add_conditional_edges(
	"should_continue",
	lambda state: "True" if state["counter"]<10: "False",
	{"True":"random_number_geneator","False":End}
)

graph=builder.compile({name="Cold"})
```

 ![[Pasted image 20250617225244.png]]

### Simple Bot using LangGraph

```python
from typing import TypeDict,List

from langchain_core.messages import SystemMessage,HumanMessage,AiMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import START,END,StateGraph
from dotenv import load_dotenv

load_dotenv()

class State(TypeDict):
	messages=List[HumanMessage|SystemMessage|AiMessage]

llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")


# =======================================

def chat(state:State)->State:
	print("Please Enter your query or type 'exit' to exit out")
	while True:
		inp=input("User: ")
		
		if inp.lower()=="exit":
			break
		 
		res=llm.invoke(state["messages"])
		print(res.content)
	
		state["messages"].append(content=HumanMessage(inp))
		state["messages"].append(content=res)
	
	return state

builder=StateGraph(state)

builder.add_node("chat",chat)
builder.add_edge(START,"chat")
builder.add_edge("chat",END)

graph=bulder.compile()

state=graph.invoke({messages=[]})
```


### Agent using LangGraph (React Agent)

```python
from typing import TypeDict,List

from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool


from langgraph.graph import START,END,StateGraph
from langgraph.graph.messages import add_message # reducer -> [prv_message] + new_message
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()

# State
class AgentState(TypeDict):
	messages=Annotated[List[BaseMessage],add_message]

# llm
llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# tools
@tool
def add(a,b):
"""Adds 2 numbers"""
	return a+b


@tool
def get_weather(city:str)->str:
"""Gets weather of a city"""
	return ...

tool_box=[add,get_weather]

# nodes
def call(state:AgentState)->AgentState:
"""Take user input and query the model"""
	system_message=[SystemMessage("You are a helpfull assistant")]

	inp=input("User: ")
	state["messages"]=HumanMessage(content=inp)
	res=llm.invoke(system_message+state["messages"])
	state["messages"]=res
	
	return state


def should_continue(state:AgentState)->AgentState:
	last_msg=state["messages"][-1]
	if not last_msg.tool_calls:
		return "end"
	return "continue"

# building graph
builder=StateGraph(AgentState)

builder.add_node("agent",call)
builder.add_node("should_continue",shoud_continue)
tool_node=ToolNode(tools=tool_box)
builder.add_node("tools",tool_node)

builder.add_edge(START,"agent")
builder.add_edge("tools","agent")

builder.add_conditional_edges(
	"our_agent",
	should_continue,
	{
		"continue":"tools",
		"end":END
	}
)

graph=builder.compile()
state=graph.invoke()
```

![[Pasted image 20250619194733.png]]

### Drafter Agent (Human in Loop)

```python
from typing import TypeDict,List

from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool


from langgraph.graph import START,END,StateGraph
from langgraph.graph.messages import add_message 
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()



# Document content to be updated
# Here We are passing document like this so tools can have access to it , but in prod we will use 'injections' to do so.
document="""
..............
"""


class AgentState(TypeDict):
	messages=List[BaseMessage,add_messages]

@tool
def update_doc(content:str):
	""" Update the document with the provided content 
		
		Args:
			content: str = Updated content
	"""
	global document
	document=content

@tool
def save_document(filename:str)->str:
	"""Save the current document to a text file 

		Args:
			filename: name of the text file
	"""
	global document
	
	if not filename.endswith(".txt"):
		filename+=".txt"
	try:
		with open(filename,'w') as file:
			file.write(document)
		print(f"Saved to {filename}")
	except Exception as e:
		return f"Failed to save {filename}\n{str(e)}"


tools=[update_doc,save_doc]

llm=ChatGoogleGenerativeAI(
	model="gemini-2.5-flash"
).bind_tools(tools)

def call(state:AgentState)->AgentState:
	global document
	
	system_prompt=[SystemMessage(content=f"""You are a drafter agent that need to draft the below message/document based on user query)
	
	document:
		{document}
	"""
	]


	query=input("What changes do you need: ")
	state["messages"]=HumanMessage(content=query)
	res=llm.invoke(system_prompt+state["messages"])
	
	state["messages"]=res
	state["messages"]=AiMessage(content="Do you want me to make any changes or should I save the document")
	res=input()
	state["messages"]=res

	return state

def is_user_satisfied(state:AgentState)->AgentState:
"""Based on given query, only answer in "yes"/"no" weather user is satisfied or not
"""
	msg=state["messages"][-1]
	if isinstance(msg,ToolMessage):
		return "yes"
	else:
		return "no"

tools=ToolNode(tools)


builder=StateGraph(AgentState)

builder.add_node("agent",call)
builder.add_node("tools",tools)

builder.add_edge(START,"agent")
builder.add_edge("agent",tools)
builder.add_conditional_edges(
	"tools",
	is_user_satisfied,
	{"yes":END,"no":agent}
)

graph=builder.compile()
graph.invoke*
```

![[Pasted image 20250619223210.png]]
