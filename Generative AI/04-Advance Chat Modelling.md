## 1. Function/Tool Calling:

[Tool calling](https://python.langchain.com/docs/concepts/#functiontool-calling) allows a chat model to respond to a given prompt by "calling a tool". Tool calling is a general technique that generates structured output from a model, and you can use it even when you don't intend to invoke any tools. An example use-case of that is extraction from unstructured text.

>[!NOTE]
>While the name "tool calling" implies that the model is directly performing some action, this is actually not the case! The model only generates the arguments to a tool, and actually running the tool (or not) is up to the user.


![[Pasted image 20241003095426.png]]

### How to pass tool outputs to chat models!

![[Pasted image 20241003095702.png]]![[Pasted image 20241003095705.png]]

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAi
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o-mini")
%%  %%
# Decorate function to use it as a tool
@tool
def add(a: int, b: int) -> int:
# Tool description -> model will use this to determine if this is a suitable tool for a task or not
    """Adds a and b.""" 
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


tools = [add, multiply]
llm_with_tools = llm.bind_tools(tools) # bind tools to out model


query = "What is 3 * 12? Also, what is 11 + 49?"  
messages = [HumanMessage(query)]    
ai_msg = llm_with_tools.invoke(messages)  
print(ai_msg.tool_calls)  
messages.append(ai_msg)
```

```Shell
[{'name': 'multiply', 'args': {'a': 3, 'b': 12}, 'id': 'call_GPGPE943GORirhIAYnWv00rK', 'type': 'tool_call'}, {'name': 'add', 'args': {'a': 11, 'b': 49}, 'id': 'call_dm8o64ZrY3WFZHAvCh1bEJ6i', 'type': 'tool_call'}]
```

`NOTE:` Model only has created arguments for the tools now we can decide to invoke them or not

```python
for tool_call in ai_msg.tool_calls:
    selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
    tool_msg = selected_tool.invoke(tool_call) # since we have wraped it using @tool else use eval()
    messages.append(tool_msg) # This will help model determine the ans for each tool call model think off
    
# Now u will get the dezire result
llm_with_tools.invoke(messages) # Generate Model Response after having answer
```

>[!NOTE]
>Note that each `ToolMessage` must include a `tool_call_id` that matches an `id` in the original tool calls that the model generates. This helps the model match tool responses with tool calls.
