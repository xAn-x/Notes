 # Python

```cmd
pip instal openai

setx OPENAI_API_KEY=MY_API_KEY # setting open ai api key in windows
```

```python
import openai

openai.api_key="API_KEY"
openai.list_models() # list of all available model
```

```python
from openai import OpenAI

client=OpenAI()
completion=client.chat.completions.create(
	model="gpt-4o-mini",
	messages=[
		{
			"role":"system", # system:determine the behaviour of model
			"content":"You are a helpfull assistant."
		},{
			"role":"user", # query
			"content":"Write me python code,to solve sudoku."
		}
	]
)

print(completion.choices[0].message)
```
## Generate Image:

```python
from openai import OpenAI
client = OpenAI()

response = client.images.generate(
    prompt="A cute baby sea otter",
    n=2,
    size="1024x1024"
)

print(response.data[0].url)
```
## Create Vector Embeddings:

```python
from openai import OpenAI
client = OpenAI()

response = client.embeddings.create(
    model="text-embedding-3-large",
    input="The food was delicious and the waiter..."
)

print(response)
```


## Function Calling:

Function calling allows you to connect models like `gpt-4o` to external tools and systems. This is useful for many things such as empowering AI assistants with capabilities, or building deep integrations between your applications and the models.

![[Pasted image 20240923072244.png]]

**Use Cases:**
1. **Enabling assistants to fetch data:** an AI assistant needs to fetch the latest customer data from an internal system when a user asks “what are my recent orders?” before it can generate the response to the user
2. **Enabling assistants to take actions:** an AI assistant needs to schedule meetings based on user preferences and calendar availability.
3. **Enabling assistants to perform computation:** a math tutor assistant needs to perform a math computation.
4. **Building rich workflows:** a data extraction pipeline that fetches raw text, then converts it to structured data and saves it in a database.
5. **Modifying your applications' UI:** you can use function calls that update the UI based on user input, for example, rendering a pin on a map.

> [!NOTE]
>  In function calling model never actually executes functions itself, instead in step 3 the model simply generates parameters that can be used to call your function, which your code can then choose how to handle, likely by calling the indicated function.

### How to use function calling

_1. Pick up a function/s  in your codebase that the model should be able to call:_
The starting point for function calling is choosing a function in your own codebase that you’d like to enable the model to generate arguments for.

_2. Describe your function/s to model so it knows how to call it:_
This definition describes both what the function does (and potentially when it should be called) and what parameters are required to call the function.

The `parameters` section of your function definition should be described using JSON Schema. If and when the model generates a function call, it will use this information to generate arguments according to your provided schema.

_3.Pass your function/s definitions as available `functions`  to model along with message:_
Next we need to provide our function definitions within an array of available “functions” when calling the model

_4.Recieve and handle models response:_
If model decide no function is required, then the response will contain a direct reply to user in normal way.

In an assistant use case you will typically want to show this response to the user and let them respond to it, in which case you will call the API again (with both the latest responses from the assistant and user appended to the `messages`).

If the model generated a function call, it will generate the arguments for the call (based on the `parameters` definition you provided).

_5.Handling the Models response indicating a function should be called:_
Provide the function call result back to model so model can generate actual response that the user will see.

_This is how the output will resemble if model decide that some function should be used_

```cmd
response = {
    "choices": [
        {
            "message": {
                "role": "assistant",
                "function_call": {
					"type": "function",
					"name": "function-name",
					"argument":"arguments"
                }
            }
        }
    ]
}
```

***Example-1:***

```python
import openai
openai.api_key="My_API_KEY"

client=openai.OpenAI()


# Helper Functions
def async book_my_restraunt(restraunt_name:str,time:DateTime):
	time=str(time)
	api_endpoint=f"restro/api/book/?restro={restraunt_name}&time={time}"
	try:
		response=await requests.get(api_endpint)
		return f"Successfull Registraintion:{response["message"]}"
	except e:
		return f"Registration Failed:{response["message"]}"



def async order_ride(source:str,destination:str,time):
	time=str(time) if time is None else Date.now()
	api_endpoint=f"uber/ride/book/?start={source}&dest={destination}"
	try:
		response=await requests.get(api_endpint)
		return f"Successfull Registraintion:{response["message"]}.Your total Charge is {response["charge"]}"
	except e:
		return f"Registration Failed:{response["message"]}"



# Function Descriptions
book_me_a_restraunt={
	"type":"function",
	"function": {
		"name": "book_my_restraunt",
		"description": "Help u booking any sort of restraunt or hotel,given its name and time",
		"parameters": {
			"restraunt_name":{
				"type":"str",
				"description":"Name of Restraunt the user wants it booking for."
			},
			"time":{
				"type":"DateTime",
				"description":"Time when user wants it booking."
			},
		},
		"required": ["restraunt_name","time"],
		"additionalProperties": False,
	}
}


order_me_a_ride={
	"type":"function",
	"function": {
		"name": "order_ride",
		"description": "Help u in booking a cab for user.",
		"parameters": {
			"source":{
				"type":"str",
				"description":"Starting/Pickup Point"
			},
			"destination":{
				"type":"str"
				"description":"Destination/Dropout Point",
			},
			"time":{
				"type":"DateTime"
				"description":"Time when the user wants its ride.If not provided use current time",
			},
			"required": ["source","destination"],
			"additionalProperties": False,
		},
	}
}


# Define Model
messages=[{
	"role":"System",
	"content":"You are a helpfull assistant that resolve queries of end user.Given list of function description if u think to use some function that can help then use it."
}],


# Input Query
query=input()
resp.messages.append(query)

resp=client.chat.completions.create(
	model="gpt-4o-mini",
	messages=messages,
	functions=[book_me_a_restraunt,order_me_a_ride],
	function_call="auto"
)

output=resp.choices[0].message

if output["function_call"] is None:
	pass
else:
	# get arguments from the function that model has detected
	func_args=json.load(output['function_call']['arguments']) 
	function_to_call=eval(output['function_call'][0])
	result=function_to_call(**func_args)
	# feeding the output after calling function back to model 
	output=resp.chat.completions.create(
		model="gpt-4o-mini",
		messages=messages.append({"user":"function","name":output['function_call']['name'],
		"content":result})
	)

print(output=resp.choices[0].message)
```


## Assistant API:
The Assistants API allows you to build AI assistants within your own applications. An Assistant has instructions and can leverage models, tools, and files to respond to user queries. _Supports 3 types of tools Code Interpreter, File Search, and Function calling._

### How assistant API works:
1. Assistants can call OpenAI’s **[models](https://platform.openai.com/docs/models)** with specific instructions to tune their personality and capabilities.
2. Assistants can access **multiple tools in parallel**. These can be both OpenAI-hosted tools — like [code_interpreter](https://platform.openai.com/docs/assistants/tools/code-interpreter) and [file_search](https://platform.openai.com/docs/assistants/tools/file-search) — or tools you build / host (via [function calling](https://platform.openai.com/docs/assistants/tools/function-calling)).
3. Assistants can access **persistent Threads**. Threads simplify AI application development by storing message history and truncating it when the conversation gets too long for the model’s context length. _You create a Thread once, and simply append Messages to it as your users reply._
4. Assistants can access files in several formats — either as part of their creation or as part of Threads between Assistants and users. When using tools, Assistants can also create files (e.g., images, spreadsheets, etc) and cite files they reference in the Messages they create.

![[diagram-assistant.webp]]

| Object    | What it represents                                                                                                                                                                                                           |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Assistant | Purpose-built AI that uses OpenAI’s [models](https://platform.openai.com/docs/models) and calls [tools](https://platform.openai.com/docs/assistants/tools)                                                                   |
| Thread    | A conversation session between an Assistant and a user. Threads store Messages and automatically handle truncation to fit content into a model’s context.                                                                    |
| Message   | A message created by an Assistant or a user. Messages can include text, images, and other files. Messages stored as a list on the Thread.                                                                                    |
| Run       | An invocation of an Assistant on a Thread. The Assistant uses its configuration and the Thread’s Messages to perform tasks by calling models and tools. As part of a Run, the Assistant appends Messages to the Thread.      |
| Run Step  | A detailed list of steps the Assistant took as part of a Run. An Assistant can call tools or create Messages during its run. Examining Run Steps allows you to introspect how the Assistant is getting to its final results. |
### Assistant API Quickstart:
A typical integration of the Assistants API has the following flow:

1. Create an [Assistant](https://platform.openai.com/docs/api-reference/assistants/createAssistant) by defining its custom instructions and picking a model. If helpful, add files and enable tools like Code Interpreter, File Search, and Function calling.
2. Create a [Thread](https://platform.openai.com/docs/api-reference/threads) when a user starts a conversation.
3. Add [Messages](https://platform.openai.com/docs/api-reference/messages) to the Thread as the user asks questions.
4. [Run](https://platform.openai.com/docs/api-reference/runs) the Assistant on the Thread to generate a response by calling the model and the tools.

__NOTE: Calls to the Assistants API require that you pass a beta HTTP header. This is handled automatically if you’re using OpenAI’s official Python or Node.js SDKs. `OpenAI-Beta: assistants=v2`__

***Example-1:***

```python
from openai import OpenAI
client = OpenAI()

# Create Assistant
assistant = client.beta.assistants.create(
  name="Math Tutor",
  instructions="You are a personal math tutor. Write and run code to answer math questions.",
  tools=[{"type": "code_interpreter"}],
  model="gpt-4o",
)

# Create Thread: Chat History
thread = client.beta.threads.create()

# Add message as query
message = client.beta.threads.messages.create(
  thread_id=thread.id,
  role="user",
  content="I need to solve the equation `3x + 11 = 14`. Can you help me?"
)




# ================= Without Streaming =====================

# runs are synchronous i.e u need to continously monitor the status until terminated
# create and poll: helper func provided by openai to assist both in creating the run and then polling for its completion.
run=client.beta.threads.runs.create_and_poll(
	thread_id=thread.id,
	assistant_id=assistant.id,
	instructions=['Adress user as champ'] 
)

# Once the Run completes, you can list the Messages added to the Thread by the Assistant.
if run.status == 'completed': 
  messages = client.beta.threads.messages.list(
    thread_id=thread.id
  )
  print(messages)
else:
  print(run.status)

# You may also want to list the Run Steps of this Run if you'd like to look at any tool calls made during this Run.




# =================== With Streaming =====================

# Run the assistant
from typing_extensions import override
from openai import AssistantEventHandler

# First, we create a EventHandler class to define
# how we want to handle the events in the response stream.
# This EventHandler help u manage different stages of a assiatant 
# Give u all information about the run steps of the assistant so u can monitor what assistant is doing under the hood
class EventHandler(AssistantEventHandler):    
  @override
  def on_text_created(self, text) -> None:
    print(f"\nassistant > ", end="", flush=True)
      
  @override
  def on_text_delta(self, delta, snapshot):
    print(delta.value, end="", flush=True)
      
  def on_tool_call_created(self, tool_call):
    print(f"\nassistant > {tool_call.type}\n", flush=True)
  
  def on_tool_call_delta(self, delta, snapshot):
    if delta.type == 'code_interpreter':
      if delta.code_interpreter.input:
        print(delta.code_interpreter.input, end="", flush=True)
      if delta.code_interpreter.outputs:
        print(f"\n\noutput >", flush=True)
        for output in delta.code_interpreter.outputs:
          if output.type == "logs":
            print(f"\n{output.logs}", flush=True)


# Then, we use the `stream` SDK helper 
# with the `EventHandler` class to create the Run 
# and stream the response.
with client.beta.threads.runs.stream(
  thread_id=thread.id,
  assistant_id=assistant.id,
  instructions="Please address the user query.", # overide the instructions of assistant same if passed tools
  event_handler=EventHandler(),
) as stream:
  stream.until_done()
```

### Deep dive Assistant API:
#### Run Lifecycle:

![[Pasted image 20240923072858.png]]

| Status            | Definition                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `queued`          | When Runs are first created or when you complete the `required_action`, they are moved to a queued status. They should almost immediately move to `in_progress`.                                                                                                                                                                                                                                                                                                                                                                                         |
| `in_progress`     | While in_progress, the Assistant uses the model and tools to perform steps. You can view progress being made by the Run by examining the [Run Steps](https://platform.openai.com/docs/api-reference/runs/step-object).                                                                                                                                                                                                                                                                                                                                   |
| `completed`       | The Run successfully completed! You can now view all Messages the Assistant added to the Thread, and all the steps the Run took. You can also continue the conversation by adding more user Messages to the Thread and creating another Run.                                                                                                                                                                                                                                                                                                             |
| `requires_action` | When using the [Function calling](https://platform.openai.com/docs/assistants/tools/function-calling) tool, the Run will move to a `required_action` state once the model determines the names and arguments of the functions to be called. You must then run those functions and [submit the outputs](https://platform.openai.com/docs/api-reference/runs/submitToolOutputs) before the run proceeds. If the outputs are not provided before the `expires_at` timestamp passes (roughly 10 mins past creation), the run will move to an expired status. |
| `expired`         | This happens when the function calling outputs were not submitted before `expires_at` and the run expires. Additionally, if the runs take too long to execute and go beyond the time stated in `expires_at`, our systems will expire the run.                                                                                                                                                                                                                                                                                                            |
| `cancelling`      | You can attempt to cancel an `in_progress` run using the [Cancel Run](https://platform.openai.com/docs/api-reference/runs/cancelRun) endpoint. Once the attempt to cancel succeeds, status of the Run moves to `cancelled`. Cancellation is attempted but not guaranteed.                                                                                                                                                                                                                                                                                |
| `cancelled`       | Run was successfully cancelled.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `failed`          | You can view the reason for the failure by looking at the `last_error` object in the Run. The timestamp for the failure will be recorded under `failed_at`.                                                                                                                                                                                                                                                                                                                                                                                              |
| `incomplete`      | Run ended due to `max_prompt_tokens` or `max_completion_tokens` reached. You can view the specific reason by looking at the `incomplete_details` object in the Run.                                                                                                                                                                                                                                                                                                                                                                                      |

#### Polling for updates:
If you are not using [streaming](https://platform.openai.com/docs/assistants/overview/step-4-create-a-run?context=with-streaming), in order to keep the status of your run up to date, you will have to periodically [retrieve the Run](https://platform.openai.com/docs/api-reference/runs/getRun) object. You can check the status of the run each time you retrieve the object to determine what your application should do next.

You can optionally use Polling Helpers in our [Node](https://github.com/openai/openai-node?tab=readme-ov-file#polling-helpers) and [Python](https://github.com/openai/openai-python?tab=readme-ov-file#polling-helpers) SDKs to help you with this. These helpers will automatically poll the Run object for you and return the Run object when it's in a terminal state.

#### Thread locks:
When a Run is `in_progress` and not in a terminal state, the Thread is locked. This means that:

- New Messages cannot be added to the Thread.
- New Runs cannot be created on the Thread.

![[Pasted image 20240923073235.png]]

Most of the interesting detail in the Run Step object lives in the `step_details` field. There can be two types of step details:

1. `message_creation`: This Run Step is created when the Assistant creates a Message on the Thread.
2. `tool_calls`: This Run Step is created when the Assistant calls a tool. Details around this are covered in the relevant sections of the [Tools](https://platform.openai.com/docs/assistants/tools) guide.


## Function Calling vs Assistant API

|                  | Why                                                                                                                                                                                                                                  |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Function Calling | Allows Developers to define functions that model can call by generating parameters. The execution of these functions occurs externally, giving developers control over how to handle these function calls based on user input        |
| Assistant API    | Integrates function calling directly within its response, executing functions in OpenAI's infrastructure without needing an external server. This simplifies implementation but limits capabilities to what OpenAI offers internally |

## Structured Output:
Structured Outputs is a feature that ensures the model will always generate responses that adhere to your supplied [JSON Schema](https://json-schema.org/overview/what-is-jsonschema), so you don't need to worry about the model omitting a required key, or hallucinating an invalid enum value.

Some benefits of Structed Outputs include:

1. **Reliable type-safety:** No need to validate or retry incorrectly formatted responses
2. **Explicit refusals:** Safety-based model refusals are now programmatically detectable
3. **Simpler prompting:** No need for strongly worded prompts to achieve consistent formatting

In addition to supporting JSON Schema in the REST API, the OpenAI SDKs for [Python](https://github.com/openai/openai-python/blob/main/helpers.md#structured-outputs-parsing-helpers) and [JavaScript](https://github.com/openai/openai-node/blob/master/helpers.md#structured-outputs-parsing-helpers) also make it easy to define object schemas using [Pydantic](https://docs.pydantic.dev/latest/) and [Zod](https://zod.dev/) respectively.

```python
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI()

class CalendarEvent(BaseModel):
    event_name: str
    date: str
    participants: list[str]

completion = client.beta.chat.completions.parse(
    model="gpt-4o-mini-2024-07-18", # supported by this or later model only
    messages=[
        {"role": "system", "content": "Extract the event information."},
        {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."},
    ],
    response_format=CalendarEvent,
)

event = completion.choices[0].message.parsed
```


### When to use structured output via function calling vs via response_format:
Structured Outputs is available in two forms in the OpenAI API:

1. When using [function calling](https://platform.openai.com/docs/guides/function-calling)
2. When using a `json_schema` response format

So,
- If you are connecting the model to tools, functions, data, etc. in your system, then you should use function calling
- If you want to structure the model's output when it responds to the user, then you should use a structured `response_format`