# FastAPI Notes

## Introduction

FastAPI is a modern, high-performance, web framework for building APIs with Python 3.7+ based on standard Python type hints.

## Key Features

* **Fast:** Very high performance, on par with NodeJS and Go.
* **Fast to code:** Increase the speed to develop features by about 200% to 300%.
* **Fewer bugs:** Reduce about 40% of human (developer) induced errors.
* **Intuitive:** Great editor support. Completion everywhere. Less time debugging.
* **Easy:** Designed to be easy to use and learn. Less time reading docs.
* **Short:** Minimize code duplication. Multiple features from each parameter declaration. Fewer bugs.
* **Robust:** Get production-ready code. With automatic interactive documentation.
* **Standards-based:** Based on (and fully compatible with) the open standards for APIs: OpenAPI (formerly known as Swagger) and JSON Schema.

## Installation

```bash
pip install fastapi uvicorn
```

## Basic Example

```python
from fastapi import FastAPI
import uvicorn # similar to nodemon

# Creating an instance of fastapi
app = FastAPI()

# defining the method/operation,route/path and function we want to execute
@app.get("/")
async def root()->dict[str,str]:
    return {"message": "Hello World"}

if __name__="__main__":
	uvicorn.run(app,host="127.0.0.1",port=8000) # if u don't use this it defaults to port 8000
```

**Running the app:**

```bash
uvicorn main:app --reload 
```

**(main:app refers to main.py and the FastAPI instance named app)**


## Path Parameters

```python
from fastapi import FastAPI,HTTPException

app = FastAPI()

@app.get("/items/{item_id}") # "{variable}"
async def read_item(item_id: int)->dict[str,str]: 
# also define it here to get access of that variable,type-hinting will let fastapi auto convert the value to given type automatically,else it would be str
    item=db.find({_id:item_id})
    if item==None:
	    raise HTTPException(status_code=404,
		    detail='item not found',headers=[{}])
	return item

# Note always have path with dynamic routing below ones, that don't accept any parameters

# This won't work as when u execute prog,python do line-by-line execution so items/"something" will hit the above route and the control never reaches here.
@app.get("/items/sold_out")
async def sold_out_items():
	return {"sold_out_items":["list of sold out items"]}
```



## Query Parameters

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/") # url: "url/items?limit=10&skip=20"
# Note if u use any parameter in function with no-default value it is required by default
# TO handle this either provide a default value or use pydantic-fields
async def read_item(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}


# Either use default values for query params or use `None` else FastAPI will always except a qury parameter from you
@app.get("/songs")
async def get_songs(genre:str|None=None)->list:
	genre=genre.lower()
	if genere:
		return filter(lambda song=>song.genre==genre,songs)
	return songs
```

## Request Body

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Fastapi use Pydantic for validation and one can define their validation data-class
# This dataclass is required whenever we want to send any data from client side to server
# Includes all validation checks,dtypes to validate the request body
class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

@app.post("/items/")
async def create_item(item: Item)->Item:
    return item
```

## Data Validation

FastAPI uses Pydantic for data validation.  The `Item` model above demonstrates basic type validation.  Pydantic supports complex validation rules.

## Automatic Interactive Documentation

FastAPI automatically generates interactive API documentation using Swagger UI. Access it at `/docs` or `/redocs` after running your application.


## Asynchronous Requests

FastAPI supports asynchronous requests using `async` and `await`.

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    # Perform some asynchronous operation
    # ...
    return {"message": "Hello Async"}

```

## Annotated Type 

`Annotated` allows for more sophisticated and custom validation rules.

**What is `Annotated`?**

`Annotated` is a type hint from the `typing` module (available in Python 3.8+). It allows you to wrap a base type with additional metadata, typically validation constraints. 
FastAPI's Pydantic integration interprets this metadata to perform validation during request processing.

```python
from fastapi import FastAPI, Annotated
from pydantic import Field, PositiveInt, validator

app = FastAPI()

# Using Annotated with Pydantic validators
@app.post("/users/")
async def create_user(
    age: Annotated[PositiveInt, Field(ge=18, le=120, description="User's age")]
):
    return {"age": age}

# Using a custom validator
@app.post("/users_custom/")
async def create_user_custom(
    username: Annotated[str, validator("username", min_length=3, max_length=20)]
):
    return {"username": username}
```


#### Why Use `Annotated` if we have `BaseModel`:

With 'Annotated' we can add more constraints on some Base-class without changing it permanently or redefining.

```python
from pydantic import BaseModel

class UserBase(BaseModel):
    name: str
    email: str

class UserCreate(UserBase):
    password: str

#-----------------------------------------
from typing import Annotated
from fastapi import FastAPI
from pydantic import BaseModel, Field, EmailStr, validator

app = FastAPI()

class UserBase(BaseModel):
    name: str
    email: str

@app.post("/users/")
async def create_user(user: Annotated[UserBase, Field(description="User details")]):
    return {"user": user}

@app.post("/users/strict/")
async def create_strict_user(
    user: Annotated[UserBase, validator("email", email_regex=r"[^@]+@[^@]+\.[^@]+")]
):
    return {"user": user}
```

We can validate different types of params using annotated also for eg:

```python
from typing import Annotated

from fastapi import FastAPI, Query

app = FastAPI()


@app.get("/items/")
async def read_items(q: Annotated[str | None, Query(max_length=50)] = None):
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q})
    return results
```

> [!Tip]
> Here we are using `Query()` because this is a **query parameter**. Later we will see others like `Path()`, `Body()`, `Header()`, and `Cookie()`, that also accept the same arguments as `Query()`.
> 

## Further Exploration

* [Official Documentation](https://fastapi.tiangolo.com/)
* [Advanced User Guide](https://fastapi.tiangolo.com/advanced/)
