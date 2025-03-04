Pydantic is a Python ==library for data validation and parsing==. It allows you to define data models that specify the expected structure and types of data, and then use those models to validate and parse incoming data.

**Key Features:**

* **Data Validation:** Verifies that data meets specified constraints (e.g., type, range, format).
* **Data Parsing:** Converts raw input data into Python objects.
* **Type Annotations:** Enforces data types using Python type hints.
* **Model Generation:** Creates Pydantic models from existing data structures.
* **Serialization and Deserialization:** Converts Pydantic models to and from JSON, XML, and other formats.

**Benefits:**

* **Improved Data Quality:** Ensures data consistency and accuracy by enforcing validation rules.
* **Simplified Data Handling:** Provides a concise and easy-to-use interface for data manipulation.
* **Increased Code Readability:** Type annotations and model definitions make code more self-explanatory.
* **Faster Development:** Automates data validation and parsing, reducing development time.

```python
% pip install pydantic

from pydantic import BaseClass

# Every cls u define should inherit from BaseClass
# Base class provide u some additional functionality such as validation and parsing
# This should be a data-class,i.e only define attributes and type
class Person(BaseClass):
	first_name: str
	last_name: str
	age: int

person=Person(first_name="abc",last_name="xyz",age=20)
# person=Person(first_name="abc",last_name="xyz",age=20)
# if u try to pass,some different dtype then pydantic will first try to convert them to specified dtype if not possible throws an error


from pydantic import ValidationError

try:
	person=Person("Deepanshu") 
	# throws exception as last_name and age is not provided
except ValidationError as ex:
	print(ex)
	print(ex.errors()) # list down all errors 
	print(ex.json()) # usefull while making apis


# To make any field have multiple values or making em optional
class Person(BaseClass):
	first_name: str = "Deepanshu" # default value
	last_name: str=None # optional 
	age: int | None # optional
	mobile_no: [str] | str  # list of string or a single string
```

## Serialization and Deserialization

U can convert these Data-models to python dtypes such as dict,list & can also convert python dtypes or json,XML objects to these pydantic Models

```python
from pydantic import BaseClass

class Person(BaseClass):
	first_name: str
	last_name: str
	age: int

p=Person("Deepanshu","Bhatt",22)

# Serialization
p.dict() # Person() => dict
p.json() # Person() => json
#u can exclude some fields while serializing
p.dict(exclude={"last_name","first_name"}) #set of attrs


# Deserialization
from DateTime import date
class Person(BaseClass):
	first_name: str
	last_name: str
	dob: date

# dict
data={
	"first_name":"Deepanshu",
	"last_name":"Bhatt",
	"dob":date(2002,7,30),
}

# parse_obj is coming from the BaseClass -> python-dtypes => DataModel
p=Person.parse_obj(data)

# json string
data='''
{
	"first_name":"Deepanshu",
	"last_name":"Bhatt",
	"dob":"2002-07-30",
}
'''

# parse_obj is coming from the BaseClass: python-dtypes => DataModel
p=Person.parse_obj(data)
```

## Field in Pydantic

A `Field` in Pydantic is a data validator and parser that defines the expected properties of a data field. It is used to specify the data type, constraints, and other options for a field in a Pydantic model.

**Example:**

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    first_name: str = Field(..., min_length=3, max_length=50)
    # alias mean,while de-serialing it will use the alias value mentioned to identify the field and not "last_name"
    last_name: str = Field(alias="LastName",default=None)
    age: int = Field(..., gt=0, lt=70)
    email: str = Field(..., regex="^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$")
```

In this example, the `Field` for the `name` field specifies that it must be a string, have a minimum length of 3 characters, and a maximum length of 50 characters. The `Field` for the `age` field specifies that it must be an integer, greater than 0, and less than 120. The `Field` for the `email` field specifies that it must be a string that matches a valid email address format.

**Use Case:**

Fields are used in Pydantic models to:

* **Enforce Data Validation:** Ensure that data meets the specified constraints, preventing invalid or inconsistent data from being processed.
* **Simplify Data Parsing:** Convert raw input data into Python objects, automatically handling type conversion and validation.
* **Improve Code Readability:** Provide clear and concise definitions of data expectations, making code easier to understand and maintain.

By using Fields in Pydantic models, you can improve the quality and consistency of your data, simplify data handling, and enhance the readability of your code.

## Config Class in PyDantic

The `Config` class in Pydantic is a way to configure the behavior of a Pydantic model. It allows you to specify options that affect the way the model is validated, parsed, and serialized.

The `Config` class is defined as a nested class within a Pydantic model. For example:

```python
from pydantic import BaseModel, Field, Config,EmailStr

# in pydantic-v1
class User(BaseModel):
    first_name: str = Field(...,default="abc",alias="firstName")
    first_name: str = Field(...,default="None",alias="lastName")
    
    age: int = Field(...,gt=0,le=75)
    password: str = Field(...,min_len=8,max_len=15,regex="exp")
    email: EmailStr = Field(...,default=None)

    class Config:
    # use the actual field name to deserialize and not alias
        allow_population_by_field_name= True
        
    # If while deserialing we get extra field,parse em or not    
        extra=False # default=False
	
	# converting field-names to alias,can be useful for webDev as 
	# json uses camelCase and python snake_case
		alias_generator=function 


# in pydantic-v2
from pydatic import ConfigDict
class User(BaseModel):
	model_config=ConfigDict(populate_by_name=True,extra=False) # use this model_config prop for config
	
    first_name: str = Field(...,default="abc",alias="firstName")
    first_name: str = Field(...,default="None",alias="lastName")
    
    age: int = Field(...,gt=0,le=75)
    password: str = Field(...,min_len=8,max_len=15,regex="exp")
    email: EmailStr = Field(...,default=None)

p.json(by_alias=True) # if while serializing u want to use alias and not field name 
```

Other common options that can be configured in the `Config` class include:

* `allow_population_by_field_name`: ==Allow fields to be populated by their field name instead of their alias.==
* `arbitrary_types_allowed`: Allow arbitrary types (i.e., types that are not natively supported by Pydantic) to be used in the model.
* `extra`: Specify additional fields that are not defined in the model.
* `json_encoders`: A dictionary of custom JSON encoders to use for specific data types.
* `schema_extra`: Additional properties to include in the OpenAPI schema for the model.

The `Config` class provides a flexible way to customize the behavior of Pydantic models to suit your specific needs.

## Strict and Constrained Type-Checking

```python
from pydantic import BaseClass,conint,constr

class Person(BaseClass):
	first_name:str
	last_name:constr(min_len=2,strip_whitespace=True,strict=True)
	age:conint(gt=0,le=70)
```

## Custom Validator
Help us to make our own custom validators and also transform the value if required

```python
from pydantic import BaseClass,field_validator,constr

class Dummy(BaseClass):
	hash_tag:constr(min_len=4,strip_white=True)
	
	@field_validator('hash_tag') # Note first all field constraints are verified and then after we use this
	@classmethod # validator are class-methods as they need to validate before creating an instance of class
	# value: **actual**-field value, 
	# values: all field-values that are validated correctly
	def validate_hash_tag(cls,value,values):
		try:
			if not value.startsWith("#"):
				raise ValueError("hash-tag should star with #")
		
		return value.lower() # convert it to lowercase and return
```


### Example: Nested Models using Composition

Consider a scenario where we have an `Address` model nested within a `User` model, and a `Company` model that includes a list of `User` models.

```python
from pydantic import BaseModel, Field, EmailStr
from typing import List

class Address(BaseModel):
    street: str = Field(..., max_length=100, description="Street name and number")
    city: str = Field(..., max_length=50, description="City name")
    state: str = Field(..., max_length=20, description="State or province")
    zip_code: str = Field(..., regex=r'^\d{5}$', description="5-digit zip code")

class User(BaseModel):
    id: int = Field(..., title="User ID", description="The unique identifier for a user")
    name: str = Field(..., title="Full Name", max_length=100)
    email: EmailStr = Field(..., description="The email address of the user")
    address: Address = Field(..., description="The user's address")

class Company(BaseModel):
    name: str = Field(..., title="Company Name", max_length=100)
    employees: List[User] = Field(..., description="List of employees working in the company")

# Example usage
address = Address(
    street="123 Main St",
    city="Anytown",
    state="Anystate",
    zip_code="12345"
)

user = User(
    id=1,
    name="John Doe",
    email="john.doe@example.com",
    address=address
)

company = Company(
    name="Tech Corp",
    employees=[user]
)

print(company)
```

### Explanation

- **Address Model**: Represents an address with `street`, `city`, `state`, and `zip_code` fields.
- **User Model**: Contains `id`, `name`, `email`, and an `address` field of type `Address`.
- **Company Model**: Includes a `name` and a list of `User` models in the `employees` field.

### Benefits of Using Nested Models

1. **Modularity**: Each model is defined separately, making it easier to manage and understand.
2. **Reusability**: Models like `Address` can be reused in different contexts.
3. **Validation**: Pydantic automatically validates nested structures, ensuring data integrity across complex models.
4. **Clarity**: Nesting models makes the relationships between different pieces of data explicit and clear.

By using nested models with composition, you can handle complex data structures in a clean and efficient manner, leveraging Pydantic’s powerful validation capabilities.

#Note There is lot of things pydantic offers read [docs](https://www.bing.com/ck/a?!&&p=23289b70b5e8999aJmltdHM9MTcxNjc2ODAwMCZpZ3VpZD0wZWUzOWRlMC03MWIyLTY5YTMtMTUyZC04ZWE4NzAxYTY4MjgmaW5zaWQ9NTIyOQ&ptn=3&ver=2&hsh=3&fclid=0ee39de0-71b2-69a3-152d-8ea8701a6828&psq=pydantic&u=a1aHR0cHM6Ly9kb2NzLnB5ZGFudGljLmRldi9sYXRlc3Qv&ntb=1) and also refer to pydantic videos of [MathByte]([(680) MathByte Academy - YouTube](https://www.youtube.com/@mathbyteacademy))
