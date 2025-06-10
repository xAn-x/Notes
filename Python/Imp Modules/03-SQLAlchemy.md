SQLAlchemy is a Python SQL toolkit and Object Relational Mapper (ORM) that gives application developers the full power and flexibility of SQL.

Key Features:

* **ORM:** Maps Python classes to database tables, allowing you to interact with your database using Python objects.
* **SQL Expression Language:** Provides a flexible and powerful way to construct SQL queries programmatically in Python.
* **Connection Pooling:** Manages database connections efficiently, improving performance.
* **Schema Definition:** Allows you to define your database schema in Python code.
* **Various Database Support:** Supports a wide range of databases including PostgreSQL, MySQL, SQLite, and more.

```sh
pip install -u SQLAlchemy
```

```python
from sqlalchemy import create_engine,text

# Create a database engine (replace with your database URL)
engine = create_engine('sqlite:///:memory:',echo=True)

# Connect with DB
 with engine.connect() as conn:
     result = conn.execute(text("select 'hello world'"))
     print(result.all())

# Commiting Changes
# DBAPI connection doesn’t commit automatically, we need to do it manually
with engine.connect() as conn:
     conn.execute(text("CREATE TABLE some_table (x int, y int)"))
     conn.execute(
         text("INSERT INTO some_table (x, y) VALUES (:x, :y)"),
         [{"x": 1, "y": 1}, {"x": 2, "y": 4}],
     )
     conn.commit() # commit to make changes in db

# Fetching rows
with engine.connect() as conn:
    result = conn.execute(text("SELECT x, y FROM some_table"))
    for row in result:
        print(f"x: {row.x}  y: {row.y}")
	
	# if col-names have spaces
	for dict_row in result.mappings(): # return dict
	    x = dict_row["x"]
	    y = dict_row["y"]

# Sending parameters to a query
with engine.connect() as conn:
    result = conn.execute(text("SELECT x, y FROM some_table WHERE y > :y"),
		     {"y": 2})
    for row in result:
        print(f"x: {row.x}  y: {row.y}")

# Can also send multiple parameters
with engine.connect() as conn:
    conn.execute(
        text("INSERT INTO some_table (x, y) VALUES (:x, :y)"),
        [{"x": 11, "y": 12}, {"x": 13, "y": 14}], # replace x and y using all these entries
    )
    conn.commit()
```

### Working with Database MetaData

The central element of both SQLAlchemy Core and ORM is the **SQL Expression Language** which allows for fluent, composable construction of SQL queries. The foundation for these queries are Python objects that represent database concepts like tables and columns. These objects are known collectively as database metadata.

```python
from sqlalchemy import create_engine,MetaData,ForeignKey
from sqlalchemy import Table, Column, Integer, String # utility classes

metadata_obj = MetaData() # info about ur whole db, create this obj globally to share 

user_table = Table(
    "user_account", # name
    metadata_obj, # MetaData
    Column("id", Integer, primary_key=True),
    Column("name", String(30),nullable=False),
    Column("age", Integer),
)

address_table = Table(
    "address",
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("user_id", ForeignKey("user_account.id"), nullable=False),
    Column("email_address", String, nullable=False),
)

engine=create_engine()
metadata_obj.create_all(engine)

## CRUD oprns

# INSERT
insert_statement=user_table.insert().value([{"name":"Mike",age:27}])
res=conn.execute(insert_statement)

# Update
update_statement=user_table.update()\
	.where(user_table.c.name="Mike").value(age=35)
res=conn.execute(update_statement)

# DELETE
delete_statement=user_table.delete().where(user_table.c.age<)
res=conn.execute(delete_statement)

# SELECT
select_statement=user_table.select().where(user_table.c.age>25)
res=conn.execute(delete_statement)

for row in res:
	print(row)

# commit changes
conn.commit()
```

### DeclarativeBase in SQLAlchemy

Base class used to define mapped classes that correspond to database tables. It simplifies the process of creating and managing database schemas by allowing you to define both the table structure and the mapping to Python classes in a single declaration.

