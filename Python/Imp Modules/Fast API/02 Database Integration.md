This example demonstrates CRUD (Create, Read, Update, Delete) operations using FastAPI with a SQL database (using SQLAlchemy as an ORM). 
## Setup 1. **Install required packages:**

```bash 

pip install fastapi uvicorn sqlalchemy databases[sqlite] pydantic
```


```shell

touch mydatabase.db  # Creates an empty SQLite database file
[]()```


```python
from typing import List, Union

import databases
import sqlalchemy
from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel

# Database configuration (same as before)
DATABASE_URL = "sqlite:///mydatabase.db"
database = databases.Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()

# --- Database Model ---
class ItemModel(BaseModel):
    id: int
    name: str
    description: Union[str, None] = None
    price: float
    on_offer: bool = False

    class Config:  # To allow ORM mode
        orm_mode = True

items = sqlalchemy.Table(  # Using ItemModel for column definitions
    __tablename__= "items",
    __natadata__= metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("name", sqlalchemy.String),
    sqlalchemy.Column("description", sqlalchemy.String, nullable=True),
    sqlalchemy.Column("price", sqlalchemy.Float),
    sqlalchemy.Column("on_offer", sqlalchemy.Boolean, default=False),
)

engine = sqlalchemy.create_engine(DATABASE_URL)
metadata.create_all(engine)


# --- CRUD Operations Class ---
class ItemCRUD:
    def __init__(self, db: databases.Database):
        self.db = db

    async def create(self, item: ItemModel) -> ItemModel:
        query = items.insert().values(**item.dict())
        last_record_id = await self.db.execute(query)
        return await self.get(last_record_id) # Get the complete created item

    async def get(self, item_id: int) -> Union[ItemModel, None]:
        query = items.select().where(items.c.id == item_id)
        item = await self.db.fetch_one(query)
        if item:
            return ItemModel(**item)  # Convert to Pydantic model
        return None

    async def get_all(self) -> List[ItemModel]:
        query = items.select()
        all_items = await self.db.fetch_all(query)
        return [ItemModel(**item) for item in all_items]

    async def update(self, item_id: int, item: ItemModel) -> Union[ItemModel, None]:
        query = (
            items.update()
            .where(items.c.id == item_id)
            .values(**item.dict())
        )
        await self.db.execute(query)
        return await self.get(item_id) # Get the updated item

    async def delete(self, item_id: int) -> None:
        query = items.delete().where(items.c.id == item_id)
        await self.db.execute(query)


# --- FastAPI App ---

app = FastAPI()

# Dependency to get ItemCRUD instance
def get_item_crud(db: databases.Database = Depends(database)):
    return ItemCRUD(db)


@app.on_event("startup")  # (Database connection same as before)
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()


# --- API Endpoints using ItemCRUD ---

# ... (The rest of the API endpoints are very similar, just using item_crud)

@app.post("/items/", response_model=ItemModel, status_code=status.HTTP_201_CREATED)
async def create_item(item: ItemModel, item_crud: ItemCRUD = Depends(get_item_crud)):
    return await item_crud.create(item)

@app.get("/items/", response_model=List[ItemModel])
async def read_items(item_crud: ItemCRUD = Depends(get_item_crud)):
    return await item_crud.get_all()

# ... (Implement other endpoints similarly)

```


### Key improvements:

* **Separate Model and CRUD Class:** Improves code organization and maintainability.
* **ORM Mode:**  Using `orm_mode = True` in the Pydantic model simplifies conversion between database results and Pydantic objects.
* **Dependency Injection for CRUD:**  The `get_item_crud` dependency makes the `ItemCRUD` instance available to the API endpoints.
* **Return Created/Updated Item:** The `create` and `update` methods now return the created/updated item, making the API more user-friendly.
* **Consistent Error Handling:**  You can add more robust error handling within the `ItemCRUD` methods (e.g., checking for database errors, raising custom exceptions).