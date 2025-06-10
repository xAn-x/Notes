Mongoose is a package that help us to connect with our mongo-db and provide easy to use function using which we can query our database for some info. Data Modelling is the most crucial part of db-management as this determines how u store the data and provides validation that help in pollution prevention.

```js
% npm i mongoose

import mongoose from "mongoose"

//IFI -> invoking function invokation soon after creating
(async ()=>{
	try{
		const connectionInstance= await mongoose.connect(`${DB_URL}/${DB_NAME}`);
		console.log(`\nMongoDb connected !! DB HOST
		:${connectionInstance.connection.host}`);
	}catch(error){
		console.log(`Failed to connected to DB:${error}`);
		throw error;
	}
})()


// Schema define the underlying structure of a field
userSchema=new mongoose.Schema({
	name : {
		type:String,
		unique:false,
		required:true
	}, 
	username : {
		type: String,
		unique:true,
		required:true,
		lowercase:true,
		// for search optimisization 
		index:true
	},
	"email" : {
		type: String,
		unique:true,
		required:true,
		validate : [validationFunc,"Msg if validation fails"]
	},
	password:{
		type:String,
		required:true,
		minlen:[8,"Password should be 8 char long atleast"],
		maxlen:[15,"You storing some Nuclear level shit or what, 15 chars max 😠"],
	},
	age:{
		type:Number,
		validate:[age=> age>=18, "This is a porno website..."]
	},
	gender:{
		// can choose from these fields only
		type: enum["M","F","O"],
		required: true
	},
	// Whenever want to ref to Something else create a seprate Model for it
	books: [{
		"name":{type:String,required:true},
		// If want to refrence to some other Model
		"author": Mongoose.Schema.Types.ObjectId,
		"ref": "User", // which model to refer
		"release_date": Date
	}]
},
{timestamps:true})
// timestamps auto stores createdAt and updatedAt field


// Model uses schema to create an instance of field
// This will allow us to modify the db content
let User=mongoose.models.users || mongoose.model("User",userSchema) 

const user=new User(userDetails)
// Note this is a async oprn moreover each db-related oprn will be async
user.save()
	.then((resp)=>console.log("User saved successfully"))
	.catch((err)=>console.log(`Some error occurred while saving the user.\nError:${error.message}`))
```

### `Functions:`

Important Mongoose functions categorized based on their operations:
### Adding Documents
1. `model.create(doc(s), options)` : Creates one or more documents in the collection.

### Updating Documents
2. `model.updateOne(conditions, update, options)` : Updates a single document in the collection based on the specified conditions, update, and options.
3. `model.updateMany(conditions, update, options)` : Updates multiple documents in the collection based on the specified conditions, update, and options.
4. `model.findOneAndUpdate(conditions, update, options)` : Finds a document and updates it based on the specified conditions, update, and options. It returns the update document.

### Removing Documents
5. `model.deleteOne(conditions)` : Deletes a single document from the collection based on the specified conditions.
6. `model.deleteMany(conditions)` : Deletes multiple documents from the collection based on the specified conditions.

### Querying Documents
7. `model.find(conditions, projection, options)` : Finds documents in the collection based on the specified conditions, projection, and options.
8. `model.findOne(conditions, projection, options)` : Finds a single document in the collection based on the specified conditions, projection, and options.
9. `model.findById(id, projection, options)` : Finds a document by its ID in the collection based on the specified projection and options.
10. `model.aggregate(pipeline)` : Performs aggregation operations on the collection using the specified pipeline.

### Other Operations
11. `model.countDocuments(conditions)` : Counts the number of documents in the collection based on the specified conditions.
12. `model.distinct(field, conditions)` : Finds the distinct values for a specified field in the collection based on the specified conditions.

These functions cover the essential operations for working with documents in MongoDB using Mongoose.

Sure, here's an example for each of the mentioned Mongoose functions with options and explanations:

### `Example Code Snips:`

### Adding Documents
```javascript
// Create a new document
const newDoc = { name: 'John Doe', age: 30 };
Model.create(newDoc, (err, result) => {
  if (err) {
    console.error(err);
  } else {
    console.log('Document created:', result);
  }
});
```

### Updating Documents
```javascript
// Update a document with specific conditions
const conditions = { _id: 'documentId' };
const update = { $set: { age: 31 } };
const options = { new: true }; // Return the updated document
Model.updateOne(conditions, update, options, (err, result) => {
  if (err) {
    console.error(err);
  } else {
    console.log('Document updated:', result);
  }
});
```

u can use atomic operators to manipulate different things.

| Operator     | Functionality                                                                                                           |
| ------------ | ----------------------------------------------------------------------------------------------------------------------- |
| `$set`       | Sets the value of a field in a document.                                                                                |
| `$unset`     | Removes a field from a document.                                                                                        |
| `$inc`       | Increments the value of a numeric field by a specified amount.                                                          |
| `$mul`       | Multiplies the value of a numeric field by a specified factor.                                                          |
| `$rename`    | Renames a field in a document.                                                                                          |
| `$min`       | Updates the value of a field to the specified value if it is lower than the current value.                              |
| `$max`       | Updates the value of a field to the specified value if it is higher than the current value.                             |
| `$addToSet`  | Adds elements to an array field only if they are not already present.                                                   |
| `$push`      | Adds elements to the end of an array field.                                                                             |
| `$pushAll`   | Adds multiple elements to an array field.                                                                               |
| `$pull`      | Removes elements from an array field that match a specified condition.                                                  |
| `$pullAll`   | Removes multiple elements from an array field.                                                                          |
| `$pop`       | Removes elements from either the beginning (`$pop: { field: -1 }`) or the end (`$pop: { field: 1 }`) of an array field. |
| `$elemMatch` | Selects documents where the array field contains at least one element that matches the specified criteria.              |
| `$slice`     | Limits the number of elements returned in an array field.                                                               |
| `$sort`      | Sorts elements in an array field.                                                                                       |
| `$not`       | Performs a logical NOT operation on the specified condition.                                                            |
| `$eq`        | Matches values that are equal to a specified value.                                                                     |
| `$ne`        | Matches values that are not equal to a specified value.                                                                 |
| `$gt`        | Matches values that are greater than a specified value.                                                                 |
| `$gte`       | Matches values that are greater than or equal to a specified value.                                                     |
| `$lt`        | Matches values that are less than a specified value.                                                                    |
| `$lte`       | Matches values that are less than or equal to a specified value.                                                        |
| `$in`        | Matches values that exist in a specified array.                                                                         |
| `$nin`       | Matches values that do not exist in a specified array.                                                                  |
| `$and`       | Performs a logical AND operation on an array of conditions.                                                             |
| `$or`        | Performs a logical OR operation on an array of conditions.                                                              |
| `$nor`       | Performs a logical NOR operation on an array of conditions.                                                             |
| `$exists`    | Matches documents where the specified field exists (or does not exist if `false`).                                      |
| `$type`      | Matches documents where the specified field has the specified BSON type.                                                |
| `$regex`     | Matches documents where the specified field matches a regular expression pattern.                                       |
| `$options`   | Specifies options for a `$regex` match, such as case insensitivity (`i`) or multiline matching (`m`).                   |

These operators are widely used in Mongoose for performing various operations such as updating documents, querying documents based on conditions, and manipulating array fields.
### Removing Documents
```javascript
// Delete a document with specific conditions
const conditions = { _id: 'documentId' };
Model.deleteOne(conditions, (err) => {
  if (err) {
    console.error(err);
  } else {
    console.log('Document deleted');
  }
});
```

### Querying Documents
```javascript
// Find documents with specific conditions and projection
const conditions = { age: { $gte: 30 } };
const projection = { name: 1, age: 1 }; // Include only 'name' and 'age' fields
const options = { limit: 10 }; // Limit to 10 results
Model.find(conditions, projection, options, (err, results) => {
  if (err) {
    console.error(err);
  } else {
    console.log('Found documents:', results);
  }
});
```

### Other Operations
```javascript
// Count documents with specific conditions
const conditions = { age: { $gte: 30 } };
Model.countDocuments(conditions, (err, count) => {
  if (err) {
    console.error(err);
  } else {
    console.log('Count:', count);
  }
});

// Find distinct values for a field with specific conditions
const field = 'name';
const conditions = { age: { $gte: 30 } };
Model.distinct(field, conditions, (err, values) => {
  if (err) {
    console.error(err);
  } else {
    console.log('Distinct values:', values);
  }
});
```

These examples demonstrate how to use each Mongoose function with options and provide a simple code snippet for each operation.

#Note: U can use these function with `async and await` or `promises` like this :
### Adding Documents
```javascript
const newDoc = { name: 'John Doe', age: 30 };
Model.create(newDoc)
  .then(result => {
    console.log('Document created:', result);
  })
  .catch(err => {
    console.error(err);
  });
```

### Updating Documents
```javascript
// Update a document with specific conditions 
const conditions = { _id: 'documentId' };
const update = { $set: { age: 31 } };
const options = { new: true }; // Return the updated document
Model.updateOne(conditions, update, options)
  .then(result => {
    console.log('Document updated:', result);
  })
  .catch(err => {
    console.error(err);
  });
```

### Removing Documents
```javascript
// Delete a document with specific conditions
const conditions = { _id: 'documentId' };
Model.deleteOne(conditions)
  .then(() => {
    console.log('Document deleted');
  })
  .catch(err => {
    console.error(err);
  });
```

### Querying Documents
```javascript
// Find documents with specific conditions and projection 
const conditions = { age: { $gte: 30 } };
const projection = { name: 1, age: 1 }; // Include only 'name' and 'age' fields
const options = { limit: 10 }; // Limit to 10 results
Model.find(conditions, projection, options)
  .then(results => {
    console.log('Found documents:', results);
  })
  .catch(err => {
    console.error(err);
  });
```

### Other Operations
```javascript
// Count documents with specific conditions
const conditions = { age: { $gte: 30 } };
Model.countDocuments(conditions)
  .then(count => {
    console.log('Count:', count);
  })
  .catch(err => {
    console.error(err);
  });

// Find distinct values for a field with specific conditions
const field = 'name';
const conditions = { age: { $gte: 30 } };
Model.distinct(field, conditions)
  .then(values => {
    console.log('Distinct values:', values);
  })
  .catch(err => {
    console.error(err);
  });
```

#NOTE: U can create your own plugins and methods for a schema that will help u do certain thing. [Check this](https://www.youtube.com/watch?v=eWnZVUXMq8k&t=46s)

### `Middleware:`

Middleware (also called pre and post _hooks_) are functions which are passed control during execution of asynchronous functions. Middleware is specified on the schema level and is useful for writing [plugins](https://mongoosejs.com/docs/plugins.html).

Mongoose has 4 types of middleware: 
 1. `Document middleware:` A document is an instance of a `Model` class. In document middleware functions, ==`this` refers to the document==. To access the model, use `this.constructor`.
	- [validate](https://mongoosejs.com/docs/api/document.html#document_Document-validate)
	- [save](https://mongoosejs.com/docs/api/model.html#model_Model-save)
	- [remove](https://mongoosejs.com/docs/api/model.html#model_Model-remove)
	- [updateOne](https://mongoosejs.com/docs/api/document.html#document_Document-updateOne)
	- [deleteOne](https://mongoosejs.com/docs/api/model.html#model_Model-deleteOne)
	- [init](https://mongoosejs.com/docs/api/document.html#document_Document-init) (note: init hooks are [synchronous](https://mongoosejs.com/docs/middleware.html#synchronous))
	
2. `Model middleware:` Don't confuse model middleware and document middleware: model middleware hooks into _static_ functions on a `Model` class, document middleware hooks into _methods_ on a `Model` class. In model middleware functions, `this` refers to the model.
	- [bulkWrite](https://mongoosejs.com/docs/api/model.html#model_Model-bulkWrite)
	- [createCollection](https://mongoosejs.com/docs/api/model.html#model_Model-createCollection)
	- [insertMany](https://mongoosejs.com/docs/api/model.html#model_Model-insertMany)

3. `Aggregate middleware:`  for `MyModel.aggregate()`. Aggregate middleware executes when you call `exec()` on an aggregate object. In aggregate middleware, `this` refers to the [aggregation object](https://mongoosejs.com/docs/api/model.html#model_Model-aggregate).
	- [aggregate](https://mongoosejs.com/docs/api/model.html#model_Model-aggregate)

4. `Query middleware:` Query middleware executes when you call `exec()` or `then()` on a Query object, or `await` on a Query object. In query middleware functions, `this` refers to the query.
	- [count](https://mongoosejs.com/docs/api/query.html#query_Query-count)
	- [countDocuments](https://mongoosejs.com/docs/api/query.html#query_Query-countDocuments)
	- [deleteMany](https://mongoosejs.com/docs/api/query.html#query_Query-deleteMany)
	- [deleteOne](https://mongoosejs.com/docs/api/query.html#query_Query-deleteOne)
	- [estimatedDocumentCount](https://mongoosejs.com/docs/api/query.html#query_Query-estimatedDocumentCount)
	- [find](https://mongoosejs.com/docs/api/query.html#query_Query-find)
	- [findOne](https://mongoosejs.com/docs/api/query.html#query_Query-findOne)
	- [findOneAndDelete](https://mongoosejs.com/docs/api/query.html#query_Query-findOneAndDelete)
	- [findOneAndReplace](https://mongoosejs.com/docs/api/query.html#query_Query-findOneAndReplace)
	- [findOneAndUpdate](https://mongoosejs.com/docs/api/query.html#query_Query-findOneAndUpdate)
	- [remove](https://mongoosejs.com/docs/api/model.html#model_Model-remove)
	- [replaceOne](https://mongoosejs.com/docs/api/query.html#query_Query-replaceOne)
	- [update](https://mongoosejs.com/docs/api/query.html#query_Query-update)
	- [updateOne](https://mongoosejs.com/docs/api/query.html#query_Query-updateOne)
	- [updateMany](https://mongoosejs.com/docs/api/query.html#query_Query-updateMany)
	- [validate](https://mongoosejs.com/docs/validation.html#update-validators)

```js
import mongoose from "mongoose"
import bcrypt from "bcrypt"

let userSchema=new mongoose.Schema({
	user:{
		type:String,
		required:true,
		unique:true,
		trim:true
	},
	password:{
		type:String,
		required:true,
		minlen:[8,"Password should be atleat 8 chars long"],
		maxlen:[15,"The max length for password can be 15"]	
	},
	email:{
		type:String,
		required:true,
		lowercase:true,
		trim:true,
	}
},{timestamps:true})


// Adding a middleware to schema to store password in a encrypted manner

//Note don't use arrow function here as we need to use 'this'
userSchema.pre("save",async function (next)=>{
	// if password field is modified then only run this,else will be run if user make any change like changing email,profile-pic etc..
	if(this.isModified("password")){
		this.password=bcrypt.hash(this.password,10)
	}
	next()
})

// As like Middleware ,in mongoose we can also add our own custom methods
userSchema.methods.isPasswordCorrect = async function(password){
	return await (password==bcrypt.compare(password,this.password))
}

export default const User=mongoose.model("User",userSchema)
``` 

### `Monog-DB Aggregation Pipeline:`

A powerful framework for processing and transforming documents in a collection. It allows you to perform complex data manipulations, transformations, and computations on the data stored in your MongoDB database. The aggregation pipeline consists of stages, where each stage performs a specific operation on the documents as they pass through the pipeline.

```js
db.orders.aggregate( [

   // Stage 1: Filter pizza order documents by pizza size
   {
      $match: { size: "medium" }
   },

   // Stage 2: Group remaining documents by pizza name and calculate total quantity
   {
      $group: { 
      _id: "$name", 
      totalQuantity: {
	       $sum: "$quantity" 
		 }
    }
    
}] )
```

==Check-Out==:[Aggregation Pipeline](https://www.youtube.com/watch?v=vx1C8EyTa7Y)

`Some usefull operations in aggregation pipeline`

| Stage              | Description                                                                                                                                           | Example                                                                                                                                                |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| $match             | Filters documents to pass only those that match specified conditions.                                                                                 | `{ $match: { status: "active" } }`                                                                                                                     |
| $group             | Groups documents by specified fields and performs aggregate functions (e.g., $sum, $avg, $max) within each group.                                     | `{ $group: { _id: "$category", totalItems: { $sum: 1 }, avgPrice: { $avg: "$price" } } }`                                                              |
| $project           | Reshapes documents by including, excluding, or renaming fields.                                                                                       | `{ $project: { name: 1, age: 1, _id: 0 } }`                                                                                                            |
| $sort              | Sorts documents based on specified fields and order (ascending or descending).                                                                        | `{ $sort: { age: 1 } }`                                                                                                                                |
| $limit             | Limits the number of documents passed to the next stage in the pipeline.                                                                              | `{ $limit: 10 }`                                                                                                                                       |
| $skip              | Skips a specified number of documents and passes the remaining documents to the next stage.                                                           | `{ $skip: 5 }`                                                                                                                                         |
| $unwind            | Deconstructs an array field into separate documents for each element in the array.                                                                    | `{ $unwind: "$tags" }`                                                                                                                                 |
| $lookup            | Performs a left outer join to retrieve documents from another collection.                                                                             | `{ $lookup: { from: "orders", localField: "orderId", foreignField: "_id", as: "orderDetails" } }`                                                      |
| $addFields         | Adds new fields to documents based on specified expressions.                                                                                          | `{ $addFields: { totalAmount: { $sum: "$items.price" } } }`                                                                                            |
| $push              | Add certain field-value in an array and return that                                                                                                   | `{ $group: { _id:"geneder",names:{$push:"$name"}}`<br>                                                                                                 |
| $replaceRoot       | Replaces the document's root with a new document.                                                                                                     | `{ $replaceRoot: { newRoot: { fullName: { $concat: ["$firstName", " ", "$lastName"] }, age: "$age" } } }`                                              |
| $lookup (pipeline) | Performs a left outer join with an additional pipeline on the joined collection. Useful for advanced join and filtering operations.                   | `{ $lookup: { from: "orders", let: { orderId: "$orderId" }, pipeline: [{ $match: { $expr: { $eq: ["$_id", "$$orderId"] } } }], as: "orderDetails" } }` |
| $redact            | Applies a redaction policy to documents, hiding certain fields based on specified conditions.                                                         | `{ $redact: { $cond: { if: { $eq: ["$level", 1] }, then: "$$DESCEND", else: "$$PRUNE" } } }`                                                           |
| $sample            | Randomly samples a specified number of documents from the collection.                                                                                 | `{ $sample: { size: 5 } }`                                                                                                                             |
| $out               | Writes the output of the aggregation pipeline to a new collection.                                                                                    | `{ $out: "newCollection" }`                                                                                                                            |
| $facet             | Allows for multiple independent pipelines to be executed within a single aggregation stage. Useful for performing multiple aggregations in one query. | `{ $facet: { averagePrice: [{ $group: { _id: "$category", avgPrice: { $avg: "$price" } } }], totalItems: [{ $count: "count" }] } }`                    |
