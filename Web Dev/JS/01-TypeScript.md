Ts a superset of JavaScript that adds optional static typing. The compiler checks your code for type errors before runtime, helping you catch bugs early. Ts compiles down to simple Js, so it runs runs anywhere Js runs. Ts is all about _type-safety.

```shell
npm install -g typescript
tsc --version
```

```ts
// define a var -> const var:dtype = value
const name:string="Deepanshu"
const age:number=17

name=7 // this will warn u, but will compile


// define a func -> function funcName(var:dtype):returnType {--def--}
function greetUser(name:string):string{
	return `Hello,${name}!!`
}

function validateUser(username:string,password:string):boolean | never{
	if(db.findOne({username,password})==false) 
		// if func throwing an error use 'never' as return type
		throw new Error("user not found")
	return true
}


// creating custom dtypes
type User={
	readonly _id:string; // 'readonly' if something can't be override
	name:string;
	age:number;
	isActive:boolean;
	published_books?:string[]; // '?' if something is opt
}
// '$' to merge 2 types -> type t3= t1 & t2 & {}
```

### Arrays vs Tuples:

The key difference between arrays and tuples in TypeScript lies in their flexibility regarding the number and types of elements they can hold.

**Arrays:**

* **Flexibility:** Arrays can hold any number of elements, and all elements can be of the same type or different types (though type safety is reduced if you mix types).
* **Type Definition:**  Defined using `type[]` (e.g., `number[]`, `string[]`, or `any[]`).  If you don't specify a type, the compiler will infer a type based on the elements you assign.

**Tuples:**

* **Fixed Length and Types:** Tuples have a fixed number of elements, and each element's type is explicitly specified.
* _Order matters_.
* **Type Definition:** Defined using `[type1, type2, ...]` (e.g., `[string, number, boolean]`).  The order of types matters.


**Code Snippet:**

```ts
// Arrays
let numbers: number[] = [1, 2, 3, 4, 5];
let mixed: any[] = ["hello", 1, true]; // Less type-safe

// Tuples
let user: [string, number] = ["John Doe", 12345]; // Fixed length and types

//user = ["Jane Doe", 67890, true]; //Error: Type '[string, number, boolean]' is not assignable to type '[string, number]'.
let point: [number, number] = [10, 20];

// Accessing elements (same for both arrays and tuples)
console.log(numbers[0]); // Output: 1
console.log(user[0]);    // Output: John Doe
console.log(point[1]);   // Output: 20


// Demonstrating the difference in type safety
// array methods can fuck up the tuples,so use carefully
// we can push any value to tuple using them and upto any-point
numbers.push("oops!"); // This compiles, but might lead to runtime errors if not handled

//user.push(true); // This will cause a compile-time error because tuples have a fixed length
```


### Enums

Enums in TypeScript provide a way to define a set of named constants.  They improve code readability and maintainability by giving descriptive names to numeric values.  The compiler assigns a numeric value to each member (starting from 0 by default, or you can assign specific values).

**Example:**

```typescript
enum Color {
  Red,  // 0
  Green, // 1
  Blue   // 2
}

let myColor: Color = Color.Green; // myColor is now 1
console.log(myColor); // Output: 1
console.log(Color.Red); // Output: 0

enum Status {
  Pending = 1,
  Approved, // 2
  Rejected = 100
}

console.log(Status.Approved) //Output:2
console.log(Status.Rejected) //Output:100
```


### Interface in Ts

In TypeScript, interfaces define the shape of an object.  They specify the names and types of the properties an object should have.  Interfaces don't provide implementation details; they only define a contract that objects must adhere to.

**Why use Interfaces?**

1. **Code Readability and Maintainability:** Interfaces make code easier to understand by clearly defining the structure of objects.  They act as documentation and improve code organization.

2. **Type Safety:** Interfaces enforce type checking at compile time.  If you try to create an object that doesn't match the interface definition, the compiler will throw an error.  This helps prevent runtime errors.

3. **Code Reusability:** Interfaces can be reused across multiple parts of your codebase.  If you have multiple objects that share a similar structure, you can define a single interface and have all objects implement it.

4. **Polymorphism:** Interfaces enable polymorphism, allowing you to work with objects of different types in a consistent way.


**Examples:**

```typescript
// Defining an interface
interface Person {
  firstName: string;
  lastName: string;
  age?: number; // age is optional
}

// u can add new properties just by reopening the interface
interface Person{
	email?:string
}

// Creating objects that implement the interface
let person1: Person = {
  firstName: "John",
  lastName: "Doe",
  age: 30
};

let person2: Person = {
  firstName: "Jane",
  lastName: "Doe",
  emai:"Jane.Doe@gmail.com"
};



// One can extend an interface this way also
interface Shape {
  getArea(): number;
}

// interface -> extends
interface Poly extends Shape{
	num_edges:number
}

// class -> implements
class Circle implements Shape {
  radius: number;
  constructor(radius: number) { this.radius = radius; }
  getArea(): number {
    return Math.PI * this.radius * this.radius;
  }
}

class Rectangle implements Shape {
    width:number;
    height:number;
    constructor(width:number, height:number){
        this.width = width;
        this.height = height;
    }
    getArea(): number {
        return this.width * this.height;
    }
}

let circle = new Circle(5);
let rectangle = new Rectangle(4,5);
console.log(circle.getArea()); //Output: 78.53981633974483
console.log(rectangle.getArea()); //Output: 20

```
#### Type vs Interface
| Feature                         | _type_                                                                                                                | _interface_                                                                                                   |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **Declaration Syntax**          | `type Name = { ... }`                                                                                                 | `interface Name { ... }`                                                                                      |
| **Extending**                   | Can use intersections (`&`) to extend other types.                                                                    | Can extend other interfaces using `extends`.                                                                  |
| **Merging Declarations**        | No declaration merging. If you try to declare a `type` with the same name multiple times, it will result in an error. | Supports declaration merging. Multiple `interface` declarations with the same name are automatically merged.  |
| **Usage**                       | Can be used to create unions, intersections, primitive types, and even more complex structures.                       | Primarily used to define the shape of objects or class structures.                                            |
| **Support for Primitive Types** | Supports defining primitive types like `string`, `number`, `boolean`, etc.                                            | Does not directly support primitive types, but can be used to describe object shapes and method signatures.   |
| **Complexity**                  | Can represent more complex types (unions, intersections, mapped types, etc.).                                         | Primarily focused on object shapes and method signatures.                                                     |
| **Immutability**                | Types are immutable once defined. Cannot be changed.                                                                  | Interfaces are mutable and can be extended or augmented.                                                      |
| **Instantiation**               | Cannot instantiate `type` directly (no `new` keyword).                                                                | Can be used with `class` or `interface` for OOP designs.                                                      |
| **Function Signatures**         | Can define function signatures, but also unions, intersections, and mapped types.                                     | Can define function signatures, but is more specific to object shapes.                                        |
| **Compatibility**               | Types can be used in places where interfaces can, but not all interface features are available in types.              | Interfaces are often preferred for defining the shape of objects and are compatible with class-based systems. |
| **Key Remapping**               | Supports advanced features like key remapping in mapped types.                                                        | Does not support key remapping.                                                                               |

#### Classes

```ts
class User{
	// declare vars and types in advance
	name:string
	email:string | string []
	// we have access modifiers also {default:public}
	private password: string
	
	// one can directly specify access modiefier here
	constructor(name:string,email:string|string[],private password:string){
		this.name=name
		this.email=email
		this.password=password
	}

	// define function and stuff
	// u can specify access modifier for methods too
	get_property(property:string):string{
		if(!this.hasOwnProperty(property)) return ''
		return this[property]
	}

	// getters and setters in Ts: using `get` and `set` keyword
	get get_password():string{
		return this.password
	}

	// Note: setter,don't have return type
	set password(old_password:string,new_password:string){
		if (old_password!==this.password)
			throw new Error("Invalid,password!!")
		this.password=new_password 
	}
}
// creating a instance of class
const user=new User(name="Deepanshu",email="d@d.com")
```

### Abstract Class

Abstract classes in TypeScript are classes that cannot be instantiated directly. They serve as blueprints for other classes (subclasses) to inherit from.  They can contain both abstract methods (methods without implementation) and concrete methods (methods with implementation).  Subclasses *must* provide implementations for all abstract methods inherited from the abstract class.

**One Point:** Abstract classes enforce a contract on subclasses, ensuring they implement specific methods, promoting code consistency and maintainability.


**Example:**

```typescript
// Defining an abstract class
abstract class Shape {
  abstract getArea(): number; // Abstract method – no implementation

  // Concrete method – has implementation
  printArea():void {
    console.log(`Area: ${this.getArea()}`);
  }
}

// Subclass inheriting from the abstract class
class Circle extends Shape {
  radius: number;
  constructor(radius: number) {
    super();
    this.radius = radius;
  }
  getArea(): number {
    return Math.PI * this.radius * this.radius;
  }
}

//Another subclass
class Rectangle extends Shape {
    width: number;
    height: number;
    constructor(width: number, height: number) {
        super();
        this.width = width;
        this.height = height;
    }
    getArea(): number {
        return this.width * this.height;
    }
}

let circle = new Circle(5);
let rectangle = new Rectangle(4, 5);

circle.printArea(); // Output: Area: 78.53981633974483
rectangle.printArea();// Output: Area: 20

//let shape = new Shape(); // Error: Cannot create an instance of an abstract class.
```

### Generics in TypeScript

Generics enhance code reusability and maintain type safety by allowing components to work with various types without sacrificing type checking.


**Example:**

```typescript
// Generic function
function identity<T>(arg: T): T {
  return arg;
}

let myString: string = identity<string>("hello"); //Explicit type argument
let myNumber: number = identity(123); // Type argument inferred by compiler
let myArray: number[] = identity([1,2,3]); // Type argument inferred by compiler


// Generic interface
interface Pair<K, V> {
  key: K;
  value: V;
}

let pair1: Pair<string, number> = { key: "age", value: 30 };
let pair2: Pair<number, string> = { key: 1, value: "hello" };


// Generic class
class KeyValuePair<K, V> {
  key: K;
  value: V;
  constructor(key: K, value: V) {
    this.key = key;
    this.value = value;
  }
}

let pair3 = new KeyValuePair<string, number>("name", 25);
let pair4 = new KeyValuePair<number, string>(1, "one");


// Generic constraints
interface HasLength {
    length: number;
}

function loggingIdentity<T extends HasLength>(arg: T): T {
    console.log(arg.length);
    return arg;
}

loggingIdentity("hello")
loggingIdentity([1,2,3])
//loggingIdentity(123); // Error: Argument of type 'number' is not assignable to parameter of type 'HasLength'.

```

### Type Narrowing in TypeScript

Type narrowing is a mechanism in TypeScript that allows the compiler to reduce the type of a variable to a more specific subtype based on runtime checks. This improves type safety and allows you to access properties or methods that are only available on the narrowed type without causing compiler errors.

**One Point:** Type narrowing refines type information based on conditional checks, enabling safer access to type-specific members and preventing runtime errors.


TypeScript employs several techniques for type narrowing:

* **`typeof` operator:** Checks the type of a variable at runtime.  Useful for distinguishing between `string`, `number`, `boolean`, `symbol`, `bigint`, `undefined`, `object`, and `function`.

* **`instanceof` operator:** Checks if an object is an instance of a particular class.

* **`in` operator:** Checks if a property exists in an object.

* **Type predicates:** Custom functions that narrow types based on their return values.  These functions are marked with a `=> boolean` return type and have a type assertion in their return statement.

* **Control flow analysis:** The compiler analyzes `if`, `else if`, `else`, `switch`, and `for`/`while` statements to narrow types based on the conditions within the statements.


**Examples:**

```typescript
// typeof operator
function processValue(value: string | number): string {
  if (typeof value === 'string') {
    return value.toUpperCase(); // value is narrowed to string within this block
  } else {
    return value.toString(); // value is narrowed to number within this block
  }
}


// instanceof operator
class Animal {}
class Dog extends Animal {}

function greetAnimal(animal: Animal): string {
  if (animal instanceof Dog) {
    return `Woof! This is a dog.`; // animal is narrowed to Dog within this block
  } else {
    return `Generic animal greeting.`;
  }
}


// in operator
interface Person {
  name: string;
  age?: number;
}

function printPersonDetails(person: Person) {
  if ('age' in person) {
    console.log(`Name: ${person.name}, Age: ${person.age}`); // person is narrowed; age is guaranteed to exist
  } else {
    console.log(`Name: ${person.name}`); // person still has name, but age is undefined
  }
}


// Type predicate
function isString(value: any): value is string {
  return typeof value === 'string';
}

let value: string | number = "hello";
if (isString(value)) {
  console.log(value.toUpperCase()); // value is narrowed to string
}


// Control flow analysis
function checkValue(value: string | number | boolean): void {
    if (typeof value === 'string') {
        console.log("String")
    } else if (typeof value === 'number') {
        console.log("Number")
    } else {
        console.log("Boolean")
    }
}
```