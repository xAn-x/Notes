
## 1. Asynchronous JavaScript:

JavaScript is a synchronous, single-threaded language. This means that JavaScript code is executed one line at a time, in the order it is written. Each line of code must wait for the previous line to finish executing before it can start.

![[Pasted image 20240323092959.png]]

Blocking vs Non-Blocking
![[Pasted image 20240323093249.png]]

JavaScript Architecture
![[Pasted image 20240323113247.png]]


**Asynchronous Programming in JavaScript**

Asynchronous programming is a programming paradigm that allows tasks ==to be executed concurrently without blocking the main thread==. This is achieved by using callbacks, promises, or async/await syntax.

**Callbacks**

Callbacks are functions that are passed as arguments to other functions. When the asynchronous operation is complete, the callback function is called with the result.

```js
// Function that perfo rms an asynchronous operation
function doSomethingAsync(callback) {
  setTimeout(() => {
    callback("result");
  }, 1000);
}

// Call the asynchronous function and pass a callback
doSomethingAsync((result) => {
  console.log(result); // Output: "result"
});
```

**Promises**

Promises are objects that represent the eventual result of an asynchronous operation. They have three states:

* Pending: The operation is in progress.
* Fulfilled: The operation completed successfully.
* Rejected: The operation failed.

```js
// Function that returns a promise
function doSomethingAsync() {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      resolve("result");
    }, 1000);
  });
}

// Call the asynchronous function and use the promise
doSomethingAsync()
  .then((result) => {
    console.log(result); // Output: "result"
  })
  .catch((error) => {
    console.error(error);
  });
```

**Async/Await**

Async/await syntax is a more concise way to write asynchronous code. It allows you to use the `await` keyword to ==pause the execution of the function until the asynchronous operation is complete==.

```js
// Function that uses async/await
async function doSomethingAsync() {
  const result = await new Promise((resolve, reject) => {
    setTimeout(() => {
      resolve("result");
    }, 1000);
  });
  return result;
}

// Call the asynchronous function using async/await
async function main() {
  const result = await doSomethingAsync();
  console.log(result); // Output: "result"
}

main();
```

**Benefits of Asynchronous Programming**

* **Improved performance:** Asynchronous programming allows tasks to be executed concurrently, which can improve the performance of applications by reducing blocking operations.
* **Scalability:** Asynchronous programming makes it easier to scale applications to handle more concurrent requests.
* **Responsiveness:** Asynchronous programming ensures that the user interface remains responsive even when performing long-running tasks.


## 2. OOPs in JavaScript:

**Object-Oriented Programming (OOP) in JavaScript**

OOP is a programming paradigm that uses "objects" to design applications and computer programs. =="Objects" are data structures consisting of data fields and methods together with their interactions.== This makes it easier to create complex programs that are easier to maintain and reuse. ===JavaScript supports OOP, although it is a prototype-based language rather than a class-based language like Java or C++.===

**Key Concepts of OOP in JavaScript**

* **Objects:** Objects are the fundamental building blocks of OOP. They encapsulate data and behavior.
* **Classes:** Classes are blueprints for creating objects. They define the properties and methods that objects of that class will have.
* **Inheritance:** Inheritance allows new classes (child classes) to be created from existing classes (parent classes). Child classes inherit the properties and methods of their parent classes.
* **Polymorphism:** Polymorphism allows objects of different classes to respond to the same message in different ways.

**Creating Objects and Classes in JavaScript**

**Using Object Literals:**

```js
const person = {
  name: "John Doe",
  age: 30,
  greet: function() {
    console.log(`Hello, my name is ${this.name}`);
  }
};
```

**Using Functions:**

```js
function User(name,age){
	return this; // returning the instance ,will auto return 'this' if u don't
}

const user1=new User("Jhon Doe",27)
// Note: If u don't use 'new' keyword it will override the properties whenever u create a a instance/object using it.
```

**Using Classes (ES6):**

```js
// syntactical sugar to create objects
class Person {
  constructor(name, age) {
    this.name = name;
    this.age = age;
  }
// no need to write 'function' keyword
  greet() {
    console.log(`Hello, my name is ${this.name}`);
  }
}

const person = new Person("John Doe", 30);
// Note: If u don't use 'new' keyword it will override the properties whenever u create a a instance/object using it.
```

**Inheritance in JavaScript**

Inheritance is achieved through the `extends` keyword.

```js
class Employee extends Person {
  constructor(name, age, salary) {
    super(name, age); // Call the parent class constructor
    this.salary = salary;
  }

  getSalary() {
    return this.salary;
  }
}

const employee = new Employee("John Doe", 30, 50000);
```

**Benefits of OOP in JavaScript**

* **Modularity:** OOP helps in organizing code into reusable and maintainable modules.
* **Encapsulation:** OOP allows data and methods to be bundled together, hiding implementation details.
* **Reusability:** OOP promotes code reuse through inheritance and polymorphism.
* **Extensibility:** OOP makes it easy to extend and modify applications by adding new classes and methods.


## 3. Prototype in JavaScript:

**Prototype** is a ===special property of JavaScript objects that allows you to add new properties and methods to existing objects.===

Every object in JavaScript has a prototype, which is itself an object. The prototype ===contains properties and methods that are inherited by the object.===

You can access the prototype of an object using the `Object.getPrototypeOf()` method. For example:

```js
const obj = {
  name: "John Doe",
  age: 30,
};

const prototype = Object.getPrototypeOf(obj);

console.log(prototype); // { constructor: ƒ, __defineGetter__: ƒ, __defineSetter__: ƒ, ... }
```

As you can see, the prototype of the `obj` object is itself an object that contains a number of properties and methods.

You can add new properties and methods to an object's prototype using the `Object.defineProperty()` method. For example:

```js
Object.defineProperty(prototype, "greet", {
  value: function() {
    console.log(`Hello, my name is ${this.name} and I am ${this.age} years old.`);
  },
  enumerable: true,
});

obj.greet(); // Hello, my name is John Doe and I am 30 years old.

// ==== or ====
Object.assign(prototype,greet)

// ==== or ====

Object.protototype.greet=function(){console.log("hello")}
```

As you can see, we have added a new method called `greet` to the prototype of the `obj` object. This method is now available to all objects that inherit from the `obj` object's prototype.


U can create your own object and then can introduce new properties and methods on go.

```js
function User(name,age){
	this.name=name
	this.age=age
	
	return this //this store the current-context
}

User.protorype.setSalary=function (salary){
	this.salary=salary
}

let obj1=new User("Deepanshu",21) 
// Note: without new keyword the prototype properties and method u have define ,can't be access there value will be undefined until new is used
```

===Note:=== without new keyword the prototype properties and method u have define ,can't be access there value will be undefined until new is used

**Prototypes** are a powerful tool that can be used to extend the functionality of JavaScript objects. They allow you to add new properties and methods to existing objects without having to modify the original object.

**Prototype** is a core concept in JavaScript and it is essential for understanding how JavaScript objects work.

## 4. new keyword:

The `new` keyword in JavaScript is used to create a new instance of a class or a constructor function. When used with a class, the `new` keyword creates a new object that inherits the properties and methods of the class. When used with a constructor function, the `new` keyword creates a new object that inherits the properties and methods of the constructor function's prototype.

**Importance of the `new` keyword for prototyping in JavaScript**

The `new` keyword is essential for prototyping in JavaScript because it allows you to create new objects that inherit from existing objects. This is a powerful feature that allows you to create complex and reusable code.

For example, the following code creates a `Person` class and then uses the `new` keyword to create a new `Person` object:

```js
class Person {
  constructor(name, age) {
    this.name = name;
    this.age = age;
  }

  greet() {
    console.log(`Hello, my name is ${this.name} and I am ${this.age} years old.`);
  }
}

const person = new Person("John Doe", 30);

person.greet(); // Hello, my name is John Doe and I am 30 years old.
```

As you can see, the `person` object inherits the `name`, `age`, and `greet` methods from the `Person` class. This is because the `new` keyword creates a new object that inherits from the prototype of the class.

**Relevance of the `new` keyword for prototyping in JavaScript**

The `new` keyword is essential for prototyping in JavaScript because it allows you to:

* Create new objects that inherit from existing objects
* Reuse code by creating new objects that inherit from existing prototypes
* Create complex and extensible code

**Conclusion**

The `new` keyword is a powerful tool that is essential for prototyping in JavaScript. It allows you to create new objects that inherit from existing objects, reuse code, and create complex and extensible code.

**How `new` works behind the scene: **

```js
class Person {
  constructor(name, age) {
    this.name = name;
    this.age = age;
  }

  greet() {
    console.log(`Hello, my name is ${this.name} and I am ${this.age} years old.`);
  }
}

const person = new Person("John Doe", 30);

person.greet(); // Hello, my name is John Doe and I am 30 years old.
```

Behind the scenes, the following steps happen when the `new` keyword is used:

1. A new object is created.
2. The `Person` constructor function is called with the `this` keyword bound to the new object. The `Person` constructor function sets the `name` and `age` properties of the new object to "John Doe" and 30, respectively.
3. The `Person` constructor function does not return an object, so the new object is not replaced.
4. The new object is returned and assigned to the `person` variable.

	The `person` object now has the `name`, `age`, and `greet` methods. This is because the `person` object inherits from the `Person` class prototype.
	
	**Conclusion**
	
	The `new` keyword is a powerful tool that allows you to create new objects that inherit from existing classes or constructor functions. When the `new` keyword is used, a new object is created, the constructor function is called with the `this` keyword bound to the new object, and the new object is returned.
 