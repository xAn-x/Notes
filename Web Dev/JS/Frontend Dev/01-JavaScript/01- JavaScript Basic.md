1. `Dates in JS`

```js
let date = new Date(); // creates a Date object representing the current date and time

console.log(date.toString()); // converts the Date object to a string in a human-readable format

// Important methods and attributes of the Date object:
console.log(date.getFullYear()); // returns the year (four digits)
console.log(date.getMonth()); // returns the month (0-11, where 0 is January)
console.log(date.getDate()); // returns the day of the month (1-31)
console.log(date.getDay()); // returns the day of the week (0-6, where 0 is Sunday)
console.log(date.getHours()); // returns the hour (0-23)
console.log(date.getMinutes()); // returns the minutes (0-59)
console.log(date.getSeconds()); // returns the seconds (0-59)
console.log(date.getTime()); // returns the number of milliseconds since January 1, 1970
console.log(date.getTimezoneOffset()); // returns the difference in minutes between the local time zone and UTC

console.log(date.toDateString()); // returns date in DateString format
console.log(date.toISOString()); // returns the date and time in ISO 8601 format (UTC)
console.log(date.toLocaleString()); // very-imp


let specefic_date1=new Date(year,month,date)
let specefic_date2=new Date("yyyy-mm-dd")
let specefic_date3=new Date("mm-dd-yyyy")
```

2. `Arrays in Js:` Arrays in java-script are dynamic and mutable in nature.  

| Method/Attribute                                      | Description                                                                                             |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `array.length`                                        | Returns the number of elements in the array.                                                            |
| `array.push(element)`                                 | Adds one or more elements to the end of the array and returns the new length.                           |
| `array.pop()`                                         | Removes the last element from the array and returns that element.                                       |
| `array.shift()`                                       | Removes the first element from the array and returns that element.                                      |
| `array.unshift(element)`                              | Adds one or more elements to the beginning of the array and returns the new length.                     |
| `array.slice(start, end)`                             | Returns a shallow copy of a portion of an array specified by start and end.                             |
| `array.splice(start, deleteCount, item1, item2, ...)` | Changes the contents of an array by removing existing elements and/or adding new elements.              |
| `array.concat(array1, array2, ...)`                   | Returns a new array by concatenating one or more arrays.                                                |
| `array.reverse()`                                     | Reverses the order of the elements in the array.                                                        |
| `array.sort(compareFunction)`                         | Sorts the elements of an array in place and returns the sorted array.                                   |
| `array.indexOf(element, fromIndex)`                   | Returns the first index at which a given element can be found in the array, or -1 if it is not present. |
| `array.includes(element, fromIndex)`                  | Determines whether an array includes a certain element, returning true or false as appropriate.         |
| `array.join(separator)`                               | Joins all elements of an array into a string, using a specified separator between each element.         |
| `array.toString()`                                    | Returns a string representing the specified array and its elements.                                     |
| `array.fill(value, start, end)`                       | Fills all elements of an array from a start index to an end index with a static value.                  |

Here is an example code snippet demonstrating the usage of these methods:

```javascript
let array = [1, 2, 3, 4, 5];

console.log(array.length); // Output: 5
console.log(array.push(6)); // Output: 6 (new length of array)
console.log(array.pop()); // Output: 6 (popped element)
console.log(array.shift()); // Output: 1 (shifted element)
console.log(array.unshift(0)); // Output: 5 (new length of array)
console.log(array.slice(1, 3)); // Output: [2, 3]
console.log(array.splice(1, 2, 8, 9)); // Output: [2, 3] (removed elements)
console.log(array.concat([10, 11])); // Output: [0, 8, 9, 4, 5, 10, 11]
console.log(array.reverse()); // Output: [11, 10, 5, 4, 9, 8, 0]
console.log(array.sort((a, b) => a - b)); // Output: [0, 4, 5, 8, 9, 10, 11]
console.log(array.indexOf(8)); // Output: 3
console.log(array.includes(7)); // Output: false
console.log(array.join('-')); // Output: 0-4-5-8-9-10-11
console.log(array.toString()); // Output: 0,4,5,8,9,10,11
console.log(array.fill(7, 2, 5)); // Output: [0, 4, 7, 7, 7, 10, 11]


console.log(array.from(iterable)) // creates array from iterable
```

3. `Some Important Functions`

```js
let numbers = [1, 2, 3, 4, 5];

// Using map to double each element in the array
let doubledNumbers = numbers.map(num => num * 2);

// Using reduce to sum all elements in the array
let sum = numbers.reduce((acc, curr) => acc + curr, 0);

// Using map and reduce together to calculate the square of the sum of squares of all elements
let squareOfSumOfSquares = numbers
  .map(num => num ** 2) // square each element
  .reduce((acc, curr) => acc + curr, 0); // sum the squares

console.log(doubledNumbers); // Output: [2, 4, 6, 8, 10]
console.log(sum); // Output: 15
console.log(squareOfSumOfSquares); // Output: 55
```

4. `Objects:` Objects in Java-script are used to store key-value mapping and are really powerful (similar to dictionary in python and map in java/C++)

```js
// Example object
let person = {
  firstName: 'John',
  lastName: 'Doe',
  age: 30,
  email: 'john.doe@example.com',
};

// Destructuring syntax to extract values from the object
let { firstName, lastName, age } = person;

// Using Object.keys() to get an array of keys
let keys = Object.keys(person);

// Using Object.values() to get an array of values
let values = Object.values(person);

// Using Object.entries() to get an array of [key,value] pairs
let entries = Object.entries(person);

```

5. `Advance Object Methods:`

```js
// Example object
let person = {
  firstName: 'John',
  lastName: 'Doe',
  age: 30,
  email: 'john.doe@example.com',
  // u can also create function 
  greeting: function (){
	  console.log(`Hello ${this.firstName}!`)
  }
};

// Using Object.assign() to copy objects or merge multiple objects
let newPerson = Object.assign({}, person); // Shallow copy of the object
let mergedObject = Object.assign({}, person, { city: 'New York' }); // Merged object with additional properties

// Using Object.freeze() to make an object immutable (cannot be modified)
Object.freeze(person);

// Using Object.seal() to prevent adding or deleting properties, but allows modifying existing properties
Object.seal(person);

// Note if u try to modify the freeze obj it won't throw error,just that change will not propogate

// Using Object.hasOwnProperty() to check if an object has a specific property
let hasProperty = person.hasOwnProperty('age'); // Returns true if 'age' is a direct property of person

// Using delete operator to remove a property from an object
delete person.email;
```

6. `Normal vs Arrow Function:`

| Feature             | Normal Function                                                                  | Arrow Function                                                                                                                                                      |
| ------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Syntax              | `function functionName() {}`                                                     | `(parameters) => {}`                                                                                                                                                |
| Arguments Binding   | Binds its own `this`                                                             | Does not bind its own `this`, inherits from the enclosing lexical context (often the parent scope)                                                                  |
| Context             | Has its own `this` context                                                       | Inherits `this` from the surrounding code                                                                                                                           |
| Scope               | Has its own `arguments` object                                                   | Does not have its own `arguments` object, uses the parent scope's `arguments` object if needed                                                                      |
| Constructor         | Can be used as a constructor                                                     | Cannot be used as a constructor                                                                                                                                     |
| Method Definition   | Can be used to define methods within objects                                     | Can be used to define methods within objects, with inherited `this` binding                                                                                         |
| Function Properties | Has properties like `name`, `length`, etc.                                       | Has properties like `name`, `length`, etc.                                                                                                                          |
| Use Cases           | Generally used for traditional function definitions, callbacks, and constructors | Commonly used for concise function expressions, avoiding issues with `this` binding, and for arrow functions within methods to access the enclosing object's `this` |
| Hoisted             | True                                                                             | False                                                                                                                                                               |
| new                 | use to create new obj instance                                                   | can't use new with arrow functions                                                                                                                                  |

Code example demonstrating the use of `this` property with normal function and arrow function:

```javascript

const myObject = {
  name: "My Object",
  
  // Normal function: "this" refers to the object when invoked as a method.
  startTimerNormal: function() {
    // Assign "this" (which is myObject) to a variable
    const that = this;
    setTimeout(function() {
      // In a normal function, "this" is not bound to myObject by default.
      // Here, we use the variable "that" to access the original context.
      console.log('Normal function:', that.name);
    }, 1000);
  },


  // Arrow function: "this" is lexically inherited from the surrounding scope.
  startTimerArrow: function() {
    setTimeout(() => {
      // Since arrow functions don't have their own "this", they inherit
      // "this" from the enclosing method, which is myObject.
      console.log('Arrow function:', this.name);
    }, 1000);
  }
};

myObject.startTimerNormal(); // After 1 second, prints: Normal function: My Object
myObject.startTimerArrow();  // After 1 second, prints: Arrow function: My Object
```

In the code above, the arrow function in `obj.greet()` maintains the `this` context of the enclosing object (`obj`), allowing it to access the `name` property correctly.


7. `Immediately Invoked Function Expression (IIFE):`

```js
// Immediately Invoked Function Expression (IIFE)
(function() {
  console.log('This is an IIFE!');
})();

// IIFE with parameters
(function(message) {
  console.log(message);
})('Hello from an IIFE!');
```

8. `Hoisting in JS:` Hoisting in JavaScript is a mechanism where variable and function declarations are moved to the top of their containing scope during the compilation phase, before the code is executed. This means that regardless of where variables and functions are declared within a scope, they are conceptually moved to the top of their scope during execution.

	However, it's important to note that **only the declarations are hoisted, not the initializations or assignments**. Let's break down hoisting for variables and functions:
	
	1. **Variable Hoisting**:
	   - Variable declarations (using `var`, `let`, or `const`) are hoisted to the top of their containing scope.
	   - Only the declaration is hoisted, not the initialization. If a variable is initialized later in the code, the value assigned during initialization is not hoisted.
	   - Variables declared with `var` are hoisted to the top of their function scope or global scope.
	   - Variables declared with `let` and `const` are hoisted to the top of their block scope (inside curly braces `{}`), but they are not initialized until the actual declaration statement.
	
	   Example of variable hoisting:
	
```javascript
console.log(x); // Output: undefined (variable declaration is hoisted)
var x = 10; // Variable initialization

console.log(x); // Output: Error (let and const are not initialized)
let x = 10; // Variable initialization
```
   
   2. **Function Hoisting**:

   * Function declarations (using `function functionName() {}`) are fully hoisted, including the function body.
   
   - Function expressions (using `const functionName = function() {}` or `const functionName = () => {}`) are not hoisted like function declarations.
   
   - Hoisted function declarations are available for use anywhere within their containing scope, even before the actual declaration in the code.

Example of function hoisting:

```js
greet(); // Output: Hello, world! (function declaration is hoisted)
function greet() {
  console.log('Hello, world!');
}
```


9. `Loops in JS:`

```js
// For-loop
for (let i = 0; i < 5; i++) {
  console.log(i); // Output: 0, 1, 2, 3, 4
}


// while loop
let i = 0;
while (i < 5) {
  console.log(i); // Output: 0, 1, 2, 3, 4
  i++;
}


// do-while loop
let i = 0;
do {
  console.log(i); // Output: 0, 1, 2, 3, 4
  i++;
} while (i < 5);



//for-in loop
const person = {
  name: 'John',
  age: 30,
  city: 'New York'
};

// Iterates over the keys (properties) of an object
for (let key in person) {
  console.log(key + ': ' + person[key]); // Output: name: John, age: 30, city: New York
}



// for-of loop
const numbers = [1, 2, 3, 4, 5];
// Iterates over the values of an iterable object (e.g., array)
for (let num of numbers) {
  console.log(num); // Output: 1, 2, 3, 4, 5
}

``` 

10. `Higher Order Function in array:`

```js

const numbers = [1, 2, 3, 4, 5];

// map: Transform each element of the array
const squaredNumbers = numbers.map(num => num * num);
// Output: [1, 4, 9, 16, 25]

// filter: Create a new array with elements that pass a condition
const evenNumbers = numbers.filter(num => num % 2 === 0);
// Output: [2, 4]

// reduce: Accumulate values from an array into a single value
const sum = numbers.reduce((acc, curr) => acc + curr, 0);
// Output: 15 (1 + 2 + 3 + 4 + 5)

// forEach: Iterate over each element and perform a function on each element
// diff b/w forEach and map is that map return a new iterator , forEach don't
numbers.forEach(num => console.log(num * 2));
// Output: 2, 4, 6, 8, 10 (doubles each element)

// find: Find the first element that satisfies a condition
const firstEvenNumber = numbers.find(num => num % 2 === 0);
// Output: 2 (first even number)

// some: Check if at least one element satisfies a condition
const hasEvenNumber = numbers.some(num => num % 2 === 0);
// Output: true (at least one even number)

// every: Check if all elements satisfy a condition
const allEvenNumbers = numbers.every(num => num % 2 === 0);
// Output: false (not all numbers are even)

```