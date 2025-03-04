Hooks are a new feature in React that allows you to use state and other React features without writing a class component. They are a simpler and more concise way to write React components.

Hooks are declared using the `useState` and `useEffect` functions. The `useState` function allows you to create and update state variables. The `useEffect` function allows you to perform side effects, such as fetching data or setting up event listeners.

Here is an example of a React component written using hooks:

```javascript
import React, { useState, useEffect } from 'react';
[[03- Context API in React]]
const MyComponent = () => {
  const [count, setCount] = useState(0);

  useEffect(() => {
    // Fetch data from a server
    fetch('https://example.com/data.json')
      .then(response => response.json())
      .then(data => setCount(data.count));
  }, []);

  return (
    <div>
      <h1>Count: {count}</h1>
    </div>
  );
};
```

This component is equivalent to the following class component:

```javascript
class MyComponent extends React.Component {
  state = {
    count: 0
  };

  componentDidMount() {
    // Fetch data from a server
    fetch('https://example.com/data.json')
      .then(response => response.json())
      .then(data => this.setState({ count: data.count }));
  }

  render() {
    return (
      <div>
        <h1>Count: {this.state.count}</h1>
      </div>
    );
  }
}
```

Hooks are a more concise and easier way to write React components. They are also more flexible than class components, as they can be used in both functional and class components.

### `Different types of Hooks`

**useState**

The `useState` hook allows you ==to create and update state variables==. It takes an initial state value as an argument and returns an array with two elements: the current state value and a function to update the state value.

**useEffect**

The `useEffect` hook allows you ==to perform side effects==, such as fetching data or setting up event listeners. It takes a function as an argument, which is called after the component is mounted and every time the component updates.

**useRef**

The `useRef` hook allows you ==to create a mutable ref object==. A ref object is a way to store a value that persists between renders. It is often used to store DOM elements or other values that need to be accessed outside of the component.

**useMemo**

The `useMemo` hook allows you ==to memoize a value==. This means that the value will only be computed once, even if the component that uses it re-renders. This can improve performance for components that have expensive computations.

**useCallback**

The `useCallback` hook allows you to ==memoize a function==. This means that the function will only be created once, even if the component that uses it re-renders. This can improve performance for components that have expensive functions.

For e.g. if one component is depending on several small components than rather than creating a copy for each button/sub-component we can use `useCallback` to cache and memozize(func and its args) to get used with all thing using a single instance.  

#### Difference b/w useMemo & useCallback:

| Feature     | `useMemo`                                                               | `useCallback`                                                            |
| ----------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Purpose     | Memoizes a value                                                        | Memoizes a function                                                      |
| Syntax      | `const memoizedValue = useMemo(() => {...}, [dependencies])`            | `const memoizedFunction = useCallback(() => {...}, [dependencies])`      |
| When to use | When you want to avoid recomputing a value that is expensive to compute | When you want to avoid recreating a function that is expensive to create |
| Example     | Memoizing the result of a complex calculation                           | Memoizing a function that handles an event listener                      |

**Example code:**

```javascript
import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';

const MyComponent = () => {
  const [count, setCount] = useState(0);

  // Fetch data from a server
  useEffect(() => {
    fetch('https://example.com/data.json')
      .then(response => response.json())
      .then(data => setCount(data.count));
  }, []);

  // Create a ref to the input element
  const inputRef = useRef(null);

  // Memoize the function that handles the input change
  const handleChange = useCallback(() => {
    // Do something with the input value
  }, []);

  // Memoize the value of the input element
  const inputValue = useMemo(() => {
    // Get the value of the input element
    return inputRef.current.value;
  }, [inputRef]);

  return (
    <div>
      <input ref={inputRef} onChange={handleChange} />
      <p>Count: {count}</p>
      <p>Input value: {inputValue}</p>
    </div>
  );
};
```

This component uses the `useState`, `useEffect`, `useRef`, `useMemo`, and `useCallback` hooks to fetch data from a server, create a ref to the input element, memoize the function that handles the input change, and memoize the value of the input element.

I hope this helps! Let me know if you have any other questions.

### `Create Custom Hooks`

To create a custom hook, you simply need to create a JavaScript function that starts with the word `use`. For example:

```javascript
import { useState } from 'react';

const useMyCustomHook = () => {
  const [count, setCount] = useState(0);

  const incrementCount = () => {
    setCount(count + 1);
  };

  return {
    count,
    incrementCount
  };
};
```

Custom hooks are a great way to share reusable logic between React components. They can be used to encapsulate common functionality, such as fetching data, managing state, or handling events.