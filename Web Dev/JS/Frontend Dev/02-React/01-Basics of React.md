React is a free and open-source front-end library for building user interfaces. It is maintained by Facebook and a community of developers.

React ==uses a declarative approach== to building UIs. This means that you describe what the UI should look like, and React takes care of updating the UI to match your description. This makes it easy to build complex and dynamic UIs.

React is also very efficient. It ==uses a virtual DOM== to keep track of the state of the UI. This means that React ==only updates the parts of the UI that need to be updated==, which can improve performance.

Some of the benefits of using React:

* Declarative programming model
* Efficient virtual DOM
* Large community and ecosystem
* Easy to learn and use
* For building single page web-applications 

### `Installation`

```js
% npx install create-react-app 
```

### `JSX and Babel`

* **JSX** is a syntax extension for JavaScript that allows you to write HTML-like code in your React applications.
* JSX looks like HTML, but it is actually JavaScript. This makes it easy to write React components that are both declarative and easy to read.

* **Babel** is a JavaScript compiler that converts JSX and other new JavaScript features into code that can be understood by browsers.


### `First React Component`

**Components** are the building blocks of React applications. They are reusable pieces of code that can be composed to create complex UIs.

There are two main types of components in React:

* **Class components** are defined using the `class` keyword. They have a state and lifecycle methods.
* **Functional components** are defined using the `function` keyword. They are simpler and easier to write than class components, but they do not have a state or lifecycle methods.

**Class components** are more powerful than functional components, but they are also more complex. They are best used for components that need to have a state or lifecycle methods.

**Functional components** are simpler and easier to write than class components. They are best used for components that do not need to have a state or lifecycle methods.

| Feature           | Class Component | Functional Component |
| ----------------- | --------------- | -------------------- |
| Syntax            | `class` keyword | `function` keyword   |
| State             | Yes             | No                   |
| Lifecycle methods | Yes             | No                   |
| Complexity        | More complex    | Simpler              |

```js
import {Component} from 'react'

// Class Component
class Navbar extends Component{
	constructor(props){
		super()
	}

	// this method will rerun every time a state or prop is updated
	render(){
		return (
			<>
				<h1>This is a functional Component</h1>
				<h2>React is Awesome</h2>
			</>
		)
	}
}


// functional Component
function Navbar(props){
	return (
		<>
			<h1>This is a functional Component</h1>
			<h2>React is Awesome</h2>
		</>
	)
}

// Exporting component to increase re-usability
export default Navbar;
```

### `States and Props:`

**State** is a special object that holds information that can change over time. It is used to keep track of the current state of a component. React detects the change in state to update the DOM. 

**Props** are short for "properties". They are passed from a parent component to a child component. Props are used to pass data and functionality to child components. These are immutable and read-only.

**Key differences between state and props:**

* State is local to a component. It can only be accessed and modified by the component that owns it.
* Props are passed from a parent component to a child component. They cannot be modified by the child component.
* State can change over time. Props are typically static.

**When to use state:**

* To keep track of the current state of a component
* To store data that needs to be shared between multiple components
* To store data that needs to be updated over time

**When to use props:**

* To pass data and functionality from a parent component to a child component
* To configure a child component
* To provide data to a child component that will not change over time

**In short:**
* **State:** Local, mutable data that can change over time.
* **Props:** Immutable data passed from parent to child components.

**Example:**


#### In Functional Component

```javascript
// useState is use to create a statefull variable in a functional component
import {useState} from React

// Parent component
const Parent = () => {
  const [count, setCount] = useState(0);
  setCount(count+1)
  return (
    <Child count={count} />
  );
};

// Child component
const Child = (props) => {
  return (
    <div>
      <h1>Count: {props.count}</h1>
    </div>
  );
};
```

#### In Class Component

```js
class Parent extends Componet{
	constructor(props){
		super(props)
		this.state={
			count:0
		}
	}
	render(){
		// update state
		this.setState({count:this.state.count+1})
		return (
			<Child count={count} />
		)
	}
}

class Child extends Component{
	constructor(props){
		super(props)
	}
	render(){
		return (
			<div>
				<p>Count:{this.props.count}</p>
			</div>
		)
	}
}
```


### `LifeCycle Methods in React:`

Lifecycle methods are special functions that are called at specific times during the lifecycle of a React component. They allow you to perform certain actions at specific times, such as when the component is mounted, updated, or unmounted.

Here is a list of the most common lifecycle methods:

* **componentDidMount()** - Called after the component is mounted to the DOM.
* **componentDidUpdate()** - Called after the component is updated.
* **componentWillUnmount()** - Called before the component is unmounted from the DOM.
* **ShouldComponentUpdate** - Called Before updating any component to determine if or if not to update. 

These lifecycle methods can be used to perform a variety of tasks, such as:

* Fetching data from a server
* Setting up event listeners
* Updating the DOM
* Cleaning up resources

**Example:**

```js
class MyComponent extends React.Component {
  // runs 1st time the component is created,for init of states and props
  constructor(props){
	  super(props)
  }
  // runs whenever a state or prop is change to update DOM
  render(){}
  // runs only once when component is mounted
  componentDidMount(){}
  // whenever some update occur
  componentWillUpdate(){}
  // whenever need to chk wether or not to update
  shouldComponentUpdate(){}
  //whenever component is deleted
  componentWillUnmount(){} 
}
```

**Real Life Example:**

```javascript
class MyComponent extends React.Component {
  componentDidMount() {
    // Fetch data from a server
    fetch('https://example.com/data.json')
      .then(response => response.json())
      .then(data => this.setState({ data }));
  }

  componentWillUnmount() {
    // Clean up event listeners
    window.removeEventListener('scroll', this.handleScroll);
  }
}
```

In this example, the `componentDidMount()` lifecycle method is used to fetch data from a server. The `componentWillUnmount()` lifecycle method is used to clean up event listeners.

**In short:**

Lifecycle methods are special functions that are called at specific times during the lifecycle of a React component. They allow you to perform certain actions at specific times, such as when the component is mounted, updated, or unmounted.

I hope this helps! Let me know if you have any other questions.

We mostly use functional Components , as every functionality of Class Component is also +nt in functional Component because of Hooks.

#### How to use in Functional Components

```js
// useEffect() is used to mimic lifecycle methods,can have multiple
import {useEffect} from React
function Component(){
	useEffect(()=>{
		// what to do
	},[states...])
	
	return(
		<div></div>
	)
}
```