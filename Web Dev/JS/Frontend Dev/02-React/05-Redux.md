Redux is a state management lib. It helps you manage the state of our application (all the data that your app needs to function) in a single, centralized location called the **store**.

Redux uses functional programming for state-management and we have to deal with pure-functions to propagate any action or changes.

### Key Concepts:

- **Store:** Holds the entire state of your application.
- **Actions:** Plain JavaScript objects that describe _what happened_ in your app (e.g., `ADD_TODO`, `TOGGLE_TODO`).
- **Reducers:** Pure functions that take the current state and an action, and return a _new_ state based on the action (they don't modify the existing state).
- **Dispatch:** The function you use to send actions to the store.

```shell
npm install react-redux @reduxjs/toolkit
```


### Configure Store

```js
// [redux]/store.js

import {consfigureStore} from "@reduxjs/toolkit" // global-store
import todoReducer from "@/todoSlice.js"

export const store=configureStore({
	reducer:todoReducer // list of reducers
})
```

#### Create Slice

```js
// [redux]/[features]/[todos]/todoSlice.js

import {createSlice,nanoid} from "@reduxjs"

export const todoSlice=createSlice({
	name:'todo ',//slice-name, will be used by redux-toolkit & dev-tool
	initalState:[],
	reducers:{
		// contains properties and functions
		// each func get curr slice-state & action-obj 
		// action-obj contains info that we want to send to our func 
		addTodo: (state,action)=>{
			const todo={
				id:nanoid(),
				msg:action.payload.msg,
				completed:false,
			}
			state=[...state,todo] // update state			
		}

		updateTodo:(state,action)=>{
			state=state.map(todo=>todo.id==action.payload.id ? {
				...todo,
				todo.msg=action.payload.msg
			} : todo)
			
		}

		deleteTodo:(state,action)=>{
			state=state.filter(todo=>todo.id!==action.payload.id)
		}
	}
})

// export actions and reducers
export const {addTodo,updateTodo,deleteTodo} = todoSlice.actions
export default todoSlice.reducer
```

### Wrapping our application

```jsx
// [src]/app.jsx

import {Provider} from 'react-redux'
import {store} from '@/store.js'

export default function(){
	return (
		<Provider store={store}>
			//...childerens
		</Provider>
	)
}
```

### How to use them inside our components

`useSelector` and `useDispatch` make it easier to interact with the Redux store from your React components.

* **`useSelector`**: This hook lets you _subscribe_ to parts of the Redux store and automatically re-render your component whenever that part of the state changes. It takes a _selector_ function as an argument. The selector function receives the entire Redux state and returns the specific piece of data you're interested in.
  
* **`useDispatch`**: This hook gives you access to the Redux store's `dispatch` function. You use `dispatch` to send actions to the store, which triggers state updates via reducers.

_Dispatching actions to update state:_

```jsx
// todoForm.jsx
import {useDispatch} from "react-redux"
import {addTodo} from "@/todoSlice.js"

export default TodoForm(){
	const dispatcher=useDispatch()

	function addTodoHandler(msg){
		action={
			payload:{
				msg
			}
		}
		dispatch(addTodo(action))
	}

	// same for update and delete
	return (
		// ...details
	)
}
```


_Getting items in store/slice_

```jsx
// todosList.jsx

import {useSelector} from 'react-redux'

export default TodosList(){
	const todos=useSelector(state => state.todo)
	return (
		...details
	)
}
```