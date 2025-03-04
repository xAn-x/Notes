[]()**Immer** (German for: always) is a tiny package that allows you to work with immutable state in a more convenient way.

 - _Immutable ds allow for (efficient) change detection_: if the reference to an object didn't change, the object itself did not change. 
 
- _Makes cloning relatively cheap_: Unchanged parts of a data tree don't need to be copied and are shared in memory with older versions of the same state.

```shell
npm install immer
```

```ts
const baseState = [  
	{  
		title: "Learn TypeScript",  
		done: true  
	},  
	{  
		title: "Try Immer",  
		done: false  
	}  
]


// without immer:
const next_state=baseState.slice() // shallow-copy of obj
next_state[1]={
	...next_state[1],
	done:true
}
next_state.push({title:"new obj pushed"}) // this is fine as we have created the copy 1st


// with immer:
import {produce} from immer
const nextState=produce(baseState,draft=>{
	// any changes u done in draft,will propogate to nextState without mutating baseState
	draft[1].done=true
	draft.push({title:"new obj pushed"})
})

```

![[Pasted image 20241201202850.png]]

#### Some Addition features one can opt

![[Pasted image 20241201203057.png]]

i.e. if u want to use immer with Maps and Sets u have to do:

```ts
import {enableMapSet,produce} from immer
enableMapSet() // this will make maps and set compatible with immer
  
const usersById_v1 = new Map([  
["michel", {name: "Michel Weststrate", country: "NL"}]  
])  
  
const usersById_v2 = produce(usersById_v1, draft => {  
draft.get("michel").country = "UK"  
})
```

#### React & Immer

1. _useState + Immer:_ The `useState` hook assumes any state that is stored inside it is treated as immutable. Deep updates in the state of React components can be greatly simplified as by using Immer.

```js
import React, { useCallback, useState } from "react";
import {produce} from "immer";

const TodoList = () => {
  const [todos, setTodos] = useState([
    {
      id: "React",
      title: "Learn React",
      done: true
    },
    {
      id: "Immer",
      title: "Try Immer",
      done: false
    }
  ]);

  const handleToggle = useCallback((id) => {
    setTodos(
	 // use produce to make changes and return newState 
      produce((draft) => {
        const todo = draft.find((todo) => todo.id === id);
        todo.done = !todo.done;
      })
    );
  }, []);

  const handleAdd = useCallback(() => {
    setTodos(
      produce((draft) => {
        draft.push({
          id: "todo_" + Math.random(),
          title: "A new todo",
          done: false
        });
      })
    );
  }, []);

  return (<div>{*/ See CodeSandbox */}</div>)
}
```

2. _useImmer:_ Since all state updaters follow the same pattern where the update function is wrapped in `produce`, it is also possible to simplify the above by leveraging the [use-immer](https://www.npmjs.com/package/use-immer) package that will wrap updater functions in `produce` automatically:
   
```js
import React, { useCallback } from "react";
import { useImmer } from "use-immer";

const TodoList = () => {
  const [todos, setTodos] = useImmer([
    {
      id: "React",
      title: "Learn React",
      done: true
    },
    {
      id: "Immer",
      title: "Try Immer",
      done: false
    }
  ]);

  const handleToggle = useCallback((id) => {
    setTodos((draft) => {
      const todo = draft.find((todo) => todo.id === id);
      todo.done = !todo.done;
    });
  }, []);

  const handleAdd = useCallback(() => {
    setTodos((draft) => {
      draft.push({
        id: "todo_" + Math.random(),
        title: "A new todo",
        done: false
      });
    });
  }, []);

  // etc

```

3. _useImmerReducer:_ When using Redux, one can use `useReducer + Immer` to make sure the changes in state are done in a pure fashion.
   
   Or we can use the `useImmerReducer` from 'use-immer' module to simplify and shorten the code.

```js
import React, { useCallback } from "react";
import { useImmerReducer } from "use-immer";

const TodoList = () => {
  const [todos, dispatch] = useImmerReducer(
    (draft, action) => {
      switch (action.type) {
        case "toggle":
          const todo = draft.find((todo) => todo.id === action.id);
          todo.done = !todo.done;
          break;
        case "add":
          draft.push({
            id: action.id,
            title: "A new todo",
            done: false
          });
          break;
        default:
          break;
      }
    },
    [ /* initial todos */ ]
  );

  //etc
```

4. _Redux + Immer:_

```js
import {produce} from "immer"

// Reducer with initial state
const INITIAL_STATE = [
    /* bunch of todos */
]

const todosReducer = produce((draft, action) => {
    switch (action.type) {
        case "toggle":
            const todo = draft.find(todo => todo.id === action.id)
            todo.done = !todo.done
            break
        case "add":
            draft.push({
                id: action.id,
                title: "A new todo",
                done: false
            })
            break
        default:
            break
    }
})
```