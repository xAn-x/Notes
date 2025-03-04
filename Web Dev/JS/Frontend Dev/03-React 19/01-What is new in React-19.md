![[Pasted image 20250109093038.png]]

![[Pasted image 20250109093120.png]]

![[Pasted image 20250109093147.png]]

![[Pasted image 20250109093228.png]]

---

### Things that became obsolete in React-19

![[Pasted image 20250109093333.png]]

---

### Suspense and use()

In React 19, the `Suspense` component and the `use` hook are powerful tools for handling asynchronous operations and improving user experience.

#### Suspense
`Suspense` allows you to suspend the rendering of a component until a certain condition is met, such as data fetching or lazy loading. It helps in creating a smoother user experience by displaying a fallback UI (like a loading spinner) while the component is not ready to be rendered. For example, you can use `Suspense` with `React.lazy` to lazy load components or with data fetching libraries to handle [loading states](https://www.youtube.com/watch?v=xwr7ZyfKzjA).

#### use Hook
The `use` hook is a new addition in React 19 that simplifies handling asynchronous code, promises, and contexts. It can be used conditionally, inside loops, and with `Suspense` and error boundaries. When called with a promise, the `use` hook suspends the component until the promise resolves, eliminating the need for manual loading states. This integration with `Suspense` enhances the user experience by making asynchronous resource handling more seamless.



![[Pasted image 20250109093447.png]]

```tsx
import {use} from "react";

const fetchTodos=async ()=>{
	const rest=await fetch("url")
	const todos=res.json()
	return todos
}

const FetchTodos=()=>{
	const todos=use(fetchTodos());
	return <div>{
		{todos.map(todo=>(
			<div>
				<div>{todo.title}</div>
				<div>{todo.message}</div>
				<div>{todo.status}</div>
			</div>
		))}
	}</div>
}

export default app=()=>{
	return (
		<Suspense>
			<FetcTodos/>
		</Suspense>
	)
}
```
### Action in React-19

**Actions** are a new feature designed to simplify handling asynchronous operations, particularly in form submissions and loading states.

Actions are asynchronous functions that can be used to handle form submissions and other asynchronous tasks more efficiently. They help reduce boilerplate code and make it easier to manage pending states, errors, and optimistic updates.

```tsx

// Method-1
type User={
	id:Date,
	name:string
}

const MyForm = () => {
  const actionFn=async (formdata)=>{
	"use server"
	const user:User={
		id=Date.now(),
		name:formdata.name
	}
	const resp=await fetch('/api/submit', {
		  method: 'POST',
		  body: JSON.stringify(user),
	});
  }

  return (
    <form action={actionFn}>
      <input type="text" name="name" />
      <button type="submit">Submit</button>
    </form>
  );
};


// Method-2
import { useActionState } from 'react'
// with this u can use action and gets is state anywhere not just form
const MyForm = () => {
const formAction= async (prv_state,formData) => {
    // Perform your async operation here
    "use server"
    const user:User={
		id=Date.now(),
		name:formdata.name
	}
    await fetch('/api/submit', {
      method: 'POST',
      body: JSON.stringify(user),
    });
  }
  
  const [state, actionFn] = useActionState(formAction);

  return (
    <form action={actionFn}>
      <input type="text" name="name" />
      <button type="submit">Submit</button>
    </form>
  );
};

/*
Similarly we have other hooks, like:

useOptimistic hook allows you to show optimistic updates in the UI providing a smooth User Experience while an action is still processing. 
  
useFormStatus hook provides information about the last 
submission of the parent’s form.
  
*/
```