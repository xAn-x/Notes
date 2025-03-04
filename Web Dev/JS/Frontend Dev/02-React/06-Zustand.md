Zustand is a small, fast, and scalable state management solution for JavaScript ( Redux but simplified).

#### Features

- Little Boilerplate
- Doesn't rely on provider
- Faster
- State merging by default
- Expandable by default

#### Re-rendering

In react , when ever we are subscribed to anything using `context-api` it renders all the components that use the context, even the component does not depends on that property.

In Zustand only those components will be re-render that uses them properties.

In simple terms, re-render only affects the components that depend on the specific parts of the state that changed. 

```shell
npm install zustand@latest
```

```ts
// [store]/cart-store.ts

import {create} from 'zustand'

type Product={
	id:number,
	name:string,
	description:string,
	price:number,
}

//set is a special func,that each of our slice get
// it help us to manipulate the state of the store
// no need to use `useState` or other hooks
// zustand will manage all these changes automatically
const useCartStore=create((set)=>{
	cart:Product [],
	
	addToCart:(product:Product):=>(
		set((state)=>(
			{cart:[...state.cart,product]}
		))
	),
	
	removeFromCart:(id:number):(
		set((state)=>(
			{cart:state.cart.filter(prod=>prod.id!=id)}
		))
	),
	
	clearCart:():=>(
		set((state)=>(
			{cart:[]}
		))
	)
})

export deafalt useCartStore;
export {type Product};

```

```ts
// products.tsx
import {useCartStore,Product} from "@/store/cart-store.ts"
export default function Products(){
	const {addToCart}=useCartStore();
	function handleAddToCart(product:Product){
		addToCart(product)
	}
	return (
		 // ... definations
	)
}
```

```ts
//cart.tsx
import {useCartStore,Product} from "@/store/cart-store.ts"
export default function Cart(){
	const {cart,removeFromCart,clearCart}=useCartStore();
	function handleRemoveFromCart(id:number){
		removeFromCart(id)
	}
	return(
		<div>
			<div className="products flex justify-around items-center">
			{
				cart.map(product=>(
					<div>
					// ..details
					</div>
					<button onClick={
						()=>handleRemoveFromCart(cart.id)}>
						Remove Item
					</button>
				))
			}
			</div>
			<button onClick={clearCart}>
				Clear Cart
			</button>
		</div>
	)
}
```

### Using Zustand with `Immer`:

#### What is Immer:

Immer is a library that simplifies working with immutable data structures in JavaScript.  Instead of manually creating entirely new copies of objects or arrays when making changes, Immer lets you modify data directly using familiar methods like `push`, `splice`, or property assignment. 

Internally, Immer uses a technique called `structural sharing` to create efficient copies only where necessary, minimizing memory usage and improving performance.  The changes you make appear as if they're in-place, but Immer ensures that the original data remains untouched, maintaining immutability.

learn-more @ [[07-Immer.js]] 

```js
// Without Immer (mutable)
let myObject = { name: "Alice", age: 30 };
myObject.age = 31; // Directly modifies the original object
console.log(myObject); // Output: { name: "Alice", age: 31 }


// With Immer (immutable)
import produce from 'immer';

let myImmerObject = { name: "Bob", age: 25 };

let updatedImmerObject = produce(myImmerObject, draft => {
  draft.age = 26; // Modifies a draft, not the original
});

console.log(myImmerObject); // Output: { name: "Bob", age: 25 }
console.log(updatedImmerObject); // Output: { name: "Bob", age: 26 } 

//More complex example showing nested objects
let complexObject = {
  user: {
    name: 'Charlie',
    address: {
      street: '123 Main St'
    }
  }
};

let updatedComplexObject = produce(complexObject, draft => {
  draft.user.address.street = '456 Oak Ave';
});

console.log(complexObject); 
console.log(updatedComplexObject); //Only the street address is changed, original object is untouched.

```

#### Using Immer with Zustand:

```ts
import { create } from 'zustand';
import produce from 'immer';

interface User {
  id: number;
  name: string;
  email: string;
  address: {
    street: string;
    city: string;
    zip: string;
  };
}

interface UserStore {
  user: User | null;
  updateUserAddress: (newStreet: string) => void;
}

const useUserStore = create<UserStore>()((set) => ({
  user: {
    id: 1,
    name: 'John Doe',
    email: 'john.doe@example.com',
    address: {
      street: '123 Main St',
      city: 'Anytown',
      zip: '12345',
    },
  },

// without immer
 updateUserAddressWithoutImmer: (newStreet: string) =>
	set(
    // wit produce we can directly mutate the property we want
    // without affecting other
      (state: UserStore) => {
        if (state.user) {
          state.user={
	          ...state.user,
	          address:{
		          ...state.user.address,
		          street:newStreet
	          }
          }
        }
      }
    ),


 // with immer
  updateUserAddress: (newStreet: string) =>
    set(
    // wit produce we can directly mutate the property we want
    // without affecting other
      produce((state: UserStore) => {
        if (state.user) {
          state.user.address.street = newStreet;
        }
      })
    ),
}));


//Example usage in a React component
import React from 'react';

const MyComponent: React.FC = () => {
  const { user, updateUserAddress } = useUserStore();

  const handleAddressChange = () => {
    updateUserAddress('456 Oak Ave');
  };

  return (
    <div>
      {user && (
        <div>
          <p>Name: {user.name}</p>
          <p>Address: {user.address.street}, {user.address.city}, {user.address.zip}</p>
          <button onClick={handleAddressChange}>Update Address</button>
        </div>
      )}
    </div>
  );
};

export default MyComponent;
```

