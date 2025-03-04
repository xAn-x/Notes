The **Context API** in React is a way to manage **global state** or shared data across your entire application, without having to pass props manually down through every component. Help to resolve problem of `Prop-Drilling.`

```js
// [context]/context.js
import {createContext} from React;

export default const Context=createContext();
```

```jsx
// [context]/ContextProvider.jsx
import Context from "@/context/context.js"

export default function ContextProvider({children}){
	const store={} // global-store,can even pass multiple-values
	return (
		<Context.Provider value={{store}}>
			{children}
		</Context.Provider>
	)
}
 ```

```jsx
// [src]/App.jsx
import ContextProvider from "@/context/ContextProvider.jsx"

export default function App(){
	return (
		// Every Component inside this will have access to 'store'
		<ContextProvider>
			<Navbar/>
			<div className="flex justify-between item-center">
				<Sidebar className="flex-1"/>
				<MainPage className="flex-3"/>
			</div>
			<Footer/>
		</ContextProvider>
	)
}
```

### How to add and access data in global context

```jsx
// Products.jsx
// adding data to global context
import React,{useEffect,useState,useContext} from 'react';
import Context from "@/context/Context.js"

export default Products(){
	// fetch-prods on list
	const [favs,setFavs]=useState([]);
	const [cartProds,setCartProds]=useState([]);
	const {store} = useContext(Context); // accessing store
	
	useEffect(()=>{
		const fav_prods=await fetchFavorites();
		setFavs(favs);
		store.favs=favs;
		
		const prods_in_cart=await fetchProdsInCart();
		setCartProds(prods_in_cart);
		store.cart=prods_in_cart;
	},[])

	// add and remove prods
	function addToCart(product){
		// this is similar to redux but u can simply push also
		store.cart=[...store.cart,product]
	}

	function removeFavorites(product_id){
		store.favs=store.favs.filter(fav=>fav.id!=product_id)
	}
	
	return (
		// ...details
	)
}
```
