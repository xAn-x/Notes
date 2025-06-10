 React Router is a simple, declarative routing library. Its only job will be matching the URL to a set of components, providing access to URL data, and navigating around the app. This strategy is popular for `Single Page Apps`.

```shell
npm install react-router-dom
```

### Routing

Routes are configured by rendering `<Routes>` and `<Route>` that couple URL segments to UI elements.

```jsx
// App.js
import React from "react"; 
import ReactDOM from "react-dom/client";
import { BrowserRouter,Routes,Route } from "react-router"; 
import App from "./app"; 
import {fetchProjects} from "@/componets/project.jsx"

const root = document.getElementById("root"); ReactDOM.createRoot(root).render(
	// Wrap your whole app with BrowserRouter
	<BrowserRouter> 
		// Wrap all your Routes inside Routes-Wrapper
		<Routes>
			// Route ele takes endpoint-path and element to display
			<Route path="login" element={<Login/>}/>
			<Route path="about" element={<About/>}/>


			// Can also create Nested Routing
			// path of the parent is auto included in the child
			// Here U can use <DashBoord/> to define Layout,
			// U can use react-router-dom's <Outlet/> for that
			<Route path="dashboard"}>
				 // Defaults to Home
				 <Route index element={<Home />} /> 
				 // '/dashboard/setting' renders the Settings component
				<Route path="settings" element={<Settings />} /> 
			</Route>


			// Handelling Dynamic Segments
			// loaders to handle async-data,check bottom for more info
			<Route path="projects" loader={fetchProjects}>
				<Route index element={<ProjectHome/>}/>
				# :edit will be treated as a variable/param
				# can access these params using useParams()
				# u can make a route segment optional by adding '?'
				<Route path=":pid" element={<Project/>}/>
				<Route path=":pid/edit" element={<EditProject/>}/>
			</Route>


			// Catch-All-Routes
			<Route path="/*" element={<PageNotFound/>}
		</Routes>
	</BrowserRouter> 
);
```

### Layout/Rendering Child Routes

```jsx
// dashboard.jsx
import React from 'react'
import {Outlet} from 'react-router' // Exposes Child-Components

export default function Dashboard() { 
	return ( <div>
		<Navbar/> 
		<h1>Dashboard</h1> 
			{/* will either be <Home/> or <Settings/> */} 
			<Outlet /> 
		</div> 
		<Footer/>
	); 
}
```


### How to Access Dynamic-Params inside a Component

```jsx
// project.jsx
import {useParams} from 'react-router'
export default Project(){
	const params=useParams() // {param_name:"value"}
	return (
		//...details
	)
}
```


### Linking

Link routes from UI with `Link` and `NavLink`

```jsx
//navbar.jsx
import {Link,NavLink,useNavigate} from "react-router"

export default Navbar(){
	let navigateTo=useNavigate()
	return (
		<nav className="flex justify-end item-center">
			<NavLink 
				to="/" // define path
				// isActive,a default prop that all Links/NavLinks get
				// isActive auto match route with Link path 
				className={({ isActive }) => isActive ? "active" : "" }
			/>
			<Link 
				to="/about" 
				className={({ isActive }) => isActive ? "active" : "" }
			/>
			// handling clicks using useNavigate()
			<button onClick={()=>navigateTo("/dashboard")}>
				Visit DashBoard
			</button>
		</nav>
		
	)
}
```


### Loaders

Route loaders provide data to route components before they are rendered. They are only called on the server when server rendering or during the build with pre-rendering.

Loaders are functions that run when a route is matched, allowing you to fetch data (or do other asynchronous tasks) before rendering the associated component (Similar to `useEffect()` but optimized as it caches data once we hit the route).

This can help with handling data fetching or other tasks before displaying the page.

#### Why use __loader__ and not __useEffect()__

When we use `useEffect()` for data fetching, for data the component renders first & then the data is fetched, this means user may see an empty or loading state in UI.

Loaders are part of `React Router's data fetching mechanism`, meaning they run before the component is rendered. This ensures the component will only render when the required data is already available, avoiding the need for a loading state within the component itself. The `loading happens at the routing level`, not inside the component.


```jsx
//project.jsx

// help to get data once-fetched inside component
import {useRouteLoaderData} from 'react-router'
import {useState,useEffect} from 'react'

//loader, async fetch the projects and supply to subseq component 
export const fetchProjects=async ()=>{
	const response= await fetch("url")
	const projects= await response.json()
	return projects;
}


// component
export default Project(){
	const [loading,setLoading]=useState(true)
	const projects=useRouteLoaderData(fetchProjects)
	
	//change loading state
	useEffect(()=>{
		if(projects) setLoading(false)
	},[projects])
	
	return loading ? <LoadingSkeleton/> : (
		//...Projects-details
	)
}

```