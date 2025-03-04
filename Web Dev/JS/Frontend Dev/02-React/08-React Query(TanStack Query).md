React Query is a JavaScript library that simplifies fetching, caching, and updating asynchronous data in React applications.

### _WHY ?_

![[Pasted image 20241206195201.png]]

![[Pasted image 20241206195254.png]]

## Simple Example:

### Without React-Query:

```tsx
import {useEffect,useState} from 'react';
import axios from 'axios';

type SuperHero = {
	id: int,
	name: string,
	description: string,
}
export default Home=()=>{
	const [loading,setLoading]=useState<boolean>(true)
	const [data,setData]=useState<SuperHero []|null>(null)
	
	useEffect(async ()=>{
		const resp=await axios.get("/superheroes")
		setData(superheros)
		setLoading(false)
	},[])
	
	if(loading) return <div>Loading Data...</div>
	
	return (
		<div>
			...details
		</div>
	)
};
```

### With React-Query

```shell
npm i @tanstack/react-query 
npm i @tanstack/react-query-devtools
npm i -D @tanstack/eslint-plugin-query  # for type checking 
```

Wrap whole app with `QueryClientProvider`

```tsx
// app.tsx

import {QueryClientProvider,QueryClient} from '@tanstack/react-query'
import {ReactQueryDevtools} from `@tanstack/react-query-devtools`

// initial config for tanstack-query
const query_client=new QueryClient()
export default function App(){
	return (
		<QueryClientProvider client={query_client}>
			{...childrens}
		<ReactQueryDevtools initalOpen={false} position="bottom-right"/>
		</QueryClientProvider>
	)
}
```

Fetching Data

```tsx
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import {minutes_to_milli} from "@/my_utils"

type SuperHero= {
  id: number;
  name: string;
  description: string;
}

export default Home = () => {
  const { isLoading, data, isError, error, isFeatching } =
  useQuery<SuperHero[], Error>({
	  queryKey:['fetchSuperHeros'], // use for caching
	  queryFn:  ()=> axios.get("/api/superheros"),
	  
	  // extra-functionalities
	  gcTime: minutes_to_mili(5), //cache res data for 5 min
	  staleTime:  minutes_to_mill(2), // time after which resp get staled and have to initiate re-fetch
	  refetchOnMount:true, // will refetch data every time comp mount and data is staled
	  refetchOnWindowFocus:true, // weather to refetch data again whenever window come in focus, syncing with remote data
	  refetchInterval: 2e3, //refetch every 2 sec
	  refetchIntervalInBackground:true, // keep refetching data in the bg even when the site is not in focus
});

  if (isLoading) return <p>Loading...</p>;
  if (isError) return <p>Error: {error?.message}</p>;

  return (
    <div>
      {data?.map((hero) => (
        <p key={hero.id}>{hero.name}</p>
        <p>{hero.description}</p>
      ))}
    </div>
  );
};
```

> [!NOTE]
> React-Query automatically caches the query response and store it for 5 min, so whenever the user redirect from a page and come back the it is immediately served with the cached data & BTS react-query again make a request to get the data in case something changes to serve the user and once fetch it updates the prv data-state.

### How to fetch data on user request

If we want to fetch data only when user request (e.g. on click of some button and not to auto invoke when component mount) we have to first inform react-query to not get call automatically.

```jsx
import {useQuery} from '@tanstack/react-query'
import axios from 'axios'

export default const Home= ()=>{
	// refetch is a callback that invoke react-query again
	const {isLoading,data,isError,error,refetch}=useQuery(
		['fetch-data'],
		async ()=> axios.get("url").then(res=>res.json)
		// disable react-query to tun automa[]()tically
		{
			enabled:false
		}
	)
	
	if (isLoading) return <h2>Loading...</h2>
	return (
		<div>
			{/*Pass `refetch` to onCLick handller*/}
			<button onClick={refetch}>Fetch Data</button>
			...details
		</div>
	)
}
```
### Mutation in react-query

We can even Mutate (Modify) or send data to a server (create, update or delete oprn) data using `useMutation` hook in react-query 

```jsx
import {useMutation,useQuery} from '@tanstack/react-query';
import axios from 'axios';

export default const Home=()=>{
	const queryClient=useQueryClient();
	const deletePost=useMutation(
		mutationFn:(id)=> axios.delete('url/id')
		mutationOptions:{
			// config
			 // what to do on sucess
			onSuccess: (api_data,id)=>{
				// we have acees to prv-data-state
				queryClient.setQueryData(["post"],(posts)=>{
					return posts?.filter(post=>post.id!=id)
				})
			};
			 // what to do if error occurs
			onError: ()=>null;
			// runs regard-less success or failure
			onSettled: ()=>null; 
			// unique-key to identify mutation in cache 
			mutationKey: [];
		}
	)
	return (
		<button onClick={deletePost.mutate(id)}>Delete</button>
	)
}
```