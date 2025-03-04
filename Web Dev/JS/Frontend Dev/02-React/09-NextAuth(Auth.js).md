## NextAuth.js:

**What it is:**  Authentication library for Next.js applications.  Handles user signup, signin, and session management. Supports various providers (e.g., Google, GitHub, email).

**Key Features:**

* **Provider Agnostic:** Easily integrate multiple authentication methods.
* **Server-Side Rendering (SSR) Friendly:** Securely handles authentication on the server.
* **Session Management:**  Manages user sessions efficiently and securely.
* **API Routes:** Uses API routes for secure authentication handling.
* **Easy Integration:** Minimal setup and configuration.

In Next.js u can build ur backend inside the _`api`_ folder. This folder is specially reserved for backend related things. U can create different routes inside it, the name of the folder will act as endpoint & there should be a _route.js_ file for handling the business logic.

---
### *Setup*

*1. Installation*

```bash
npm install next-auth@beta
```

*2. Get AUTH_SECRET*

```sh
npx auth secret
```

*3.Set Environment Variable*

```env
AUTH_SECRET=".........."
```

*4.Config file*

Help us to export all the required functions and providers from a single place. It should be placed @ the root dir.

```ts
// [src]/auth.ts
import NextAuth from "next-auth"
import Github from "next-auth/providers/github"
import Google from "next-auth/providers/google"
import Credentials from "next-auth/providers/credentials"
import {hash} from "becryptjs"
import User from "@db/userSchema.ts"
import dbConnection form "@utils"

export const { handlers, signIn, signOut, auth } = NextAuth({
  providers: [
	  Github({	
		// if name in env is same as here it will auto-pick
		AUTH_GITHUB_ID:process.env.GITHUB_ID,
		AUTH_GITHUB_SECRET:process.env.GITHUB_SECRET,
	  }),
	  
	  Google({
		// if name in env is same as here it will auto-pick
		AUTH_GOOGLE_ID:process.env.GOOGLE_CLIENT_ID,
		AUTH_GOOGLE_SECRET:process.env.GOOGLE_CLIENT_SECRET
	  }),
	
	  // login using db
	  Credentials({
		  // define field and its type
		  credentials:{
			  email:{type:"email",name:"email"},
			  password:{type:"password",name:"password"}
		  },
		  
		  // define auth logic
		  authorize:async (credentials)=>{
			const username=credentials.username;
			const password=credentials.password;
			
			// eastablish db connection
			dbConnection();
			
			const user=await User.findOne({username});
			
			if(user) {
				throw new Error("User already Exist")
			} 
			
			const hashedPassword=hash(password,12);
			const user=await User.create({
				username,password:hashedPassword
			});
			
			// this user will be used to set cookie by default
			// u can alter it in callbacks
			return user;
		  }
	  })
  ],
  pages:{ 
	  // help u provide your own signin,signout etc.
	  // else will serve their own page @ /auth/signin
	  signin:"/login",
	  signup:"/register"
  },
  callbacks:{
	  // help u alter the default signin,signout and other funcs
	  async jwt({token,user}){
		  //...
		  return token
	  },
	  
	  async session({session,token}){
		  return session
	  },
	
	  async signIn({account,user}){
		  if(account?.provider==='google'){
			  try{
				  // info from signup-page
				  const {username,email,password}=user;
				  await dbConnection();

				  const user=await User.findOne({
					  $or:{email,username}
				  });

				  if (user){
					  throw new Error("User Already Exist")
				  }
				  
					//...rest logic
			  }catch(error){
				  throw new Error("Error while creating user")
			  }
		  }
	  } 
  }
})
```

_Export Handlers from this file_
```ts
// [src]/[api]/[...nextauth]/route.ts

import { handlers } from "@/auth"
export const { GET, POST } = handlers
```

_Create middleware if req._

```ts
// [src]/middleware.ts

export { auth as middleware } from "@/auth"
```

---

### *Frontend*

```tsx
//[src]/[app]/(auth)/[login]/page.tsx

import {signIn,auth} from "@/auth.ts";
import {redirect} from "@/next/navigation";

export default function Login(){
	// this will return the cookie if autheticated
	const {session}=await auth();
	const user=session?.user;
	if(user) redirect("/")

	const [user,setUser]=useState({
		username="",password="",email=""
	});
	const router=useRouter();
	
	function handleChange(event){
		setUser({
			...user,
			event.target.name:event.target.value
		})
	}
	
	async function login(formData:FormData){
		"use server"
		await signIn("credentials",formData);
	}
	
	<form action={login}>
			<div>
			<label htmlFor="username">Email</label>
			<input type="text" name="username" id="username"  
			 value={user.username} onChange={handleChange}/>
			</div>
			
			<div>
			<label htmlFor="password">Email</label>
			<input type="password" name="email" id="password" 
			 value={user.password} onChange={handleChange}/>
			</div>
			
			<input type="submit">Submit</input>
			<Link href="/register">Create Account</Link>
			
			<hr/>
			
			<form action={async ()=>{
				"use server"
				await signin("github",)
			}}>
				<button type="submit">Github</button>
			</form>
			
			<form action={async ()=>{
				"use server"
				await signin("google")
			}}>
				<button type="submit">Github</button>
			</form>
		</form>
}
```