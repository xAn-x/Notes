 ![[Pasted image 20240316092137.png]]

### `Folder Structure:`

```Folder Structure
[src]
   index.js : connects with db
   [DB] : Db related funcs
   [Models] : Schemas and Models of a DB
   [Controllers] : Functionality 4 processing data
   [Routes] : Api Endpoints 4 route handling
   [Middlewares] : Security (runs before routes)
   [Utils] : helper funcs
```

### `Basic Server in Express-js`

```js
%% npm install express

import express from "express"
const app=express() // a functional instance of express


const port=3000 // port over which our server runs/listen for requests

// handling get req for route:"/" {home-page}
app.get("/",(req,res)=>{
	res.send("<h1>Hello,from server</h1>")
})
 
// asking server to listen on port
app.listen(port,()=>{
	console.log(`Listening to port:${port}`)
})
```

### `Connect Frontend with Backend:`

From your frontend make a request to URL / PORT where the server is listening to (can easily done using fetch or axios) that's it.

``` js
// ========== Backend ==============
import express from "express"
const app=express() /
const port=3000 

let data=[
	{"user":"deepanshu","age":21},
	{"user":"ram","age":23},
	{"user":"rani","age":20},
]

app.get("/api/data",(req,res)=>{
	res.status(200).send(data)
})


// ========== Frontend ==============
import axios from axios
async function fetchData(){
	// Note this will be async req as data is coming from server
	try{
		// Never enter the base url use proxy instead
		const resp=await axios.get("http://localhost:3000/api/data")
		return resp;
	} catch(err){
		console.log(`Some error occured:${err.message}`)
	}
}
let fetchBtn=document.getElementById("fetch-btn")
fetchBtn.addEventListener("click",fetchData)
```

  ==`proxy`==: rather than using `base-url/endpoint` we use proxy, where we mention the base url in the config file and the just mention the end-points this make api dev easy as we can manage the production and dev field quiet easily. Each application has a different way to add proxy (like cra , vite , next-js ) so read about `proxy in package-manager`.It also manages `cors (cross-origin)`.

### `Middleware:`

Middleware refers to software components or functions that are used to handle requests and responses in an application or system. In the context of web development, middleware ==sits between the client and server== and plays a crucial role in processing HTTP requests, performing operations such as authentication, logging, error handling, and data transformation.

Middleware functions are ==executed in the order they are added to the application's middleware stack==. They have access to the request and response objects, and they can modify these objects, pass control to the next middleware function in the stack, or terminate the request-response cycle.

Add functionality to the request-response cycle without modifying the core application logic.

```js
// some important middleware that u always want to use in express-app

% npm i cors cookie-parser

import express from "express"
import cors from cors
import cookieParser from "cookie-parser"

const app=express()

// app.use(middleWare) : are use to apply any middleware

// for parsing json data
app.use(express.json({limit:"16kb"})) 

// for data i.e. coming through url
app.use(express.urlencoded(extended:true,limit:"16kb"))

// for handling cross-origin requests
app.use(cors())

// for performing CRUD oprns on cookie safely
app.use(cookieParser())

// for serving any static files
app.use(express.static("public-folder-path"))
```

### `Important Express concepts:`

1. How to access search-params from its url

```js
	import express from "express"
	require('dotenv').config()
	
	const app=express()
	const PORT=process.env.PORT || 3000

	// :username -> search param
	app.get("/api/user/:username",(req,res,next)=>{
		// To access search parameter
		const {username}=req.params
		// To access query params
		const query_params=req.query
	})
	
	app.listen(PORT)
```

2. How to serve static and Dynamic Html file

```js
	import express from "express"
	require('dotenv').config()

	
	const app=express()
	const PORT=process.env.PORT || 3000

	// serving static-file
	// all static files should be inside this folder
	// by default this folder name is "public" in industry
	app.use(express.static("folder with static files"))
	
	app.get("/api/static",(req,res,next)=>{
		res.sendFile("file-name")
	})

	// serving dynamic html file
	// all dynamic files should be inside this folder
	// by default this folder name is "views"
	app.set("view","./views") 
	app.set("view engine","ejs") // setting up template view engine -> for serving dynamic HTML
	app.get("/api/static",(req,res,next)=>{
		// Inside file acess em as locals.param_name
		res.render("file-name",{param_name:value})
		// dont mention extension name with file
	})
	
	app.listen(PORT)
```

3. U can use `express-generator` to create the folder structure

```js
% npm i express-generator -g
% express <<app-name>> --view=<<view-engine>>
% cd <<app-name>>
% npm i 
% npx nodemon 
```

4. Cookies

```js
// Cookies are stored in browser and store some info about the user
// Cookies, auto send in each req made by client so that can use used by the servevr to extract usefull info

% npm install cookie-parser //package for performing crud oprn on cookies

import express from "express"
import CookieParser from "cookie-parser"

const app=express()

// use this as a middleware
// secret-code is required if u want to have signed cookies
app.use(CookieParser("secret-code"))

app.get("/api/cookie-crud",(req,res,next)=>{
	// getting all cookies 
	console.log(`Cookies:${req.cookies}`)
	// Cookies that have been signed
	// signed cookies are cookies that are encrypted for security
	console.log('Signed Cookies: ', req.signedCookies)
	
	// setting or updating cookie
	res.cookie("cookie-name","value"|{json_obj},{
		maxAge:"time in mill-se",
		expires:date,
		httpOnly:true,
		signed:true,
		etc...
	})
	
	// Deleting Cookies
	res.clearCookie("cookie-name",{options})
})
```

4. Session

```js
// used to store user-specific data across multiple requests.

// They are often implemented using session middleware such as `express-session` along with session stores like `express-session-mongo` for storing session data in MongoDB.

% npm install express-session express-session-mongo

import express from "express"
import session from "express-session"
import MongoStore from "express-session-mongo"

const app=express()

// congiguring session middleware
app.use(session({ 
name:"session-name",
secret: 'your_secret_key', // Secret key for session encryption
resave: false, // Don't save session if not modified 
saveUninitialized: false, // Don't save new sessions with no data
store: new MongoStore({ url: 'mongo-db-url' }),
// MongoDB session store 
cookie: { secure: false, maxAge: 86400000 } // Cookie options (secure: true for HTTPS) 
// cookie here is used to maintain session state and user authentication across multiple requests. It stores a unique session ID that is associated with server-side session data.
}));

// note: here everything is controlled using "req"
app.get("/api/session-oprns",(req,res,next)=>{
	// setting and updating data in session
	req.session["session-name"]["prop"]=value
	
	// accessing session data
	console.log(req.session)
	
	// deleting a session
	req.session["session-name"].destroy((err)=>{
		if(err){
			console.log(`Error ${err}`)
			res.status(500).send("Error while deleting session")
		}
		else{
			res.send("session deleted sucessfully")
		}
	})
})
```

5. Cookie vs Session

| Aspect           | Cookie                                                   | Session                                                   |
| ---------------- | -------------------------------------------------------- | --------------------------------------------------------- |
| **Purpose**      | Used for storing data on the client side                 | Used for maintaining session state on the server side     |
| **Lifespan**     | Can have long or short lifespan, based on settings       | Typically shorter lifespan (e.g., session duration)       |
| **Content**      | Can store various types of data (e.g., preferences, IDs) | Stores session-specific data (e.g., user authentication)  |
| **Security**     | May or may not have additional security measures         | Often includes security measures (e.g., httpOnly, secure) |
| **Transmission** | Sent with every HTTP request and response                | Session ID transmitted instead of actual data             |
| **Storage**      | Stored on the client side                                | Session data stored on the server side                    |
