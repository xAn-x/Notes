### `Async Handler:`
A function or method that is designed to handle asynchronous operations.

```js
const asyncHandler = async (requestHandler) => {
	return (req,resp,next)=>{
		try{
			const res=await requestHandler(req,resp,next);
			return res
		} catch(err){
			console.log(`Request failed :${error.message}`)
			return error
		}
	}
}
```

### `Creating a Controller:`
A controller handles all the functionality for a specific use-case/application (eg user)

```js
import asyncHandler from "./utils/asyncHandler"
import User from "./models/user.model.js"

const registerUser= asyncHandler(async (req,res)=>{
	res.status(200).json{
		// get user details from body
		const user=req.body;
		// validate user-data
		if([user.username,user.email].some(field=>field?.trim()==="")){
			return res.status(400).json({
				message:"All fields are required.",
				success:false
			})
}

		// chk if user already exist
		const if_exist=await User.findOne({
			$or : [{user.username},{user.email}]
		})
		if (if_exist){
			return res.status(403).json({
				message:"User already exist.",
				success:false
			})
		}
		//else register user
		const new_user=await new User(user)
		new_user.save()
		return res.status(201).json({
			message:"User registered succesfully",
			success:true
		})
	}
})

const loginUser= asyncHandler(async (req,res)=>{
	// chk for user data
	const {username,password}=req.body
	//validate field
	if([username,password].some(field=>field?.trim()==="")){
		return res.status(400).json({
			message:"All fields are required",
			success:false
		})
	}
	// chk if user we have user in DB
	const exist=await User.findOne({username})
	if(!exist){
		res.status(404).json({
			message:"User does not exist",
			success=False
		})
	} 
	return res.status(200).json({
		message:"User login sucessfull",
		sucess:true,
		data:exist
	})
})

export {registerUser,loginUser}
```


### `Creating and Exporting a Router:`

Router help us handling all the `api-endpoints` for a specific routes
We can create a router and then export it to use in our main file, this helps in better code management as all endpoints related to a service is present at a specific file.

```js
// Creating a Router
import {Router} from "express"
const router=Router()

// Define all Routes
router.route("/register").post(registerUser)
router.route("/login").post(loginUser)

export default router
```

```js
// Importing router in main file
import express from "express"

const app=express()

// Importing routers
import userRouter from "./routes/user.router.js"

// To handle router we need to use them as a middleware
app.use("/api/users",userRouter) 

app.listen(port,()=>console.log(`Listening to port ${port}`))
```