*Folder Structure:

```sh
[src]
	[app] #fronted
		(root)
			page.tsx
		(auth)
			[login]
				page.tsx
			[signup]
				page.tsx

	[api] #backend
		[users]
			[login]
				route.ts
			[signup]
				route.ts

	[models] # db-models --> not necessary to create sep folder 
			 # but gud practice
		userModel.ts

	middleware.ts # -> handle req-resp intercepts
	
```


*Creating User-Schema*
```ts
// [models]/userSchema.ts
import mongoose from "mongoose";

const userSchema=new mongoose.Schema({
	username:{
		type:String,
		required:[true,"Please Provide Username!"],
		unique:true
	},
	email:{
		type:String,
		required:[true,"Please Provide Username!"],
		unique:true
	},
	password:{
		type:String,
		required:[true,"Please Provide Username!"],
		validate:[password=>password.length>=8,"Password is short"]
	},
	isVerified:{type:Boolean,default:false},
	isAdmin:{type:Boolean,default:false},
	forgotPasswordToken:String,
	forgotPasswordTokenExpiry:Date,
	verifyToken:String,
	verifyTokenExpiry:String
});

const User=mongoose.models.users||mongoose.model("users",userSchema)

export default User;
```
---

*DB-Connection*

```ts
//[dbConfig]/dbConfig.ts
import mongoose from "mongoose";

export default async function connect(){
	try{
		mongoose.connect(
			`${process.env.MONGO_URI!}/${DB_NAME}`
		)
		const connection=mongoose.connection;
		connection.on('connected',()=>{
			console.log("DB connected sucessfully!!")
		})
		connection.on('error',(error)=>{
			console.log("Con Error,make sure DB is up and running")
			console.log(error)
			process.exit()
		})
	}catch(error){
		console.log("Something went wrong in connecting to db")
		console.log(error)
	}
}
```

---
*Signup & Login Routes*

```tsx
// [api][signup]/route.ts

import connect from "@dbConfig/dbConfig.ts";
import {User} from "@models/userSchema.ts";
import {NextResponse,NextRequest} from "next/server";
import bcryptjs from "bcryptjs"

// in every route where u need to acess db u need to invoke connect()
// as next runs apps in edge devices
connect()

export async function POST(request:NextRequest){
	try{
		const body=await request.json();
		const {username:String,password:String,email:String}=body;
		
		
		const user=await User.findOne({$or:[username,password]})
		if (user){
			return NextResponse.json.json({error:"User already exist"},
									 {status:400})
		};

		const salt=await bcryptjs.genSalt(10);
		const hashedPassword=await bcryptjs.hash(password,salt)
		
		const newUser=new User({
			username,email,password:hashedPassword
		});
		
		const savedUser=newUser.save();
		
		// send verification mail,u can use nodeMailer 4 that
		
	}catch(error){
		return NextResponse.json.json({error:error.message},
								 {status:500})
	}
};
```

```ts
// [api]/verifyUser/[route.ts]

import connect from "@dbConfig/dbConfig.ts";
import {User} from "@models/userSchema.ts";
import {NextResponse,NextRequest} from "next/server";

connect();
export async function POST(request:NextRequest){
	const reqBody=await request.json();
	const {token:string}=reqBody;
	try{
		const user=await User.findOne({
			verifyToken:token,
			verifyTokenExpiry:{$gt:Date.now()}
		});
		
		if (!user){
			return NextResponse.json({error:"User Verif failed"}
								{status:400}
			);
		}

		// UPDATE USER IN DB
		user.verifyToken=undefined;
		user.verifyTokenExpiry:undefined;
		user.isVerified:true;
		
		await user.save();

		return NextResponse.json({message:"User Verified"});
	} catch(error){
		return NextResponse.json({error:error.message},{status:500});
	}
};
```

```ts
//[api]/[login]/route.ts
import connect from "@dbConfig/dbConfig.ts";
import {User} from "@models/userSchema.ts";
import {NextResponse,NextRequest} from "next/server";
import jwt from "jsonwebtokens";
import bcryptjs from "bcryptjs";

connect();
export async function POST(request:NextRequest){
	const reqBody=await request.json();
	const {email,password}=reqBody;
	try{
		const user=await User.find({email});
		if(!user){
			return NextResponse.json(
				{error:"user does not exist"},{status:400}
			)
		}
		// verify user
		const validPassword=await bcryptjs.compare(
		password,user.password);
		
		if (!validPassword){
			return NextResponse.json(
				{error:"Invalid Password"},{status:400}
			)
		}
		
		// save user as cookie
		const userTokenData={id:user._id};
		const userToken=jwt.sign(
			userTokenData,process.env.TokenSecret,
			{expiresIn:'1d'}
		);

		const response=NextResponse.json(
			{message:"User login sucessfull!"},
			{status:200}
		)

		response.cookies.set("user",userToken,{
			httpOnly:True, // read-only
		});
		return response;
	}catch(err){
		return NextResponse.json({error:err.message},{status:500})
	}
}
```

```ts
//[api]/[logout]/route.ts

import connect from "@dbConfig/dbConfig.ts";
import {User} from "@models/userSchema.ts";
import {NextResponse,NextRequest} from "next/server";
import jwt from "jsonwebtokens";
import bcryptjs from "bcryptjs";

connect();
export async function GET(request:NextRequest){
	try{
		const response=NextResponse.json({
			message:"user logout sucessfull!",
			sucess:true,
		},{status:200});
		// reset the cookie
		response.cookies.set("user","",{
			httpOnly:true,
		})
	}catch(err){
		return NextResponse.json({error:err.message},{status:500})
	}
}
```
---
*Frontend:*

```tsx
//(auth)/[user]/[signin]/page.tsx
'use-client'; 
import axios from axios;
import {useRouter} from "next/navigation"

export default function page(){
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
	
	async function handleSubmit(event){
		event.preventDefault();
		try{
			const resp=await axios.post("baseuri/api/sigin");
			console.log("signup-sucess!!!");
			router.push("/login")
		} catch(error){
			console.log("signup-fail");
		}
	}
	
	return (
		<form onSubmit=handleSubmit>
			<div>
			<label htmlFor="username">Email</label>
			<input type="text" name="username" id="username"  
			 value={user.username} onChange={handleChange}/>
			</div>
			
			<div>
			<label htmlFor="email">Email</label>
			<input type="email" name="email" id="email" 
			 value={user.email} onChange={handleChange}/>
			</div>		
			
			<div>
			<label htmlFor="password">Email</label>
			<input type="password" name="email" id="password" 
			 value={user.password} onChange={handleChange}/>
			</div>
			
			<input type="submit">Submit</input>
			<Link href="/login">Login-Page</Link>
		</form>
	)
}
```