
**Access Token**

* A short-lived token (typically valid for a few minutes to an hour) that grants access to protected resources.
* Used for authenticating requests to an API or other protected endpoint.
* Can be revoked or expire, requiring the user to re-authenticate.

**Refresh Token**

* A longer-lived token (typically valid for days or weeks) that can be used to obtain a new access token.
* Stored securely on the backend server.
* Used when the access token expires or is revoked.

**Usage in Backend Development**

In backend development, access tokens and refresh tokens are typically used together:

1. **User Authentication:** When a user logs in, the backend server issues an access token and a refresh token.
2. **API Access:** The access token is used to authenticate requests to the API.
3. **Token Expiration:** When the access token expires, the client application sends a request to the backend server with the refresh token.
4. **New Access Token:** The backend server verifies the refresh token and issues a new access token.

**Advantages of Access Tokens and Refresh Tokens**

* **Improved Security:** Refresh tokens are stored securely on the backend server, reducing the risk of unauthorized access to protected resources.
* **Extended Session Duration:** Refresh tokens allow users to remain authenticated even after their access token expires, providing a seamless user experience.
* **Reduced Server Load:** By using refresh tokens, the backend server does not need to handle frequent authentication requests.

**Best Practices**

* Store refresh tokens securely in a database or other secure storage mechanism.
* Set appropriate expiration times for both access tokens and refresh tokens.
* Implement a mechanism to revoke refresh tokens when necessary (e.g., if a user's account is compromised).
* Use a reliable and secure authentication mechanism to protect access to the API.

```js
import mongoose from "mongoose";

const UserSchema = new mongoose.Schema({

  username: {

    type: String,

    unqiue: true,

    required: true,

  },

  email: {

    type: String,

    unqiue: true,

    required: true,

    validate: {

      validator: function (value) {

        return /^(([^<>()[\]\\.,;:\s@"]+(\.[^<>()[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/.test(

          value

        );

      },

      message: "Please enter a valid email",

    },

  },

  password: {

    type: String,

    required: true,

    minLen: [8, "Password should be at least 8 characters long"],

    maxLen: [15, "Password should be at most 15 characters long"],

  },
});

// Usally we make a mongoose method to create tokens
UserSchema.methods.generateRefreshToken = function () {
  return jwt.sign({ _id: this._id }, process.env.REFRESH_TOKEN_SECRET,{
    expiresIn: process.env.REFRESH_TOKEN_EXPIRY,
  });
};

UserSchema.methods.generateAccessToken = function () {
  return jwt.sign({ _id: this._id }, process.env.ACCESS_TOKEN_SECRET,{
    expiresIn: process.env.ACCESS_TOKEN_EXPIRY,
  });
};



// Login-Logic
const loginUser = async function (req,res){

   const {username,email,password}=req.body;

   // validate user
  const user=await User.findOne({$or:[{username}, {email}]});
  if(!user) return res.status(401).json({message:"Invalid Credentials"});

  
  //password check
  const isMatch=await bcrypt.compare(password,user.password);
  if(!isMatch) return res.status(401).json({message:"Invalid Credentials"});

   
  //access and refresh token
  const accessToken=await user.generateAccessToken();
  const refreshToken=await user.generateRefreshToken();

  
  // save refreshToken in db and save user
  user.refreshToken=refreshToken
  await user.save({validateBeforeSave:false}); // false to avoid validation of required fields like password
  
  
  // send cookie
  const options={
	  httpOnly:true,
	  secure:true
  }
  return res.status(200)
	.cookie("refreshToken",refreshToken,options)
	.cookie("accessToken",accessToke,options)
	.json({
		message:"User Login Sucessfull",
		success:true,
	})
}

```