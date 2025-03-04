Whenever we want to store some sort of file in server we want to have a robust backend to handle it.

Most of the time we don't store file in our server we use 3rd-party services suck as `AWS` or `cloudinary` to store file and then just store the URL for that file so can fetch it whenever required.

```js
% npm i cloudinary
// utils > cloudinary.js
import {v2 as cloudinary} from "cloudinary"
import fs from "fs"

// setting config for cloudinary
cloudinary.config({
	cloud_name: cloudinary_cloud_name,
	api_key: cloudinary_api_key,
	api_secret: cloudinary_api_secret
})


let uploadOnCloudinary = async function (localFilePath){
	try{
		if (!localFilePath) throw new Error("File not found..")
		//upload the file from server to cloudinary
		const resp=await cloudinary.uploader.upload(localFilePath,{
			resource_type:"auto"
		})
		//file has been uploaded successfully
		console.log(`File uploaded Succesfully ${resp.url}`)
		return resp;
	} catch(error){
		//if some error occur , remove the file from server
		fs.unlinkSync(localFilePath)
		return error
	}
}

export {uploadOnCloudinary}
```

==Note==: We use `multer` as middleware upload file in our cloud platform. [Multer](https://github.com/expressjs/multer) is a middleware for handling multipart/form-data, which is primarily used for uploading files in Node.js applications. It is commonly used with frameworks like Express to handle file uploads. `Multer` allows you to easily handle file uploads, including validation, saving files to disk or cloud storage, and processing file data before storing it. It simplifies the process of handling file uploads in Node.js web applications. 

```js
% npm install multer

import multer from "multer"

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, destnPath)
  },
  filename: function (req, file, cb) {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9)
    // cb(null, file.fieldname + '-' + uniqueSuffix)
    cb(null, file.originalname)
  }
})

const uploadSingleFile = multer({ storage: storage })

export {uploadSingleFile}
```