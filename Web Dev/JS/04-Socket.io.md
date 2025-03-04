Socket.IO enables real-time, bidirectional communication between web clients and servers. It's built on top of the WebSocket protocol, offering additional features like fallback mechanisms for older browsers and automatic reconnection handling.  It simplifies building interactive applications like chat apps, real-time dashboards, and collaborative tools.

### Terminology:

1. **Sockets**
  - A socket is one endpoint of a two-way communication link between two programs running on a network.
  - Each socket has an id to identify it on a network.
  - Sockets are identified by an IP address and a port no.
  - Different types of sockets exist (e.g., TCP, UDP) offering different levels of reliability and performance
2. **I/O**
   - I/O refers to the communication between a computer and the outside world, including devices, networks, and users. It's how data gets into and out of a computer.
   - I/O operations involve sending and receiving data over a network connection, often using sockets. When a program _writes_ data to a socket, it's sending data out. When it _reads_ from a socket, it's receiving data

In Socket.IO, `emit` and `on` are the core methods for sending and receiving data, forming the basis of real-time communication.

*   **`emit()`**:  Sends a message (event) with optional data to the server or other clients. Think of it as "emitting" a signal.

    * On the client:  `socket.emit('eventName', data)` sends an event named `eventName` with the associated `data` to the server.
      
    * On the server:  `socket.emit('eventName', data)` sends the event to the specific client connected through that `socket`. `io.emit('eventName', data)` broadcasts the event to *all* connected clients.

*   **`on()`**: Listens for incoming messages (events).  It sets up a listener that triggers a callback function when a specific event is received.

    * On the client:  `socket.on('eventName', (data) => { ... })` listens for the `eventName` from the server. When received, the provided callback function executes, with `data` being the data sent by the server.
      
    * On the server: `socket.on('eventName', (data) => { ... })` listens for the `eventName` from a specific client.  `io.on('connection', (socket) => { socket.on('eventName', (data) => { ... }) })` listens for the event from any client after they connect.

- **`broadcast`:**

	- **Purpose:** Sends a message to all connected clients _except_ the sender. Imagine making an announcement in a room.
	- **Usage:** `socket.broadcast.emit('eventName', data)`.

- **`to()`:**

	- **Purpose:** Sends a message to all clients in a specific room. Like addressing a specific group within a larger audience.
	- **Usage:** `io.to('roomName').emit('eventName', data)`. You first get the `io` object (representing the server), then use `to()` to target a room.

- **`join()`:**

	- **Purpose:** Adds the current socket to a specific room. A socket can join multiple rooms.
	- **Usage:** `socket.join('roomName')`. This is how you group sockets together for targeted communication.


```sh
// client
npm install socket.io-client

// server
npm install socket.io
```

### *Server(Node.js)*

```js
import { Server } from "socket.io";
import express from "express";
import {createServer} from "http";
import cors from "cors";

// for cross-origin connection (frontend and backend on differnt host)
app.use(cors())

 

const app=express();
const PORT=3000;

const server=createServer(app);
const io=new Server(server,{
	cors:{
		origin:"frontend-url",
		methods:["GET","POST"], // methods to allow
		credentials:true,
	}
});

app.get("/",(req,res)=>{
	res.send("Socket.io")
})

// whenever any socket/client is connected 
io.on("connection",(socket)=>{
	console.log("User Connected");
	console.log("ID: ",socket.id);
})

socket.on("send-message",(data)=>{
	let {message,roomID}=data;
	// send the data to particular room/socket
	// when using to() -> socket.to() or io.to() is same
	io.to(roomID).emit("receive-message",message)
	
})

// make user join a room
socket.on("join-room",(data)=>{
	socket.join(data.room)
})

 
// use server.listen to handle websocket connection
// this will handle for app too 
server.listen(PORT,()=>{
	console.log(`Server Running on PORT ${PORT}`)
	socket.emit("welcome","Welcome to server")
});
```

### *Frontend (React)*

```jsx

import {io} from "socket.io-client";
import {useEffect,useState,useMemo} from "react";

const RenderMessages=(messages)=>{
	<div>
		{messages.map(message)=>{
			 <div>message</div>
		 }}
	</div>
}

const app(){
	const socket=io("backend-url");
	const [messages,setMessages]=useState([]);
	const [room,setRoom]=useState(null);
	useEffect(
		socket.emit("connect",()=>{
			console.log(`Connected to server, ID:${socket.id}`)
		)
		
		socket.on("receive-message",(message)=>messages.push(message))

		return ()=>{
			socket.disconnect();
		}
	},[])

	
	const handleSubmit=(event)=>{
		event.preventDefault();
		socket.emit("send-message",{message,roomID})
	}

	const handleJoinRoom=(event)=>{
		event.preventDefault();
		socket.emit("join-room",{room})
	}

	return (
		<RenderMessages messages={messages}/>
		<hr/>
		<form onSubmit={handleSubmit}>
			<input type="text" placeholder="send message" value={message} 
				onChange={e=>setMessage(e.target.value)}/>
			<button type="submit">Send</button>
		</form>
		// make user join some room
		<form onSubmit={handleJoinRoom}>
			<input type="text" placeholder="Room Name" value={message} 
				onChange={e=>setRoom(e.target.value)}/>
			<button type="submit">Join Room</button>
		</form>
	)
}
```