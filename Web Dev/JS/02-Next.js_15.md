Next.js is a popular React framework for building web applications, including static websites, server-rendered pages, and API routes.  It aims for performance and developer experience through features like:

**Key Features:**

* **Server-Side Rendering (SSR):** Improves SEO and initial load time.
* **Static Site Generation (SSG):** Creates fully static HTML at build time, ideal for content-heavy sites.
* **API Routes:** Enables serverless functions within your Next.js app.
* **Image Optimization:** Automatically optimizes images for various devices.
* **Fast Refresh:**  Enables quick development iteration with instant updates.
* **File-system Routing:**  Organizes routes based on your file structure.
* **Data Fetching:**  Provides built-in methods for fetching data (`getStaticProps`, `getStaticPaths`, `getServerSideProps`).
* **Incremental Static Regeneration (ISR):**  Revalidates and updates static pages periodically.
* **Built-in support for TypeScript:** Improves code maintainability and type safety.
* **Deployment flexibility:** Supports various deployment platforms (Vercel, Netlify, AWS, etc.).

#### 1. Routing:

Routing in Next is as simple as creating a folder [route-name] and a file [page.tsx] to serve that route.

#### 2. Server and client side component:

Everything in Next-js is server side render by default for performance optimization and if some component req browser actions that one have to use `'use client'` directive at top of the page.

#### 3. Layout and errors:

* **Layout:** Next-js uses `layout.ts` file for the layout. Any Component You define there will be visible to every page.
  
	We can define `layout.ts` file for each route, but rather than overriding the global layout, the will just append them components for them routes. 

	To have separate layout we have to use _route groups_.Route groups are special directories with name in format: `(folder-name)` these dir don't create routes but seprate them so each can have a different structure.

#### 4. Links and Image Optimisation:

Next-js provide special component like `<Link/>` and `<Image/>` that use lazy-loading for optimization. Only things that are available on the viewport or page to whom the link is available on the page will be loaded, once something get loaded Next also catches them results to avoid re-loading.

#### 5. S
