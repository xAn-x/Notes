ZOD is a TypeScript-first **schema declaration and validation library**.  It helps you **define the shape of your data**, ensuring that it conforms to your expectations. This leads to more robust and predictable code, especially when dealing with external data sources like APIs or user inputs.

> [!Note]
 
 TypeScript primarily provides _static type checking_, meaning it verifies types during compilation. ZOD, on the other hand, performs _runtime validation_. It checks the actual data values at runtime to ensure they conform to the defined schema.
 
Why we need it:

* **Type Safety:**  It allows you to define a schema and then use it to validate data, ensuring that the validated data matches the expected TypeScript type. This prevents runtime errors caused by unexpected data types.

* **Data Validation:** ZOD provides a powerful way to validate data against a defined schema. Catch errors early, preventing them from propagating through your application.  It supports various validation rules, including type checks, custom functions, and more.

* **Improved Code Maintainability:** Schemas clearly define the expected data structure, making it easier to understand how data is used in your application.

Example (using a simple schema):

```tsx
import {z} from 'zod';
import {useEffect,useState} from 'react';

// this will be used by tsc for static checks
type Product={
	id:string;
	name:string;
	price:number;
}

// this will be used on the values @ runtime to validate them
type productSchema=z.object({
	id:z.string().length(12,{"message":"Incorrect id"}),
	name:z.string(),
	price:z.int().positive({"message":"Invalid Price"})
})

// rather than redefining type every time for static checks
// we can directly infer it from zod object
type Product=z.infer<typeof productSchema>;

export default function Product(){
	const [product,setProduct]=useState<Product|null>(null);
	
	useEffect(()=>{
		fetch("/api/product")
		.then((res)=>res.json())
		.then((product:Product)=>{
			// Method-1
			try{
				// only get parsed if valid else throw error
				const validatedProduct=productSchema.parse(product)
				const setProduct(validatedProduct.data)
			} catch(error){
				alert(error.message)
			}
			
			// Method-2
			// or use safeParse->if parsing fails it return error
			const validatedProduct=productSchema.parse(product);
			if (!validatedProduct.success){
				alert(validatedProduct.error)
			}else{
				setProduct(validatedProduct.data)
			}
		});
	},[]);
	
	{product && return <div>...Product Details</div>}
	return <div>Loading Product Details....</div>
}
```

---

```typescript
import { z } from 'zod';

// Basic types
const stringSchema = z.string();
const numberSchema = z.number();
const booleanSchema = z.boolean();
const dateSchema = z.date();

// Objects
const user = z.object({
  name: z.string(),
  age: z.number().int().positive(),
  email: z.string().email(),
  isActive: z.boolean().default(true), // Default values
  createdAt: z.date(),
});

// Arrays
const stringArray = z.array(z.string());

// Unions
const stringOrNumber = z.union([z.string(), z.number()]);

// Literals
const status = z.literal('active');

// Enums
const Color = z.enum(['red', 'green', 'blue']);

// Optionals
const optionalString = z.string().optional();

// Nullable types
const nullableString = z.string().nullable();

// Custom validation with .refine()
const password = z.string().min(8).refine((val) => /[A-Z]/.test(val), {
  message: 'Password must contain at least one uppercase character',
});

// Transforming data with .transform()
const uppercaseName = z.string().transform((val) => val.toUpperCase());


// Parsing and validation
try {
  const parsedUser = user.parse({
    name: 'John Doe',
    age: 30,
    email: 'john.doe@example.com',
    createdAt: new Date(),
  });
  console.log(parsedUser);

  const parsedArray = stringArray.parse(['a', 'b', 'c']);
  console.log(parsedArray);

  const parsedUnion = stringOrNumber.parse(123);
  console.log(parsedUnion);

  const parsedStatus = status.parse('active');
  console.log(parsedStatus);

  const parsedColor = Color.parse('red');
  console.log(parsedColor);

  const parsedPassword = password.parse('SecurePassword123');
  console.log(parsedPassword);

    const transformedName = uppercaseName.parse("john doe");
    console.log(transformedName);


} catch (error) {
    console.error(error)
}

// Inferring types
type User = z.infer<typeof user>; // User type is automatically inferred


```
---

### Advance E.g. :

```typescript
import { z } from 'zod';

// Define a schema for product categories, with a recursive structure to allow for subcategories
const CategorySchema = z.lazy(() =>
  z.object({
    id: z.string().uuid(),
    name: z.string().min(3),
    // Recursive reference
    subcategories: z.array(CategorySchema).optional(), 
  })
);


// Schema for product variants, including size, color, and price adjustments
const VariantSchema = z.object({
  size: z.enum(['S', 'M', 'L', 'XL']).optional(),
  color: z.string().regex(/^#?([a-f0-9]{6}|[a-f0-9]{3})$/i), // Hex color code validation
  priceAdjustment: z.number().min(-100).max(100), // Price adjustment in percentage
});


// Main product schema
const ProductSchema = z.object({
  id: z.string().uuid(),
  name: z.string().min(3),
  description: z.string().min(10),
  price: z.number().positive(),
  category: CategorySchema, // Nested category object
  variants: z.array(VariantSchema),
  tags: z.array(z.string()).min(1), // At least one tag required
  releaseDate: z.date().max(new Date(), { message: "Release date cannot be in the future" }), // Custom validation message
  available: z.boolean().default(true),
});


// Example product data
const productData = {
  id: 'a1b2c3d4-e5f6-7890-1234-567890abcdef',
  name: 'Awesome T-Shirt',
  description: 'A high-quality t-shirt made of 100% cotton.',
  price: 29.99,
  category: {
    id: 'f1e2d3c4-b5a6-9870-4321-0987654321fe',
    name: 'Clothing',
    subcategories: [
      {
        id: 'g2h3i4j5-k6l7-m8n9-o0p1-q2r3s4t5u6v',
        name: 'T-Shirts',
      },
    ],
  },
  variants: [
    { size: 'S', color: '#FF0000', priceAdjustment: 0 },
    { size: 'M', color: '#00FF00', priceAdjustment: 5 },
  ],
  tags: ['clothing', 't-shirt', 'cotton'],
  releaseDate: new Date('2023-01-01'),
};


try {
  // Parse and validate the product data against the schema
  const validatedProduct = ProductSchema.parse(productData);
  console.log(validatedProduct);

} catch (error) {
  // Handle validation errors
  console.error(error);
}

// Infer the type from the schema
type Product = z.infer<typeof ProductSchema>; // You can now use the Product type
```


This example demonstrates:

* **Complex object schemas:**  Nested objects and arrays.
* **Recursive schemas:**  `z.lazy()` for defining recursive structures like categories/subcategories.
* **Advanced validations:**  Regex, custom validation functions with `.refine()`, and custom error messages.
* **Default values:**  Using `.default()` to set default values for fields.
* **Type inference:**  `z.infer` to get the TypeScript type from the schema.
* **Enums, literals, unions:**  For more specific type constraints.
* **Practical use case:** A more realistic product data structure.
