### 1. String Functions:

```plsql
-- 1. length() : Returns the length of a string.
SELECT length('Hello, world!'); -- Returns 13

-- 2. lower() : Converts a string to lowercase.
SELECT lower('Hello, World!'); -- Returns 'hello, world!'

-- 3. upper() : Converts a string to uppercase.
SELECT upper('Hello, World!'); -- Returns 'HELLO, WORLD!'


-- 4. substring() : Extracts a substring from a string.
SELECT substring('Hello, world!', 1, 5); -- Returns 'Hello'  (Starts at position 1, length 5)
SELECT substring('Hello, world!' from 7 for 5); -- Returns 'world' (Starts at position 7, length 5)


-- 5. trim() : Removes leading and trailing whitespace from a string.
SELECT trim('   Hello, world!   '); -- Returns 'Hello, world!'

-- 6. ltrim() : Removes leading whitespace from a string.
SELECT ltrim('   Hello, world!   '); -- Returns 'Hello, world!   '

-- 7. rtrim() : Removes trailing whitespace from a string.
SELECT rtrim('   Hello, world!   '); -- Returns '   Hello, world!'


-- 8. replace() : Replaces occurrences of a substring with another substring.
SELECT replace('Hello, world!', 'world', 'PostgreSQL'); -- Returns 'Hello, PostgreSQL!'


-- 9. concat() : Concatenates two or more strings.
SELECT concat('Hello', ', ', 'world!'); -- Returns 'Hello, world!'


-- 10. left() : Extracts a specified number of characters from the left side of a string.
SELECT left('Hello, world!', 5); -- Returns 'Hello'


-- 11. right() : Extracts a specified number of characters from the right side of a string.
SELECT right('Hello, world!', 5); -- Returns 'world'


-- 12. position() : Finds the starting position of a substring within a string.
SELECT position('world' in 'Hello, world!'); -- Returns 8

-- 13 concat(str1,str2,...): Concat all the strings
SELECT concat('Hello',', ','World') -- Returns 'Hello, World'

-- 14 concat_ws(sep,str1,str2,...): Concat with seprator
SELECT concat_ws(',','Deepanshu','Bhatt','SWE') -- 'Deepanshu,Bhatt,SWE'
```

### 2. Array Functions:

```plsql
-- 1. array_append(anyarray, anyelement) : Adds an element to the end of an array.

SELECT array_append('{1,2,3}', 4); -- Returns '{1,2,3,4}'


-- 2. array_cat(anyarray, anyarray) : Concatenates two arrays.

SELECT array_cat('{1,2,3}', '{4,5}'); -- Returns '{1,2,3,4,5}'


-- 3. array_dims(anyarray) : Returns the dimensions of an array.

SELECT array_dims(ARRAY[[1,2],[3,4]]); -- Returns "1:2,1:2"


-- 4. array_length(anyarray, int) : Returns the length of an array in a given dimension.

SELECT array_length(ARRAY[1,2,3], 1); -- Returns 3


-- 5. array_lower(anyarray, int) : Returns the lower bound of an array in a given dimension.

SELECT array_lower(ARRAY[1,2,3], 1); -- Returns 1


-- 6. array_upper(anyarray, int) : Returns the upper bound of an array in a given dimension.

SELECT array_upper(ARRAY[1,2,3], 1); -- Returns 2


-- 7. unnest(anyarray) : Expands an array into multiple rows.

SELECT unnest('{apple,banana,cherry}'); -- Returns each element in a separate row


-- 8. array_to_string(anyarray, delim) : Converts an array to a string.

SELECT array_to_string(ARRAY['apple','banana','cherry'], ', '); -- Returns 'apple, banana, cherry'


-- 9. string_to_array(text, sep) : Converts a string to an array.

SELECT string_to_array('apple,banana,cherry', ','); -- Returns '{apple,banana,cherry}'


-- Example using these functions with a table (replace with your actual table and column):

-- Let's assume you have a table named 'products' with a column 'colors' of type text[] (text array):

 CREATE TABLE products (
     id SERIAL PRIMARY KEY,
     name TEXT,
     colors TEXT[]
 );

 INSERT INTO products (name, colors) VALUES ('Shirt', '{"red","blue"}');

 SELECT name, colors, array_append(colors, 'green') AS new_colors
 FROM products;

SELECT name, unnest(colors) AS color
FROM products;

```

### 3. Date/Time Functions

```plsql
-- 1. now() : Returns the current date and time.
SELECT now();


-- 2. current_date : Returns the current date.
SELECT current_date;


-- 3. current_time : Returns the current time.
SELECT current_time;


-- 4. extract(field FROM source) : Extracts a specific field from a date/time value.

SELECT extract(year FROM now());
SELECT extract(month FROM now());
SELECT extract(day FROM now());


-- 5. date_part(field, source) : Similar to extract(), but with slightly different syntax.

SELECT date_part('year', now());


-- 6. age(timestamp, timestamp) : Calculates the difference between two timestamps.

SELECT age(timestamp '2024-01-01', timestamp '2023-01-01'); -- Returns '1 year'


-- 7. to_char(timestamp, format) : Formats a timestamp as a string.

SELECT to_char(now(), 'YYYY-MM-DD HH24:MI:SS');


-- 8. to_timestamp(text, format) : Converts a string to a timestamp.

SELECT to_timestamp('2024-01-01 10:00:00', 'YYYY-MM-DD HH24:MI:SS');


-- 9. date 'YYYY-MM-DD' :  Creates a date value.

SELECT date '2024-01-01';


-- 10. interval 'value units' : Creates an interval value.

SELECT interval '1 day';
SELECT interval '3 months';
```

### 4. JSON Functions

```plsql
-- PostgreSQL JSON Functions Example

-- These examples use JSON, but JSONB (binary JSON) is often preferred for performance.

-- 1. json_build_object() : Creates a JSON object.

SELECT json_build_object('name', 'John', 'age', 30);


-- 2. json_object('{"key1":"value1", "key2":"value2"}') : Creates a JSON object from a text string.

SELECT json_object('{"name":"John", "age":30}');



-- 3. json_extract_path(json, path_elements) : Extracts a value from a JSON object using a path.

SELECT json_extract_path('{"name": "John", "address": {"city": "New York"}}', 'address', 'city'); -- Returns "New York"


-- 4. json_each(json) : Expands a JSON object into a set of key/value pairs.
-- Requires a FROM clause (often used with a LATERAL JOIN)

SELECT key, value
FROM json_each('{"name": "John", "age": 30}');


-- 5. jsonb_contains(jsonb, jsonb) (for JSONB): Checks if a JSONB document contains another JSONB document.
-- JSONB is generally recommended over JSON for better performance

SELECT jsonb_contains('{"a": 1, "b": 2}', '{"a": 1}'); -- Returns true


-- 6. jsonb_set(jsonb, path, jsonb) (for JSONB):  Replaces a value in a JSONB document at a specified path.

SELECT jsonb_set('{"a": 1, "b": 2}', '{b}', '3'); -- Returns '{"a": 1, "b": 3}'

```

