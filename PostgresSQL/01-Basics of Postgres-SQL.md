**PostgreSQL in Short:**

PostgreSQL (often shortened to Postgres) is a powerful, open-source object-relational database management system (ORDBMS).Highly organized, structured way to store and manage data.  Known for its reliability, data integrity features, and support for advanced SQL features (and beyond). 


**SQL vs. PostgreSQL: Key Differences**

It's important to understand that PostgreSQL *is* a type of SQL database.  The difference isn't one of "versus," but rather of "is a part of." SQL is the *language* used to interact with relational databases, while PostgreSQL is a *specific implementation* of a database system that uses SQL.

| Feature                  | Generic SQL Database                                  | PostgreSQL                                                          |
| ------------------------ | ----------------------------------------------------- | ------------------------------------------------------------------- |
| **Type**                 | Abstract concept; a standard for database interaction | Specific implementation of an ORDBMS                                |
| **Data Types**           | Basic data types (INT, VARCHAR, etc.)                 | Richer set of data types (JSON, arrays, ranges, etc.)               |
| **Extensions**           | Limited extensibility                                 | Highly extensible with user-defined functions, types, and operators |
| **Transactions**         | Supports ACID properties                              | Robust transaction management, including advanced features          |
| **Concurrency**          | Varies depending on implementation                    | Advanced concurrency control mechanisms                             |
| **Open Source**          | Can be open source or proprietary                     | Open-source, with a large and active community                      |
| **Standards Compliance** | Varies; may not fully comply with all SQL standards   | Aims for high SQL standard compliance                               |
| **Cost**                 | Varies greatly                                        | Free (open source)                                                  |


**Postgres vs MySQL**

| Feature                  | PostgreSQL                                                          | MySQL                                                                       |
| ------------------------ | ------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| **Data Types**           | Richer set, including arrays, JSON, ranges                          | Fewer data types, simpler overall                                           |
| **ACID Compliance**      | Stricter ACID compliance                                            | More relaxed ACID compliance in some cases                                  |
| **Standards Compliance** | Generally better SQL standard compliance                            | Can deviate from SQL standards sometimes                                    |
| **Scalability**          | Excellent scalability, handles large datasets well                  | Good scalability, but can become challenging with extremely large datasets  |
| **Extensibility**        | Highly extensible with user-defined functions, types, and operators | Less extensible, although plugins are available                             |
| **Transactions**         | Advanced transaction management features                            | Simpler transaction management                                              |
| **Full-Text Search**     | Built-in, powerful full-text search                                 | Requires separate plugins or extensions                                     |
| **JSON Support**         | Native JSON support with querying capabilities                      | Native JSON support, but querying capabilities are less advanced            |
| **Community & Support**  | Large and active community, good documentation                      | Large community, but support can vary depending on the version and provider |
| **Cost**                 | Free (open-source)                                                  | Free (open-source), but commercial support is available                     |
| **Ease of Use**          | Steeper learning curve                                              | Generally considered easier to learn and use                                |


**In short:**

* **Choose PostgreSQL if:** You need a robust, highly scalable database with strong ACID compliance, advanced features (like JSON support and powerful full-text search), and a commitment to SQL standards.  You're comfortable with a slightly steeper learning curve.  Complex applications and large datasets benefit from PostgreSQL's capabilities.

* **Choose MySQL if:** You need a simpler, easier-to-learn database for smaller to medium-sized applications.  You prioritize ease of use and a large, well-established community.  You might find its relaxed ACID compliance acceptable for your application's needs.  It's often a good choice for web applications and applications where speed of development is paramount.

----

```plsql
-- PostgreSQL Basic Commands Example

-- Database Management

-- Create a database
CREATE DATABASE your_database;

-- listing all db
\list ; --This is a psql meta-command, not standard SQL

-- Connect to a database / similar to use DATABASE db
\c your_database;  -- This is a psql meta-command, not standard SQL

-- Drop a database
DROP DATABASE your_database;


-- Table Management --

-- Create a table
CREATE TABLE User (
    id SERIAL PRIMARY KEY, -- SERIAL === AUTO INCREMENT
    name VARCHAR(255) NOT NULL,
    age SMALL INTEGER DEFAULT 18,
	ADD CONSTRAINT check_age CHECK (age>18),
);

-- Insert data into a table
INSERT INTO User (name, age) 
VALUES 
('Alice', 30), ('Bob', 25);

-- Select data from a table
SELECT * FROM User;

-- Update data in a table
UPDATE User SET age = 31 WHERE name = 'Alice';

-- Delete data from a table
DELETE FROM User WHERE name = 'Bob';

-- Drop a table
DROP TABLE User;


-- Data Manipulation 

-- Selecting specific columns
SELECT name, age FROM User;

-- Filtering data with WHERE
SELECT * FROM User WHERE age BETWEEN 19 AND 25;

-- Ordering data with ORDER BY
SELECT * FROM User ORDER BY age DESC;

-- Grouping data with GROUP BY
SELECT age,COUNT(*) FROM User GROUP BY age;

-- Limiting results with LIMIT
SELECT * FROM User LIMIT 2;

-- Altering table and table data
ALTER TABLE User 
ADD COLUMN email default concat(User.name,'@gmail.com')

-- drop
ALTER TABLE User
DROP COLUMN email;

-- modify
ALTER TABLE User
ALTER COLUMN name
SET DATA TYPE VARCHAR(100) 
DROP CONSTRAINT NOT NULL;

--multiple updates
ALTER Table User
MODIFY age INT NOT NULL, Check (age>=18 and age<=60);

-- show column details
SHOW COLUMN User;


-- Copy Table
-- same db
Create TABLE new_tab_name
SELECT col1,col2 from old_table_name;

-- different db
Create TABLE new_db.new_tab_name
SELECT col1,col2 from old_db.old_table_name;

-- Copy With Contraints
Create TABLE new_table_name
LIKE old_table_name;

-- CASE: Conditional
SELECT name,age, 
CASE
	WHEN age >= 60 THEN 'Senior'
	ELSE 'Junior'
END as 'AGE CLASS'
FROM User ;



-- Commit the transaction (save changes)
COMMIT;

-- Rollback the transaction (discard changes)
ROLLBACK;


-- Other useful commands

-- Show all tables in the current database
\dt  -- psql meta-command

-- Show the structure of a table
\d table_name  -- psql meta-command

-- Explain a query's execution plan
EXPLAIN SELECT * FROM User;
```


### Relationships:

Relationships in SQL describe how different tables are connected to each other.  The most common types of relationships are:

1. **One-to-One:** One row in table A is related to at most one row in table B, and vice-versa.  Example: A person has one passport, and a passport belongs to one person.

2. **One-to-Many (or Many-to-One):** One row in table A can be related to multiple rows in table B, but each row in table B is related to only one row in table A. Example: A customer can place many orders, but each order belongs to only one customer.

3. **Many-to-Many:** Rows in table A can be related to multiple rows in table B, and vice-versa. Example: Students can enroll in many courses, and courses can have many students.


Here's a SQL file demonstrating these relationships in PostgreSQL:

```plsql
-- 1. One-to-One Relationship (Example: Persons and Passports)

CREATE TABLE persons (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255)
);

CREATE TABLE passports (
    id SERIAL PRIMARY KEY,
    passport_number VARCHAR(255) UNIQUE,
    person_id INTEGER UNIQUE REFERENCES persons(id)  -- Enforces one-to-one
);


-- 2. One-to-Many Relationship (Example: Customers and Orders)

CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255)
);

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    order_date DATE,
    customer_id INTEGER,
    FOREIGN KEY (customer_id) REFERENCES customers(id) 
);


-- 3. Many-to-Many Relationship (Example: Students and Courses)
-- Requires a junction table (student_courses)

CREATE TABLE students (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255)
);

CREATE TABLE courses (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255)
);

CREATE TABLE student_courses (  -- Junction table
    student_id INTEGER REFERENCES students(id),
    course_id INTEGER REFERENCES courses(id),
    PRIMARY KEY (student_id, course_id) -- Composite primary key
);



-- Inserting some sample data (adapt as needed)

INSERT INTO persons (name) VALUES ('Alice'), ('Bob');
INSERT INTO passports (passport_number, person_id) VALUES ('P123', 1), ('P456', 2);

INSERT INTO customers (name) VALUES ('Charlie'), ('David');
INSERT INTO orders (order_date, customer_id) VALUES ('2024-01-01', 1), ('2024-02-01', 1), ('2024-03-01', 2);

INSERT INTO students (name) VALUES ('Eve'), ('Frank');
INSERT INTO courses (title) VALUES ('Math'), ('Science');
INSERT INTO student_courses (student_id, course_id) VALUES (1, 1), (1, 2), (2, 1);



-- Queries to demonstrate relationships (adapt as needed)

-- One-to-one: Retrieve person and passport information
SELECT p.name, pa.passport_number
FROM persons p
JOIN passports pa ON p.id = pa.person_id;

-- One-to-many: Retrieve customer and their orders
SELECT c.name, o.order_date
FROM customers c
JOIN orders o ON c.id = o.customer_id;

-- Many-to-many: Retrieve students and their courses
SELECT s.name, c.title
FROM students s
JOIN student_courses sc ON s.id = sc.student_id
JOIN courses c ON sc.course_id = c.id;
```


**Important Functions**

| Category                  | Function Name(s)                                                                                                                      | Short Description                                                                          |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| **String Functions**      | `length()`, `lower()`, `upper()`, `substring()`, `trim()`, `replace()`, `string_agg()`, `reverse()`, `replace()`, `left()`, `right()` | Manipulate strings (length, case conversion, extraction, trimming, substitution)           |
| **Numeric Functions**     | `abs()`, `round()`, `ceil()`, `floor()`, `sqrt()`, `random()`                                                                         | Mathematical operations (absolute value, rounding, square root, random numbers)            |
| **Date/Time Functions**   | `now()`, `current_date`, `current_time`, `date_part()`, `extract()`, `age()`                                                          | Work with dates and times (current date/time, extracting parts, calculating age)           |
| **Aggregate Functions**   | `count()`, `sum()`, `avg()`, `max()`, `min()`, `array_agg()`                                                                          | Perform calculations on sets of rows (count, sum, average, max, min, aggregate into array) |
| **Window Functions**      | `rank()`,dense_rank(), `row_number()`, `lag()`, `lead()`, `sum() OVER (PARTITION BY ...)`                                             | Perform calculations across a set of table rows related to the current row                 |
| **Array Functions**       | `array_append()`, `array_length()`, `unnest()`                                                                                        | Manipulate arrays (add elements, get length, expand array into rows)                       |
| **JSON Functions**        | `json_build_object()`, `json_extract_path()`, `json_each()`, `jsonb_contains()`                                                       | Work with JSON data (create JSON objects, extract values, iterate, check for containment)  |
| **Conditional Functions** | `CASE WHEN ... THEN ... ELSE ... END`, `COALESCE()`, `NULLIF()`                                                                       | Handle conditional logic and NULL values                                                   |

