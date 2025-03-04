****### 1. Joins

**Joins** are fundamental operations in SQL used to combine rows from two or more tables based on a related column between them.  The different types of joins determine which rows are included in the result set.

Here's an explanation and a SQL file illustrating the various join types in PostgreSQL:

**Types of Joins:**

1. **INNER JOIN:** Returns rows only when there is a match in *both* tables based on the join condition.

2. **LEFT (OUTER) JOIN:** Returns *all* rows from the left table (the one specified before `LEFT JOIN`), and the matching rows from the right table. If there's no match in the right table, `NULL` values are returned for the right table's columns.

3. **RIGHT (OUTER) JOIN:** Returns *all* rows from the right table, and the matching rows from the left table.  If there's no match in the left table, `NULL` values are returned for the left table's columns.

4. **FULL (OUTER) JOIN:** Returns *all* rows from *both* tables. If there's a match, the corresponding rows are combined. If there's no match on one side, `NULL` values are returned for the other table's columns.

5. **CROSS JOIN:** Returns the Cartesian product of both tables (all possible combinations of rows).  Use with caution as it can generate very large result sets.


```plsql
CREATE TABLE departments (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255)
);

CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    department_id INTEGER REFERENCES departments(id)
);

INSERT INTO departments (name) VALUES ('Sales'), ('Marketing'), ('Engineering');
INSERT INTO employees (name, department_id) VALUES ('Alice', 1), ('Bob', 1), ('Charlie', 2), ('David', NULL); -- David has no department


-- 1. INNER JOIN

SELECT e.name AS employee_name, d.name AS department_name
FROM employees e
INNER JOIN departments d ON e.department_id = d.id;  -- Only Alice and Bob (and Charlie)


-- 2. LEFT JOIN

SELECT e.name AS employee_name, d.name AS department_name
FROM employees e
LEFT JOIN departments d ON e.department_id = d.id; -- All employees, including David with NULL department


-- 3. RIGHT JOIN

SELECT e.name AS employee_name, d.name AS department_name
FROM employees e
RIGHT JOIN departments d ON e.department_id = d.id; -- All departments, including Engineering with no employees


-- 4. FULL JOIN

SELECT e.name AS employee_name, d.name AS department_name
FROM employees e
FULL JOIN departments d ON e.department_id = d.id;  -- All employees and all departments, with NULLs where there's no match


-- 5. CROSS JOIN (Cartesian Product - use with caution!)

SELECT e.name AS employee_name, d.name AS department_name
FROM employees e
CROSS JOIN departments d; -- Every employee paired with every department
```


**Joining More Than Two Tables:**

You can join more than two tables by chaining join clauses together.  The `ON` clause specifies the join condition for each pair of tables being joined.

**Example:**

Let's say you have three tables: `Customers`, `Orders`, and `Products`.

```plsql
CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255)
);

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    order_date DATE
);

CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    price DECIMAL
);

CREATE TABLE order_items (  -- Junction table for many-to-many relationship
    order_id INTEGER REFERENCES orders(id),
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER,
    PRIMARY KEY (order_id, product_id)
);


INSERT INTO customers (name) VALUES ('Alice'), ('Bob');
INSERT INTO orders (customer_id, order_date) VALUES (1, '2024-01-15'), (1, '2024-02-20'), (2, '2024-03-10');
INSERT INTO products (name, price) VALUES ('Product A', 10.00), ('Product B', 20.00);
INSERT INTO order_items (order_id, product_id, quantity) VALUES (1, 1, 2), (1, 2, 1), (2, 2, 3), (3,1,1);


SELECT c.name AS customer_name, o.order_date, p.name AS product_name, oi.quantity
FROM customers c
JOIN orders o ON c.id = o.customer_id
JOIN order_items oi ON o.id = oi.order_id
JOIN products p ON oi.product_id = p.id;

```


**Important Points to Remember about Joins (in PostgreSQL and Standard SQL):**

* **JOIN without ON is equivalent to a CROSS JOIN:**  If you omit the `ON` clause, you'll get a Cartesian product (all possible combinations of rows).

* **Join Order Matters (Sometimes):** The order in which you join tables can affect performance, especially with large tables.  The database's query optimizer usually determines the best execution plan, but in some cases, you might need to experiment with join order for optimization.

* **Parentheses for Clarity:**  When joining multiple tables, use parentheses to explicitly group join operations, especially if you have a mix of `LEFT`, `RIGHT`, or `FULL` joins. This makes the query easier to read and understand.

* **Aliases for Readability:** Use aliases (e.g., `c` for `customers`, `o` for `orders`) to make your queries more concise and readable.

* **Filtering with WHERE:** You can add a `WHERE` clause to filter the results of a join, just as you would with a single table.

* **Outer Joins and NULLs:** Be mindful of `NULL` values when using outer joins (`LEFT`, `RIGHT`, `FULL`).  If there's no match in the joined table, the columns from that table will have `NULL` values.


---
---

### 2. SET Operations

1. **`UNION` (already covered):** Combines the results of two or more `SELECT` statements, removing duplicate rows.

2. **`UNION ALL` (already covered):** Combines the results, keeping all rows (including duplicates).

3. **`INTERSECT`:** Returns only the rows that are common to *both* `SELECT` statements (the intersection of the sets).

4. **`EXCEPT` (or `MINUS` in some databases, but not PostgreSQL):** Returns the rows that are in the first `SELECT` statement but *not* in the second `SELECT` statement (the difference of the sets).

5. **`EXCEPT ALL`:** Similar to `EXCEPT`, but keeps all rows from the first `SELECT` statement, even if they appear multiple times, as long as they are not present in the second `SELECT` statement. (PostgreSQL-specific).


**Example in PostgreSQL:**

```plsql
-- PostgreSQL Set Operations Example

-- Sample tables (same as previous example)
CREATE TABLE table1 (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    age INTEGER
);

CREATE TABLE table2 (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    age INTEGER
);

INSERT INTO table1 (name, age) VALUES ('Alice', 10), ('Bob', 20), ('Charlie', 30);
INSERT INTO table2 (name, age) VALUES ('Bob', 20), ('David', 40), ('Eve', 50);


-- 1. UNION (removes duplicates)

SELECT name, age FROM table1
UNION
SELECT name, age FROM table2;  -- Bob appears only once


-- 2. UNION ALL (keeps duplicates)

SELECT name, age FROM table1
UNION ALL
SELECT name, age FROM table2; -- Bob appears twice

-- 3. INTERSECT

SELECT name, age FROM table1
INTERSECT
SELECT name, age FROM table2; -- Only Bob (common to both)


-- 4. EXCEPT (Difference)

SELECT name, age FROM table1
EXCEPT
SELECT name, age FROM table2;  -- Alice and Charlie (in table1, not in table2)

SELECT name, age FROM table2
EXCEPT
SELECT name, age FROM table1;  -- David and Eve (in table2, not in table1)



-- 5. EXCEPT ALL (Keeps duplicates in the first SELECT)

INSERT INTO table1 (name, age) VALUES ('Bob', 20); -- Add another Bob to table1

SELECT name, age FROM table1
EXCEPT ALL
SELECT name, age FROM table2;  -- Alice, Charlie, and both Bobs


```

**Important Rules (same as UNION/UNION ALL):**

* Same number of columns.
* Compatible data types.
* Column names from the first `SELECT`.
* `ORDER BY` at the end (if needed).

---
---
### 3. Group By

They are used for aggregating and filtering data.  You can use them within PL/pgSQL, but their behavior and syntax are the same as in regular SQL queries.

**1. `GROUP BY` Clause:**

* **Purpose:** The `GROUP BY` clause groups rows with the same values in specified columns into a summary row.  This is useful for performing aggregate functions (like `COUNT`, `SUM`, `AVG`, `MAX`, `MIN`) on each group.
* **Syntax:**  `GROUP BY column1, column2, ...`
* **How it Works:**  The `GROUP BY` clause divides the rows into groups based on the values in the specified columns.  Then, aggregate functions can be used to calculate values for each group.

**2. `HAVING` Clause:**

* **Purpose:** The `HAVING` clause filters the results *after* the `GROUP BY` clause has been applied.  It filters based on the aggregated values (the results of the aggregate functions).
* **Syntax:** `HAVING condition`
* **How it Works:**  The `HAVING` clause is applied to the grouped results.  Only the groups that satisfy the `HAVING` condition are included in the final result set.

**When to Use `GROUP BY` and `HAVING`:**

* **Summarizing Data:** Use `GROUP BY` to summarize data by different categories or groups.  For example, you might want to find the average salary for each department, the total sales for each region, or the number of orders for each customer.
* **Filtering Aggregated Data:** Use `HAVING` to filter the summarized data **based** on the aggregated values. For example, you might want to find only the departments with an average salary greater than $50,000 or the regions with total sales less than $1 million.

**Example in PostgreSQL:**

```sql
-- Sample table
CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    department VARCHAR(255),
    salary INTEGER
);

INSERT INTO employees (department, salary) VALUES
('Sales', 50000), ('Sales', 60000), ('Marketing', 70000),
('Marketing', 80000), ('Engineering', 90000), ('Engineering', 100000);


-- Using GROUP BY and HAVING

SELECT department, AVG(salary) AS average_salary
FROM employees
GROUP BY department
HAVING AVG(salary) > 75000; -- Filter groups with average salary > 75000

-- Clean up (optional)
-- DROP TABLE employees;
```

**Key Differences between `WHERE` and `HAVING`:**

* **`WHERE` filters rows *before* grouping.** It operates on individual rows.
* **`HAVING` filters groups *after* grouping.** It operates on the aggregated results.

---
---

### 4. Views 

**Views in SQL (PostgreSQL):**

* A view is a virtual table based on the result-set of an SQL statement.  It doesn't store data itself but provides a customized or simplified way to access data from underlying tables (It just run the query defined under the view-name).

* **Why use them?**
    * **Simplify complex queries:**  Hide complex joins or calculations behind a view, making queries easier to write and understand.
    * **Data security:** Restrict access to specific columns or rows in underlying tables by creating views that only expose necessary data.
    * **Data consistency:** Provide a consistent view of data, even if the underlying table structure changes (as long as the view definition can still be satisfied).

	>[!Note]
	>View stores the underlying structure of the table once created so any changes in table in future won't effect the view.
	>It will run the underlying query over new table but using old structure of table only.
	>Can't update views with GROUP BY, With and DISTINCT clause and those which use window-functions.

```plsql
CREATE VIEW view_name AS
SELECT column1, column2, ...
FROM table_name
WHERE condition;
```

* **Example:**

```sql
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    price DECIMAL,
    category VARCHAR(255)
);

INSERT INTO products (name, price, category) VALUES
('Product A', 10.00, 'Electronics'), ('Product B', 20.00, 'Clothing'),
('Product C', 30.00, 'Electronics'), ('Product D', 40.00, 'Clothing');

CREATE VIEW electronics_products AS
SELECT name, price
FROM products
WHERE category = 'Electronics';

SELECT * FROM electronics_products; -- Accessing the view

-- DROP VIEW view_name;
```
---
---
### 5. Subquery:

A query within a query is called sub-query, the inner query will execute first and its result will be used as new search-space for outer query.

```plsql
-- get all user whose salry falls in top-3
SELECT * FROM Employees
WhERE Salary IN (
	SELECT DISTINCT(Salary) 
	FROM User 
	ORDER BY Salary Desc
	LIMIT 3
) as "Top 3 Salaries";

-- IMP Operators
-- ALL : if all condns True->True
-- ANY : if any conds is True->True
-- EXISTS: check if some value Exsist in result or not
-- NOT EXISTS: ~ EXSITS
SELECT * FROM Employees
WHERE Salary < ALL (
	Select Salary FROM Employees
	Where ROLE IN ('Asst.Manager','HR','Manager')
);

SELECT * FROM Employees 
WHERE emp_id EXISTS (
	SELECT * FROM Projects
	WHERE assigned=False AND ROLE IN ('Dev','Intern','Devops')
);

```

---
---

### 6. With Clause

The `WITH` clause (also known as a Common Table Expression or CTE) lets you define a temporary, named result set within a single query.  It's essentially a subquery that you can reference multiple times.

**Why Use:**

* **Improved Readability:** Makes complex queries easier to understand by breaking them into smaller, logical parts.
* **Code Reusability:** Avoids repeating the same subquery multiple times.
* **Simplified Logic:**  Can simplify queries with recursive structures (e.g., hierarchical data).


**Example:**

```mysql
-- 1. Fetch employees who earn more than average salary of all emp

-- without with
SELECT * FROM Employee
WHERE salary >(
	SELECT AVG(salary) FROM Employee
);

-- with with-clause
WITH average_salary (avg_sal) AS (
	SELECT AVG(salary) from Employee
)
-- Main Query
Select *
FROM employee e, average_salary av
WHERE e.salary>av.avg_sal;



-- 2. Find stores who's sales better than the average sales across all stores

-- without with
select * 
FROM (
	select s.store_id,sum(s.qty*s.price) as tot_salses_per_store
	from sales s
	group by store_id
)
join (
	select avg(tot_sales_per_store) as avg_sales_for_all_stores
	from (
		select s.store_id,sum(s.cost*s.price) as total_sales_per_store
		from sales s
		group by store_id
	) x
) avg_sales
on total_sales.total_sales_per_store>avg_sales.avg_sales_for_all_store;


-- with with-clause
with total_sales, (store_id,total_sales_per_store) as (
	select s.store_id,sum(s.cost*s.price)
	from sales s
	group by s.store_id
), avg_sales (avg_sales_per_store) as (
	select avg(total_sales_per_store) as avg_sales_for_all_stores
	from total_sales
)
Select * 
from total_sales ts
join avg_sales as
on ts.total_sales_per_store>av.avg_sales_per_store;
```


---
---
### 7. Procedures 

**Stored Procedures in SQL (PostgreSQL):**

* A stored procedure is a set of SQL statements that are stored in the database.  They can accept input parameters and return results.
* **Why use them?**
    * **Code Reusability:** Encapsulate complex logic into reusable units.
    * **Performance:** Reduce network traffic by sending a single call to execute a procedure instead of multiple queries.
    * **Security:** Control access to data through procedures, granting specific permissions to execute them.
* **Syntax (PostgreSQL - using PL/pgSQL):**

```sql
CREATE OR REPLACE PROCEDURE procedure_name(parameter1 datatype, parameter2 datatype, ...)
LANGUAGE plpgsql
AS $$
DECLARE
    -- Declare variables here (optional)
BEGIN
    -- SQL statements and PL/pgSQL code here
END;
$$;
```

* **Example:**

```sql
CREATE OR REPLACE PROCEDURE update_product_price(product_id INTEGER, new_price DECIMAL)
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE products
    SET price = new_price
    WHERE id = product_id;
END;
$$;

-- Call the procedure:
CALL update_product_price(1, 12.00); 

-- Verification:
SELECT * FROM products WHERE id = 1; -- Price should now be 12.00

```
---
---

### 7. Functions

A function is a pre-compiled SQL code block that performs a specific task and _returns_ a single value.

**Key Characteristics:**

- **Returns a value:** This is the crucial difference from a procedure. A function _must_ return a value of a specified data type.
- **Can be called from other SQL statements:** Functions can be embedded directly within `SELECT`, `INSERT`, `UPDATE`, etc., statements.
- **Can accept input parameters (arguments):** You can pass values to a function to customize its behavior.

_Example (PL/SQL):_

```plsql
CREATE OR REPLACE FUNCTION calculate_total_salary (p_dept_id NUMBER)
RETURN NUMBER
IS -- used for declaration Section
 -- variable in plsql
  v_total_salary NUMBER := 0;
BEGIN
  SELECT SUM(salary) INTO v_total_salary FROM employees WHERE department_id = p_dept_id;
  RETURN v_total_salary;
END;


SELECT department_id, calculate_total_salary(department_id) AS total_salary
FROM departments;
```

## Procedures vs. Functions in SQL

|Feature|Function|Procedure|
|---|---|---|
|**Return Value**|Must return a single value|Does not return a value (or returns void)|
|**Usage**|Can be used within SQL statements|Called independently; typically used for side effects|
|**Purpose**|Compute and return a result|Perform actions (updates, inserts, etc.)|
|**Call Syntax**|Embedded in queries (`SELECT`, etc.)|Called using `EXECUTE` or `CALL` statement|


---
---

### 8. Transactions

A transaction in SQL is a sequence of one or more SQL statements that are treated as a single unit of work.  Either all the statements in a transaction are executed successfully, or none of them are. This ensures data consistency and integrity.

**Key Properties of Transactions (ACID):**

* **Atomicity:**  All changes within a transaction are treated as a single, indivisible unit. Either all changes are committed, or none are.
* **Consistency:** A transaction maintains the integrity of the database by ensuring that all data conforms to defined rules and constraints.
* **Isolation:** Transactions are isolated from each other, meaning that changes made in one transaction are not visible to other transactions until the first transaction is committed.
* **Durability:** Once a transaction is committed, the changes are permanent and survive even system failures.

**Transaction Control Statements in PostgreSQL:**

* `BEGIN;` or `START TRANSACTION;`: Starts a new transaction.
* `COMMIT;`: Saves all changes made within the transaction.
* `ROLLBACK;` or `ABORT;`: Discards all changes made within the transaction.
* `SAVEPOINT savepoint_name;`: Creates a savepoint within a transaction. You can later roll back to a savepoint without discarding all changes.
* `ROLLBACK TO savepoint_name;`: Rolls back the transaction to the specified savepoint.
* `RELEASE SAVEPOINT savepoint_name;`:  Releases a savepoint.


**Example in PostgreSQL:**

```sql
-- Create a sample table
CREATE TABLE accounts (
    id SERIAL PRIMARY KEY,
    balance DECIMAL
);

INSERT INTO accounts (balance) VALUES (1000), (500);


-- Example transaction
BEGIN;  -- Start the transaction

UPDATE accounts SET balance = balance - 200 WHERE id = 1; -- Deduct from account 1
UPDATE accounts SET balance = balance + 200 WHERE id = 2; -- Add to account 2

-- Check if account 2 has sufficient funds (imagine some complex business rule)
-- In a real application, you would have more robust error handling here.

IF (SELECT balance FROM accounts WHERE id = 2) < 0 THEN
    ROLLBACK;  -- If not enough funds, rollback the entire transaction
    RAISE EXCEPTION 'Insufficient funds in account 2';
ELSE
    COMMIT;  -- If enough funds, commit the transaction
END ;

-- Verify the changes
SELECT * FROM accounts;
```

**Explanation:**

1. The `BEGIN` statement starts a transaction.
2. Two `UPDATE` statements transfer funds between accounts.
3. An `IF` statement checks a condition (in this case, sufficient funds).
4. If the condition is not met, `ROLLBACK` undoes both `UPDATE` statements.
5. If the condition is met, `COMMIT` saves the changes permanently.

**Using Transactions in PL/SQL:**


Transactions are automatically managed within PL/pgSQL functions and procedures.  By default, a PL/pgSQL block runs within a single transaction.  You can use `BEGIN`, `COMMIT`, `ROLLBACK`, and savepoints within a PL/pgSQL block to control transaction boundaries more finely if needed.  However, if an error occurs that is not caught and handled within the PL/pgSQL block, the entire transaction will be rolled back automatically.
