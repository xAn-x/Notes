### 1. Window Functions

Window functions in SQL perform calculations across a set of table rows (a "window") that are somehow related to the current row.  Unlike aggregate functions (like `SUM`, `AVG`, `COUNT`), window functions don't collapse the rows into a single result; they return a value for *every* row in the result set.  

Think of them as allowing you to perform aggregate-like operations within a defined group without actually grouping the results.

**Key Concepts:**

* **Window:** The set of rows related to the current row over which the function operates.  This window is defined by the `OVER()` clause.
* **Partitioning:** Dividing the result set into groups (partitions) using the `PARTITION BY` clause within `OVER()`.  The window function is then applied separately to each partition.
* **Ordering:** Sorting the rows within each partition using the `ORDER BY` clause within `OVER()`.  This is crucial for functions like `LAG`, `LEAD`, `RANK`, and `ROW_NUMBER`.
* **Framing:**  Further restricting the window by specifying a starting and ending point relative to the current row.  This is done with the `ROWS` or `RANGE` clause within `OVER()`.

**Example Scenario (PostgreSQL):**

Let's say you have a table `sales` with sales data for different products on different dates:

| product | date       | sales |
|---------|------------|-------|
| A       | 2023-10-26 | 100   |
| B       | 2023-10-26 | 150   |
| A       | 2023-10-27 | 120   |
| B       | 2023-10-27 | 180   |
| A       | 2023-10-28 | 90    |
| B       | 2023-10-28 | 200   |


**1. Running Total (using `SUM() OVER()`):**

```plsql
SELECT
    product,
    date,
    sales,
    SUM(sales) OVER (PARTITION BY product ORDER BY date) AS running_total
FROM sales;
```

This calculates the running total of sales for each product, ordered by date.

**2. Ranking (using `RANK() OVER()`):**

```sql
SELECT
    product,
    date,
    sales,
    RANK() OVER (PARTITION BY product ORDER BY sales DESC) AS sales_rank
FROM sales;
```

This ranks the sales for each product, with 1 being the highest sales.

**3. Previous Row's Value (using `LAG() OVER()`):**

```sql
SELECT
    product,
    date,
    sales,
    LAG(sales, 1, 0) OVER (PARTITION BY product ORDER BY date) AS previous_sales
FROM sales;
```

This gets the sales from the previous day for each product.  The `1` specifies the offset (1 row previous), and `0` is the default value if there's no previous row.

**4. Moving Average (using `AVG() OVER()` with framing):**

```sql
SELECT
    product,****
    date,
    sales,
    AVG(sales) OVER (PARTITION BY product ORDER BY date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS moving_average
FROM sales;
```

This calculates a 3-day moving average of sales for each product. The `ROWS BETWEEN 2 PRECEDING AND CURRENT ROW` clause defines the window frame.


**Important Cases and Considerations:**

* **Default Window:** If you omit the `OVER()` clause entirely, the window function acts like a regular aggregate function, collapsing all rows into a single result.
* **Empty `OVER()`:**  `OVER()` with nothing inside means the window includes all rows in the result set (no partitioning or ordering).
* **Framing Options:**  Besides `ROWS`, you can use `RANGE` for framing based on values rather than row numbers.  You can also use `UNBOUNDED PRECEDING` and `UNBOUNDED FOLLOWING` to specify the beginning and end of the partition, respectively.
* **Performance:** Window functions can be computationally intensive, especially with large datasets and complex window definitions.  Proper indexing and query optimization are essential.


![[Pasted image 20241206190513.png]]

---
---

### 2. CTE (Common Table Expressions)

A Common Table Expression (CTE) is a temporary, named result set defined within the execution scope of a single SQL statement. _It exists only during the execution of that statement and is not stored as a permanent object like a view or table_.  

CTEs provide a way to organize and modularize complex queries, making them more readable and easier to understand.  They are particularly useful for recursive queries and situations where you need to refer to the same subquery multiple times within a larger query.

While CTEs are part of standard SQL and available in most databases (including PostgreSQL, MySQL 8+, SQL Server, etc.), they are *not* directly part of PL/SQL itself.  PL/SQL is Oracle's procedural extension to SQL.  However, you can use CTEs *within* SQL statements that are embedded in a PL/SQL block.

**Example in PL/SQL (Oracle):**


```plsql
-- Create a temp table contain max-sal per dep
WITH max_sal as (
	SELECT dept,max(Salary) as Max_salary
	FROM Employee 
	GROUP BY (dept)
)
-- using the table instantly here to find emp in each dept having
-- max-salary in their dept
SELECT
	e.employee_id,e.name,e.dept,e.salary,ms.max_salary
FROM Employee as e
JOIN Max_salary as ms
ON e.dept=ms.dept
WHERE
	e.salary>ms.max_salary;
```



**Key Benefits of CTEs:**

* **Improved Readability:**  CTEs break down complex queries into smaller, logical units, making them easier to understand and maintain.
* **Code Reusability:** A CTE can be referenced multiple times within the same query.
* **Recursive Queries:** CTEs are essential for performing recursive queries (e.g., traversing hierarchical data).
* **Simplified Query Logic:** CTEs can encapsulate complex logic, making the overall query structure cleaner.


**Important Note:**  CTEs have a scope limited to the single SQL statement in which they are defined. They cannot be referenced across multiple PL/SQL blocks or stored procedures.  If you need a reusable named query, consider creating a view instead.  If you need to pass data between PL/SQL blocks, use variables or collections.
CTE is a temp result set that you can define within a query to simplify complex SQL statements. 

---
---

### 3. Triggers

Triggers in SQL are stored programs that are **automatically executed** (triggered) in response to certain events on a particular table or view. These events can be DML operations (INSERT, UPDATE, DELETE) or DDL operations (CREATE, ALTER, DROP). Triggers are powerful tools for enforcing business rules, maintaining data integrity, auditing changes, and performing automated actions.

**Key Components of a Trigger:**

* **Event:** The DML or DDL operation that activates the trigger (e.g., `BEFORE INSERT`, `AFTER UPDATE`, `INSTEAD OF DELETE`).
* **Triggering Statement:** The SQL statement that caused the event.
* **Trigger Body:** The PL/SQL block of code that is executed when the trigger is fired.
* **Trigger Timing:**  `BEFORE` or `AFTER` the triggering statement.  `INSTEAD OF` replaces the triggering statement.
* **Trigger Level:** `ROW`-level triggers fire for each row affected by the triggering statement. `STATEMENT`-level triggers fire only once per triggering statement, regardless of the number of rows affected.
* **Trigger Type:**  DML triggers (for data modifications) or DDL triggers (for schema changes).

**SYNTAX**

![[Pasted image 20241206192425.png]]


**PL/SQL Example (DML Trigger):**

Let's create a trigger that automatically updates an `order_total` column in an `orders` table whenever a new order item is inserted into an `order_items` table.

```plsql
-- function to chk salary and set it to 0 if sal<0
CREATE OR REPLACE FUNCTION validate_sal()
RETURNS TRIGEER AS $$
BEGIN
 IF NEW.salary<0 THEN NEW.salary=0
 END IF;
 RETURN NEW
END;
$$ LANGUAGE plpgsql;

-- creating the trigger using the function
CREATE TRIGGER before_update_salary
BEFORE UPDATE ON Employee
FOR EACH ROW
EXECUTE FUNCTION validate_sal();
```


**Important Variants and Considerations:**

* **`BEFORE` Triggers:**  Execute before the triggering statement. Useful for validation or modifying data before it's inserted/updated.  `:NEW` is available for `INSERT` and `UPDATE` triggers.
* **`AFTER` Triggers:** Execute after the triggering statement.  Useful for auditing, logging, or performing actions based on the modified data. `:NEW` and `:OLD` (for the previous row values in `UPDATE` and `DELETE` triggers) are available.
* **`INSTEAD OF` Triggers:**  Replace the triggering statement. Primarily used for views that are not directly updatable.
* **`STATEMENT`-Level Triggers:** Fire only once per triggering statement. More efficient than row-level triggers when you don't need to access individual row values.
* **`DDL Triggers`:** Fire in response to DDL events like `CREATE`, `ALTER`, or `DROP`. Useful for auditing schema changes or enforcing naming conventions.  They operate at the statement level.
* **`:OLD` Pseudo-Record:**  Available in `UPDATE` and `DELETE` triggers. Refers to the row *before* the modification or deletion.
* **Mutating Table Restrictions:**  `BEFORE` and `AFTER ROW` triggers cannot directly modify the table that fired the trigger (mutating table).  Workarounds involve using autonomous transactions or package variables.


Triggers are a powerful feature, but overuse can lead to complex dependencies and performance issues.  Carefully consider the appropriate trigger type and level for your specific needs, and thoroughly test your triggers to ensure they function correctly and don't introduce unintended side effects.
