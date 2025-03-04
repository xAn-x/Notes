
**Python Asyncio** provides **asynchronous programming** with **coroutines**.

**Asynchronous programming** is a popular programming paradigm that allows a large number of lightweight tasks to run concurrently with very little memory overhead, compared to threads.

# What is Asynchronous Programming

**Asynchronous programming** is a programming paradigm that _does not block_. Instead, requests and function calls are issued and executed somehow in the background at some future time. This _frees the caller to perform other activities and handle the results of issued calls at a later time_ when results are available or when the caller is interested.

`Asynchronous: Separate execution streams that can run concurrently in any order relative to each other are asynchronous.`

**Future**: A handle on an asynchronous function call allowing the status of the call to be checked and results to be retrieved.

**Asynchronous Task**: Used to refer to the aggregate of an asynchronous function call and resulting future.
- Asynchronous routines are able to “pause” while waiting on their ultimate result and let other routines run in the meantime.
- [Asynchronous code](https://realpython.com/python-async-features/), through the mechanism above, facilitates concurrent execution. To put it differently, asynchronous code gives the look and feel of concurrency.

_Used with non-blocking I/O_ (don't wait for i/o to complete)

**Asynchronous I/O**: A shorthand that refers to combining asynchronous programming with non-blocking I/O. async IO is a single-threaded, single-process design, but gives a feeling of concurrency despite using a single thread in a single process. Async I/O utilizes coroutines to give this feeling of parallelism.

**Coroutine**: Coroutines are a more generalized form of subroutines. Subroutines are entered at one point and exited at another point. Coroutines can be entered, exited, and resumed at many different points. Coroutines benifts from cooperative multitasking

**Cooperative multitasking** is a fancy way of saying that a program’s event loop communicates with multiple tasks to let each take turns running at the optimal time.

**Asynchronous I/O** is a style of concurrent programming, but it is not parallelism. It’s more closely aligned with threading than with multiprocessing but is very much distinct from both of these and is a standalone member in concurrency’s bag of tricks.

![[Python Libraries/Advance Python/Screen_Shot_2018-10-17_at_3.18.44_PM.c02792872031.avif]]



Async I/O may at first seem counterintuitive and paradoxical. How does something that facilitates concurrent code use a single thread and a single CPU core? 

Async I/O takes long waiting periods in which functions would otherwise be blocking and allows other functions to run during that downtime. (A function that blocks effectively forbids others from running from the time that it starts until the time that it returns.)

# The `asyncio` Package and `async`/`await`
At the heart of async IO are coroutines. A ___coroutine___ is a specialized version of a Python generator function. Let’s start with a baseline definition and then build off of it as you progress here: _a coroutine is a function that can suspend its execution before reaching `return`, and it can indirectly pass control to another coroutine for some time._

```python
#!/usr/bin/env python3
# countasync.py

import asyncio

# async is used to create a coroutine
async def count():
    print("One")
    await asyncio.sleep(1) # add this coroutine to event queue,and jump to next instruction without pausing
    
    # time.sleep(1) # this wont push the couroutine to event loop as ->`await` describes that this oprn can take time, so the coroutine immediately haults saving its state and python can execute other instructions while the output for this being prepare & later we can check if the output is prepared and starts execution from here only.
    
    print("Two")

async def main():
    await asyncio.gather(count() , count(), count()) # gather all these coroutines

if __name__ == "__main__":
    import time
    s = time.perf_counter()
    asyncio.run(main())
    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.2f} seconds.")
```

```shell
One
One
One
Two
Two
Two
countasync.py executed in 1.01 seconds.
```

When each task reaches `await asyncio.sleep(1)`, the function yells up to the event loop and gives control back to it, saying, “I’m going to be sleeping for 1 second. Go ahead and let something else meaningful be done in the meantime.”

`time.sleep()` can represent any time-consuming blocking function call, while `asyncio.sleep()` is used to stand in for a non-blocking call (but one that also takes some time to complete).

# Rules of Async IO
- The syntax `async def` introduces either a **native coroutine** or an **asynchronous generator**. The expressions `async with` and `async for` are also valid.

- The keyword `await` passes function control back to the event loop. (It suspends the execution of the surrounding coroutine.) If Python encounters an `await f()` expression in the scope of `g()`, this is how `await` tells the event loop, “Suspend execution of `g()` until whatever I’m waiting on—the result of `f()`—is returned. In the meantime, go let something else run.”

- To use `yield` in an `async def` block. This creates an [asynchronous generator](https://www.python.org/dev/peps/pep-0525/), which you iterate over with `async for`. 

- - Just like it’s a `SyntaxError` to use `yield` outside of a `def` function, it is a `SyntaxError` to use `await` outside of an `async def` coroutine. You can only use `await` in the body of coroutines. 

-  when you use `await f()`, it’s required that `f()` be an object that is [awaitable](https://docs.python.org/3/reference/datamodel.html#awaitable-objects). For now, just know that an awaitable object is either (1) another coroutine or (2) an object defining an `.__await__()` dunder method that returns an iterator.

# Design pattern in Async I/O

## Chaining Coroutines
A key feature of coroutines is that they can be chained together. (Remember, a coroutine object is awaitable, so another coroutine can `await` it.) This allows you to break programs into smaller, manageable, recyclable coroutines:

```python
import asyncio
import random
import time

async def part1(n: int) -> str:
    i = random.randint(0, 10)
    print(f"part1({n}) sleeping for {i} seconds.")
    await asyncio.sleep(i)
    result = f"result{n}-1"
    print(f"Returning part1({n}) == {result}.")
    return result

async def part2(n: int, arg: str) -> str:
    i = random.randint(0, 10)
    print(f"part2{n, arg} sleeping for {i} seconds.")
    await asyncio.sleep(i)
    result = f"result{n}-2 derived from {arg}"
    print(f"Returning part2{n, arg} == {result}.")
    return result

async def chain(n: int) -> None:
    start = time.perf_counter()
    p1 = await part1(n)
    p2 = await part2(n, p1)
    end = time.perf_counter() - start
    print(f"-->Chained result{n} => {p2} (took {end:0.2f} seconds).")

async def main(*args):
    await asyncio.gather(*(chain(n) for n in args))

if __name__ == "__main__":
    import sys
    random.seed(444)
    args = [9,6,3]
    start = time.perf_counter()
    asyncio.run(main(*args))
    end = time.perf_counter() - start
    print(f"Program finished in {end:0.2f} seconds.")
```

```shell
$ python3 chained.py 
part1(9) sleeping for 4 seconds.
part1(6) sleeping for 4 seconds.
part1(3) sleeping for 0 seconds.
Returning part1(3) == result3-1. # as soon as part_1(3) completed is started the execution
part2(3, 'result3-1') sleeping for 4 seconds.
Returning part1(9) == result9-1.
part2(9, 'result9-1') sleeping for 7 seconds.
Returning part1(6) == result6-1.
part2(6, 'result6-1') sleeping for 4 seconds.
Returning part2(3, 'result3-1') == result3-2 derived from result3-1. 
-->Chained result3 => result3-2 derived from result3-1 (took 4.00 seconds).
Returning part2(6, 'result6-1') == result6-2 derived from result6-1.
-->Chained result6 => result6-2 derived from result6-1 (took 8.01 seconds).
Returning part2(9, 'result9-1') == result9-2 derived from result9-1.
-->Chained result9 => result9-2 derived from result9-1 (took 11.01 seconds).
Program finished in 11.01 seconds.
```