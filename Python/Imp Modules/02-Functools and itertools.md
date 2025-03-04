## 1.functools:

Pythons `functools` is for higher-order functions: functions that act on or return other functions.

**1. Caching:** We can `memoize/cache` a function state without modifying it. `functools` provide many wrappers and function that we can use to cache a function such as _@functools.lru_cache_ , _@functools.cache_ , _@functools.cached__property_ etc.

```python
from functools import cache,lru_cache

# memoize the function based on args, if they match for a function call return the output if +nt.
@cache 
def factorial(n):
if n<=1:
	return 1
return n*factorial(n-1) 
```

__2. cmp_to_key() :___ Transform an old-style comparison function to a [key function](https://docs.python.org/3/glossary.html#term-key-function). A _comparison function_ is any callable that accepts two arguments, compares them, and returns a negative number for less-than, zero for equality, or a positive number for greater-than. A _key function_ is a callable that accepts one argument and returns another value to be used as the sort key.

```python
from functools import cmp_to_key

def cmp(v1,v2):
	return len(v1)-len(v2)

arr=['Deepanshu','Rahul','Aaditya']
sorted(arr,key=-len) == sorted(arr,key=cmp) # python style comparision --> Java/C++ style comparision
```

__3. @functools.total_ordering:__ Given a class defining one or more rich comparison ordering methods, this class decorator supplies the rest. This simplifies the effort involved in specifying all of the possible rich comparison operations:

The class must define one of `__lt__()`, `__le__()`, `__gt__()`, or `__ge__()`. In addition, the class should supply an `__eq__()` method.

```python
form functools import total_ordering

@total_ordering
class Student:
    def _is_valid_operand(self, other):
        return (hasattr(other, "lastname") and
                hasattr(other, "firstname"))
    def __eq__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return ((self.lastname.lower(), self.firstname.lower()) ==
                (other.lastname.lower(), other.firstname.lower()))
    def __lt__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return ((self.lastname.lower(), self.firstname.lower()) <
                (other.lastname.lower(), other.firstname.lower()))

# We have only define __eq__ and __lt__, using this the other ording functions will be made automatically.
```

> [!Note]
> 1. While this decorator makes it easy to create well behaved totally ordered types, it _does_ come at the cost of slower execution and more complex stack traces for the derived comparison methods, if this acts as a bottleneck for a given application, implementing all six rich comparison methods instead is likely to provide an easy speed boost.
>    <br>
> 2. This decorator makes no attempt to override methods that have been declared in the class _or its superclasses_. Meaning that if a superclass defines a comparison operator, _total_ordering_ will not implement it again, even if the original method is abstract.
>    <br>

__4. Partial function:__ Return a new [partial object](https://docs.python.org/3/library/functools.html#partial-objects) which when called will behave like _func_ called with the positional arguments _args_ and keyword arguments _keywords_. If more arguments are supplied to the call, they are appended to _args_. If additional keyword arguments are supplied, they extend and override _keywords_.

```python
from functools import partial
basetwo=partial(int,2)
basetwo('10010')
```

__5. Reduce:__  It applies a function of two arguments repeatedly on the elements of a sequence so as to reduce the sequence to a single value.

```python
from functools import reduce
arr=list(range(1,100,2))
sum=reduce(lambda acc,val: acc+val , arr, 0)
```

**6. SingleDispatch:**  It is a function decorator. It transforms a function into a generic function so that it can have different behaviors depending upon the type of its first argument. It is used for function overloading, the overloaded implementations are registered using the register() attribute.

```python
from functools import singledispatch
 
@singledispatch
def fun(s):
    print(s)
 
@fun.register(int)
def _(s):
    print(s * 2)
 
 
fun('Hello') # -> Hello 
fun(10) # -> 20
```

## 2. itertools:

Python’s `itertool` is a module that provides various functions that work on iterators to produce complex iterators. This module works as a fast, memory-efficient tool that is used either by themselves or in combination to form **iterator algebra**.

_Different types of iterator provided by `itertools`_ :

### 1. Infinite Iterators:
An iterator that never exhaust is called infinite.

1. **count(start, step):** This iterator **starts printing from the “start” number and prints infinitely**. If steps are mentioned, the numbers are skipped else step is 1 by default. See the below example for its use with for in loop.

2. **cycle(iterable):** This iterator prints all values in order from the passed container. It restarts **printing from the beginning again when all elements are printed in a cyclic manner**.

3. **repeat(val, num):** This iterator repeatedly prints the passed value an infinite number of times. If the optional keyword num is mentioned, then it repeatedly prints num number of times.

```python
from itertools import count,cycle,repeat

# count
for i in count(5,5):
	if i>30:
		break
	print(i,end="" )

arr=list(range(1,100,2))
it=cycle(arr)
for i in range(100):
	print(next(it), end=" ")

arr=list(range(1,100,2))
it=repeat(arr,3)
for i in range(100):
	print(next(it), end=" ")
```

### 2. Combinatoric iterators:
The recursive generators that are used to simplify combinatorial constructs such as permutations, combinations, and Cartesian products are called combinatoric iterators.

1. **product():** This tool **computes the cartesian product** of input iterables. The output of this function is tuples in sorted order.
   
2. **permutations():** **Generate all possible permutations of an iterable**. All elements are treated as unique based on their position and not their values. This function takes an iterable and group_size, if the value of group_size is not specified or is equal to None then the value of group_size becomes the length of the iterable.
   
3. **combinations():** This iterator prints **all the possible combinations(without replacement)** of the container passed in arguments in the specified group size in sorted order.

4. **combinations_with_replacement():** This function returns a subsequence of length n from the elements of the iterable where n is the argument that the function takes determining the length of the subsequences generated by the function. **Individual elements may repeat itself** in combinations_with_replacement function. 

```python
from itertools import product,permutation,combination,combination_wit_replacement

# ========================== product =========================
print("The cartesian product using repeat:")
print(list(product([1, 2], repeat=2)))
print()
 
print("The cartesian product of the containers:")
print(list(product(['geeks', 'for', 'geeks'], '2')))
print()
 
print("The cartesian product of the containers:")
print(list(product('AB', [3, 4])))


# ========================== permutation =========================
print("All the permutations of the given list tkaen 2 elements a time is:")
print(list(permutations([1, 'geeks'], 2)))
 
print("All the permutations of the given string is:")
print(list(permutations('AB')))
 
print("All the permutations of the given container is:")
print(list(permutations(range(3), 2)))


# ========================== combination =========================
print ("All the combination of list taken 2 ele @ a time in sorted order(without replacement) is:")  
print(list(combinations(['A', 2], 2))) 
   
print ("All the combination of string in sorted order(without replacement) is:") 
print(list(combinations('ABC',None))) 
   
print ("All the combination of list in sorted order(without replacement) is:") 
print(list(combinations(range(2), 1))) 


# ========================== combination_with_replacement =========================
print("All the combination of string in sorted order(with replacement) is:")
print(list(combinations_with_replacement("AB", 2)))
 
print("All the combination of list in sorted order(with replacement) is:")
print(list(combinations_with_replacement([1, 2], 2)))
 
print("All the combination of container in sorted order(with replacement) is:")
print(list(combinations_with_replacement(range(2), 1)))
```

### 3. Terminating iterators
Terminating iterators are used to work on the short input sequences and produce the output based on the functionality of the method used.

1. **accumulate(iter, func):** This iterator takes two arguments, iterable target and the function which would be followed at each iteration of value in target. If no function is passed, addition takes place by default. If the input iterable is empty, the output iterable will also be empty.
   
2. **chain(iter1, iter2..):** This function is used to print all the values in iterable targets one after another mentioned in its arguments.
   
3. **chain.from_iterable():** This function is implemented similarly as a chain() but the argument here is a list of lists or any other iterable container.
   
4. **compress(iter, selector):** This iterator selectively picks the values to print from the passed container according to the boolean list value passed as other arguments. The arguments corresponding to boolean true are printed else all are skipped.
   
5. **dropwhile(func, seq):** This iterator starts printing the characters only after the func. in argument returns false for the first time.
   
6. **filterfalse(func, seq):** As the name suggests, this iterator prints only values that return false for the passed function.
   
7. **islice(iterable, start, stop, step):** This iterator selectively prints the values mentioned in its iterable container passed as argument. This iterator takes 4 arguments, iterable container, starting pos., ending position and step.
   
8. **starmap(func., tuple list):** This iterator takes a function and tuple list as argument and returns the value according to the function from each tuple of the list.
   
9. **takewhile(func, iterable):** This iterator is the opposite of dropwhile(), it prints the values till the function returns false for 1st time.
   
10. **tee(iterator, count):-** This iterator splits the container into a number of iterators mentioned in the argument.
    
11. **zip_longest( iterable1, iterable2, fillval):** This iterator prints the values of iterables alternatively in sequence. If one of the iterables is printed fully, the remaining values are filled by the values assigned to fillvalue.

```python
from itertools import accumuate,chain,compress,dropwhile,filterfalse,isslice,starmap,tee,zip_longest

# =================== accumulate ===================
for ele in accumulate([1,2,3],lambda prv_val,x: prv_val*x): # [cumm_product]
	print(ele)

for ele in accumulate([1,2,3]) # [cumm_sum]
	print(ele)


# =================== chain ===================
for ele in chain(range(1,3),["deepanshu","bhatt"],"hello,world"):
	print(ele)


# =================== chain.from_iterable ===================
for ele in chain.from_iterable([range(1,3),range(-1,-10,-1)],[["deepanshu","bhatt"]],["hello,world"]):
	print(ele)


# =================== compress ===================
print("The compressed values in string are : ", end="")
print(list(compress('GEEKSFORGEEKS', 
	[1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]))) # OUTPUT: ['G','F','G']


# =================== dropwhile ===================
li = [2, 4, 5, 7, 8] 
print ("The values after condition returns false : ", end ="") 
print (list(dropwhile(lambda x : x % 2 == 0, li)))  # OUTPUT: [5, 7, 8]


# =================== filterfalse ===================
print ("The values that return false to function are : ", end ="") 
print (list(filterfalse(lambda x : x % 2 == 0, li)))  # OUTPUT: [5,7]


# =================== isslice ===================
li = [2, 4, 5, 7, 8, 10, 20] 
# starts printing from 2nd index till 6th skipping 2 
print ("The sliced list values are : ", end ="") 
print (list(islice(li, 1, 6, 2))) 


# =================== starmap ===================
li = [ (1, 10, 5), (8, 4, 1), (5, 4, 9), (11, 10, 1) ]    
# selects min of all tuple values 
print ("The values acc. to function are : ", end ="") 
print (list(starmap(min, li))) # OUTPUT: [1,1,4,1]


# =================== takewhile ===================
li = [2, 4, 6, 7, 8, 10, 20] 
# using takewhile() to print values till condition is false. 
print ("The list values till 1st false value are : ", end ="") 
print (list(takewhile(lambda x : x % 2 == 0, li )))


# =================== tee ===================
li = [2, 4, 6, 7, 8, 10, 20]

iti = iter(li)
 
# makes list of 3 iterators having same values.
it = tee(iti, 3)
 
# printing the values of iterators
print("The iterators are : ")
for i in range(0, 3):
    print(list(it[i]))

# =================== ziplongest ===================
print(*(zip_longest('GesoGes', 'ekfrek', fillvalue='_')))
```


## 3. Toolz Module in Python:

**Toolz** package provides a set of utility functions for iterators, functions, and dictionaries. These functions extend the standard libraries **itertools** and **functools** and borrow heavily from the standard libraries of contemporary functional languages.

Toolz is functools/itertools on steroids.

```shell
pip install toolz
```

This package consists of following modules –

- _dicttoolz_
- _functoolz_
- _itertoolz_
- _recipes_
- _sandbox_

### 1. itertoolz:

```python
from toolz import itertoolz

# remove(predicate_func,iterable): Return those items of sequence for which predicate(item) is False
list(remove(lambda x:x%2==0, [1, 2, 3, 4])) # -> 1,3


# accumulate(binop, seq, initial='__no__default__'): Repeatedly apply binary function to a sequence, accumulating results, good for making functions like cumulative sum
from operator import add,mul
list(itertoolz.accumulate(add, [1, 2, 3, 4, 5])) # -> [1, 3, 6, 10, 15]
list(itertoolz.accumulate(mul, [1, 2, 3, 4, 5])) # -> [1, 2, 6, 24, 120]


# itertoolz.groupby(key, seq): Group a collection by a key function
names = ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank']
itertoolz.groupby(len, names)  # -> {3: ['Bob', 'Dan'], 5: ['Alice', 'Edith', 'Frank'], 7: ['Charlie']}

# Non-callable keys imply grouping on a member.
itertoolz.groupby('gender', [{'name': 'Alice', 'gender': 'F'},
                  {'name': 'Bob', 'gender': 'M'},
				  {'name': 'Charlie', 'gender': 'M'}]) 
# {'F': [{'gender': 'F', 'name': 'Alice'}],
# 'M': [{'gender': 'M', 'name': 'Bob'},
#       {'gender': 'M', 'name': 'Charlie'}]}


# itertoolz.meger_sorted(*sequences,**kwargs): Merge and sort a collection of sorted collections. This works lazily and only keeps one value from each iterable in memory.
list(itertoolz.merge_sorted([1, 3, 5], [2, 4, 6])) # -> [1,2,3,4,5,6]
''.join(itertoolz.merge_sorted('abc', 'abc', 'xy')) # 'aabbcczy'
# The “key” function used to sort the input may be passed as a keyword
list(itertoolz.merge_sorted([2, 3], [1, 3], key=lambda x: x // 3)) #->[2, 1, 3, 3]


# itertoolz.interleave(seqs): interleave sequence of sequences
''.join(itertoolz.interleave(('ABC', 'XY'))) # -> AXBYC


# itertoolz.unique(seq, key=None): return only unique valbe based on a seq
tuple(itertoolz.unique((1, 2,1, 3))) # -> (1,2,3)
# uniqueness can be defined using key
tuple(itertoolz.unique(['cat', 'mouse', 'dog', 'hen'], key=len)) # ->('cat', 'mouse')


# itertoolz.isiterable(x)
itertoolz.isiterable([1, 2, 3]) # -> True
itertoolz.isiterable(5) # -> False


# itertoolz.isdistinct(seq): return true if all vals are distinct 
itertoolz.isdistinct([1, 2, 3]) # -> True


# itertoolz.take(n, seq): take first n elements from the seq
list(itertoolz.take(2,[10,20,30,40,50])) # -> [10,20] 


# itertoolz.drop(n, seq): drop first n elements
list(itertoolz.drop(2, [10, 20, 30, 40, 50])) # -> [30,40,50]


# itertoolz.take_nth(n, seq): take every nth element
list(itertoolz.take_nth(2, [10, 20, 30, 40, 50])) # -> [10, 30, 50]


# itertoolz.get(ind, seq, default='__no__default__'): Get element in a sequence or dict
itertoolz.get([1, 2], 'ABC') # (B,C)


# itertoolz.concat(seqs): concat all seq to 1, an infinite sequence will prevent the rest of the arguments from being included
itertoolz.list(concat([[], [1], [2, 3]])) # [1,2,3]


# itertoolz.mapcat(func, seqs): Apply func to each sequence in seqs, concatenating results
list(mapcat(lambda s: [c.upper() for c in s],[["a", "b"], ["c", "d", "e"]])) # -> ['A', 'B', 'C', 'D', 'E']


# itertoolz.frequencies(seq): Find number of occurrences of each value in seq
frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat'])  # -> {'cat': 3, 'ox': 1, 'pig': 2}


# itertoolz.iterate(func, x): Repeatedly apply a function func onto an original input, Yields x, then func(x), then func(func(x)), then func(func(func(x))), etc.
def func(x):
	return x*2

counter=itertoolz.iterate(func,1)
next(counter) # 2
next(counter) # 4


# itertoolz.topk(k, seq, key=None): Find the k largest elements of a sequence, Operates lazily in `n*log(k)` time
```


### 2.functoolz:

```python
from toolz import functoolz

# functoolz.apply(*func_and_args, **kwargs): apply func and return result
def double(x): return 2*x
def inc(x):    return x + 1
apply(double, 5) # -> 10
tuple(map(apply, [double, inc, double], [10, 500, 8000])) # -> (20,501,16000)


# thread_first(val, *forms): Thread value through a sequence of functions/forms
functoolz.thread_first(1, inc, double) # 1 ---inc-> 2 ---double-> 4
# If the function expects more than one input you can specify those inputs in a tuple. The value is used as the first input.
from operator import add,pow
functoolz.thread_first(1, (add, 4), (pow, 2))  # pow(add(1, 4), 2) -> 25
# in general thread_first(x, f, (g, y, z)) --> g(f(x), y, z)


# functoolz.thread_last(val, *forms): Thread value through a sequence of functions/forms but right to left
thread_last(1, (add, 4), (pow, 2))  # pow(2, add(4, 1)) -> 32


# functoolz.memoize: Cache a function’s result for speedy future evaluation
# It is also possible to provide a `key(args, kwargs)` function that calculates keys used for the cache, which receives an `args` tuple and `kwargs` dict as input, and must return a hashable value.
@functoolz.memoize(cache={1:1,0:1},key=lambda args,kwargs: return args)
def factorial(n):
	return n*factorila(n-1)


# functoolz.compose(*funcs): Compose functions to operate in series.Returns a function that applies other functions in sequence. Functions are applied from right to left so that `compose(f, g, h)(x, y)` is the same as `f(g(h(x, y)))`.
inc = lambda i: i + 1
functoolz.compose(str, inc)(3) # -> '4' 

# functoolz.compose_left(*funcs): compose_left(f, g, h)(x, y) -> h(g(f(x,y)))
functoolz.compose_left(str, inc)(3) # -> '31'


# functoolz.pipe(data, *funcs): Pipe a value through a sequence of functions,I.e. `pipe(data, f, g, h)` is equivalent to `h(g(f(data)))`
double = lambda i: 2 * i
pipe(3, double, str) # : -> '6'


# functoolz.complement(func): Convert a predicate function to its logical complement.In other words, return a function that, for inputs that normally yield True, yields False, and vice-versa.
def iseven(n): return n % 2 == 0
isodd = functoolz.complement(iseven)
iseven(2) # True
isodd(2) # False


# functoolz.juxt(*funcs): Creates a function that calls several functions with the same arguments. 
functoolz.juxt(inc, double)(10) # -> (11,20)

```

### 3.dicttoolz:

```python
from toolz import dicttoolz

# dicttoolz.merge_with(func, *dicts, **kwargs) : Merge dictionaries and apply function to combined values.A key may occur in more than one dict, and all values mapped from the key will be passed to the function as a list, such as func([val1, val2, …]).
merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20, 3:30}) # -> {1: 1, 2: 2, 3: 30}


# dicttoolz.valmap(func, d, factory=<class 'dict'>): Apply function to values of dictionary
bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
valmap(sum, bills)  # -> {'Alice': 65, 'Bob': 45}


# dicttoolz.keymap(func, d, factory=<class 'dict'>): Apply function to key of dictionary
keymap(str.lower,bills) # {'alice': [20, 15, 30], 'bob': [10, 35]}


# dicttoolz.itemmap(func, d, factory=<class 'dict'>): Apply function to items of dictionary
accountids = {"Alice": 10, "Bob": 20}
itemmap(reversed, accountids)  # -> {10: "Alice", 20: "Bob"}

# dicttoolz.valfilter(predicate, d, factory=<class 'dict'>): filter values using predicate, vals that return false will be taken
iseven = lambda x: x % 2 == 0
d = {1: 2, 2: 3, 3: 4, 4: 5}
valfilter(iseven, d) # -> {1: 2, 3: 4} 
# similarly we have : keyfilter,itemfilter


# dicttoolz.assoc(d, key, value, factory=<class 'dict'>): Return a new dict with new key value pair,New dict has d[key] set to value. Does not modify the initial dictionary.
assoc({'x': 1}, 'x', 2) # -> {'x': 2}
assoc({'x': 1}, 'y', 3) # -> {'x':2,'y':3}

# dicttoolz.dissoc(d, *keys, **kwargs): Return a new dict with the given key(s) removed.New dict has d[key] deleted for each supplied key. 
dissoc({'x': 1, 'y': 2}, 'y') # -> {'x': 1}
dissoc({'x': 1, 'y': 2}, 'y', 'x') # -> {}
dissoc({'x': 1}, 'y') # Ignores missing keys -> {'x': 1}
```


> [!Also Read About]
>1. [Boltons](https://boltons.readthedocs.io/en/latest/)  : Boltons is a Python library designed to provide a collection of utility functions and types that are often missing from the standard library. It is sometimes referred to as "everyone's util.py," reflecting its aim to fill gaps in Python's built-in capabilities.
> <br>
