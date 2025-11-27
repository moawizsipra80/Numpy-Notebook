# NumPy Comprehensive Quiz Revision Guide
## Complete Guide: Lectures 3.01–3.08

---

**Prepared by:** AI Tutor  
**For:** Data Science Quiz Preparation  
**Level:** Beginner to Intermediate  
**Time to Complete Revision:** 2-3 hours (intensive study)  
**Format:** Professional Implementation Guide with Complete Code Examples & Quiz Answers

**Last Updated:** November 2025

---

## TABLE OF CONTENTS

1. **Lecture 3.01** - Array Creation & Attributes
2. **Lecture 3.02** - Array vs List
3. **Lecture 3.03** - Basic Operations
4. **Lecture 3.04** - Indexing and Slicing
5. **Lecture 3.05** - Broadcasting and Reshaping
6. **Lecture 3.06** - Array Manipulation
7. **Lecture 3.07** - Stacking and Splitting
8. **Lecture 3.08** - IO Operations
9. **Final Revision Checklist**
10. **Practice Questions (20 Questions)**

---

## ⚡ QUICK START: 5-Minute Overview

**Key Topics That Will Appear in Quiz:**
1. Array creation functions and attributes
2. Indexing, slicing, boolean indexing
3. Broadcasting and reshaping
4. Stacking/splitting arrays
5. Array manipulation (append, insert, delete, etc.)
6. File I/O (save/load)
7. Performance comparison with Python lists
8. Statistical operations and aggregations

---

# LECTURE 3.01: Array Creation & Attributes

## Section 1.1: Basic Array Creation

### Function 1: np.array()

**Purpose:** Create array from Python list or tuple  
**Syntax:** `np.array(object, dtype=None)`

```python
# Example 1: 1D array
arr_1d = np.array([1, 2, 3, 4, 5])
print(arr_1d)
# Output: [1 2 3 4 5]

# Example 2: 2D array (matrix)
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
print(arr_2d)
# Output:
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]

# Example 3: With specific datatype
arr_int32 = np.array([1.5, 2.7, 3.9], dtype='int32')
print(arr_int32)
# Output: [1 2 3]

# Example 4: 3D array (tensor)
arr_3d = np.array([[[1, 2], [3, 4]], 
                   [[5, 6], [7, 8]]])
print(arr_3d.shape)
# Output: (2, 2, 2)
```

---

### Function 2: np.zeros()

**Purpose:** Create array filled with zeros  
**Syntax:** `np.zeros(shape, dtype=float)`

```python
# Example 1: 1D array of 5 zeros
zeros_1d = np.zeros(5)
print(zeros_1d)
# Output: [0. 0. 0. 0. 0.]

# Example 2: 2D array (3x4 matrix)
zeros_2d = np.zeros((3, 4))
print(zeros_2d)
# Output:
# [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]]

# Example 3: With specific dtype
zeros_int = np.zeros((2, 3), dtype='int32')
print(zeros_int)
# Output:
# [[0 0 0]
#  [0 0 0]]

# Real-world use: Initialize matrix before filling with data
score_matrix = np.zeros((5, 5))
```

---

### Function 3: np.ones()

**Purpose:** Create array filled with ones  
**Syntax:** `np.ones(shape, dtype=float)`

```python
# Example 1: 1D array of 4 ones
ones_1d = np.ones(4)
print(ones_1d)
# Output: [1. 1. 1. 1.]

# Example 2: 2D array of ones
ones_2d = np.ones((2, 3))
print(ones_2d)
# Output:
# [[1. 1. 1.]
#  [1. 1. 1.]]

# Example 3: Scaling with ones
weights = np.ones((3, 3)) * 0.5
print(weights)
# Output:
# [[0.5 0.5 0.5]
#  [0.5 0.5 0.5]
#  [0.5 0.5 0.5]]
```

---

### Function 4: np.full()

**Purpose:** Create array filled with a specific value  
**Syntax:** `np.full(shape, fill_value, dtype=None)`

```python
# Example 1: Fill with 7
full_arr = np.full((3, 3), 7)
print(full_arr)
# Output:
# [[7 7 7]
#  [7 7 7]
#  [7 7 7]]

# Example 2: Fill with float value
full_float = np.full((2, 4), 3.14)
print(full_float)
# Output:
# [[3.14 3.14 3.14 3.14]
#  [3.14 3.14 3.14 3.14]]

# Example 3: Fill with custom value
matrix = np.full((3, 2), -1)
print(matrix)
# Output:
# [[-1 -1]
#  [-1 -1]
#  [-1 -1]]
```

---

### Function 5: np.eye()

**Purpose:** Create identity matrix (diagonal = 1, rest = 0)  
**Syntax:** `np.eye(N, dtype=float)`

```python
# Example 1: 3x3 identity matrix
identity_3x3 = np.eye(3)
print(identity_3x3)
# Output:
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

# Example 2: 4x4 identity matrix
identity_4x4 = np.eye(4)
print(identity_4x4.shape)
# Output: (4, 4)

# Example 3: With integer dtype
identity_int = np.eye(2, dtype='int32')
print(identity_int)
# Output:
# [[1 0]
#  [0 1]]
```

---

## Section 1.2: Creating Arrays from Ranges

### Function 6: np.arange()

**Purpose:** Create array with evenly spaced values  
**Syntax:** `np.arange(start, stop, step, dtype=None)`

```python
# Example 1: 0 to 9
arr1 = np.arange(10)
print(arr1)
# Output: [0 1 2 3 4 5 6 7 8 9]

# Example 2: 5 to 15
arr2 = np.arange(5, 15)
print(arr2)
# Output: [5 6 7 8 9 10 11 12 13 14]

# Example 3: With step of 2
arr3 = np.arange(0, 10, 2)
print(arr3)
# Output: [0 2 4 6 8]

# Example 4: Float values with step
arr4 = np.arange(0, 1, 0.2)
print(arr4)
# Output: [0.  0.2 0.4 0.6 0.8]

# Example 5: Negative step (counting down)
arr5 = np.arange(10, 0, -2)
print(arr5)
# Output: [10  8  6  4  2]
```

---

### Function 7: np.linspace()

**Purpose:** Create array with N evenly spaced values between start and stop  
**Syntax:** `np.linspace(start, stop, num=50, dtype=None)`

```python
# Example 1: 5 values between 0 and 10
lin1 = np.linspace(0, 10, 5)
print(lin1)
# Output: [ 0.   2.5  5.   7.5 10. ]

# Example 2: 11 values between 0 and 1
lin2 = np.linspace(0, 1, 11)
print(lin2)
# Output: [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. ]

# Example 3: With endpoint=False
lin3 = np.linspace(0, 10, 5, endpoint=False)
print(lin3)
# Output: [0. 2. 4. 6. 8.]

# KEY DIFFERENCE:
# arange(0, 10, 2)     → [0 2 4 6 8]        (specify STEP)
# linspace(0, 10, 5)   → [0 2.5 5 7.5 10]   (specify COUNT)
```

---

## Section 1.3: Random Arrays

### Function 8: np.random.rand()

**Purpose:** Create random array with values in [0, 1)  
**Syntax:** `np.random.rand(d0, d1, ..., dn)`

```python
# Example 1: 1D array of 5 random values
rand1 = np.random.rand(5)
print(rand1)
# Output: [0.37154092 0.64524823 0.15601864 0.88214291 0.55627194]

# Example 2: 2D array of random values
rand2 = np.random.rand(3, 3)
print(rand2)
# Output: 3x3 array with random floats between 0-1

# Example 3: Scale random values (0 to 100)
scaled = np.random.rand(4, 4) * 100
# Output: Array with values between 0 and 100

# Real-world use: Initializing neural network weights
weights = np.random.rand(10, 5) * 0.01
```

---

### Function 9: np.random.randint()

**Purpose:** Create random integers within range  
**Syntax:** `np.random.randint(low, high, size=None, dtype=int)`

```python
# Example 1: Single random integer from 0-9
rand_int = np.random.randint(0, 10)
print(rand_int)
# Output: 7 (or any random int 0-9)

# Example 2: Array of 5 random integers
rand_ints = np.random.randint(0, 10, 5)
print(rand_ints)
# Output: [3 7 2 8 1]

# Example 3: 2D array of random integers
rand_2d = np.random.randint(1, 100, (3, 3))
print(rand_2d)
# Output: 3x3 array with random ints 1-99

# Example 4: Dice roll simulation
dice_rolls = np.random.randint(1, 7, 100)
print(f"Average: {dice_rolls.mean()}")
# Output: Average: 3.48
```

---

### Function 10: np.random.randn()

**Purpose:** Create random array from standard normal distribution  
**Syntax:** `np.random.randn(d0, d1, ..., dn)`

```python
# Example 1: 5 values from normal distribution
randn1 = np.random.randn(5)
print(randn1)
# Output: [-0.23415  1.45829 -0.67234  0.34156  0.89023]

# Example 2: 2D array from normal distribution
randn2 = np.random.randn(2, 3)
# Output: 2x3 array from normal distribution

# Example 3: Scale to different mean and std
mean, std = 100, 15
normal_scores = np.random.randn(1000) * std + mean
print(f"Mean: {normal_scores.mean():.1f}, Std: {normal_scores.std():.1f}")

# Real-world use: IQ scores (mean=100, std=15)
iq_scores = np.random.randn(100) * 15 + 100
```

---

## Section 1.4: Array Attributes (CRITICAL FOR QUIZ!)

### Key Attributes Table

| Attribute | Description | Example |
|-----------|-------------|---------|
| ndim | Number of dimensions | arr.ndim |
| shape | Dimensions (rows, cols) | arr.shape |
| size | Total elements | arr.size |
| dtype | Data type | arr.dtype |
| itemsize | Bytes per element | arr.itemsize |
| nbytes | Total bytes | arr.nbytes |

### Implementation Examples

```python
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

print(f"ndim: {arr.ndim}")           # Output: 2
print(f"shape: {arr.shape}")         # Output: (3, 4)
print(f"size: {arr.size}")           # Output: 12
print(f"dtype: {arr.dtype}")         # Output: int64
print(f"itemsize: {arr.itemsize}")   # Output: 8
print(f"nbytes: {arr.nbytes}")       # Output: 96

# Memory calculation: 12 * 8 = 96 bytes
```

---

## Lecture 3.01: QUIZ QUESTIONS & ANSWERS

**Q1: Which function creates a 3x3 identity matrix?**
- A: `np.eye(3)`

**Q2: How do you create an array of 5 zeros?**
- A: `np.zeros(5)`

**Q3: What's the difference between np.arange(0, 10, 2) and np.linspace(0, 10, 5)?**
- A: arange specifies STEP size, linspace specifies NUMBER of elements
- arange output: [0 2 4 6 8]
- linspace output: [0. 2.5 5. 7.5 10.]

**Q4: What does arr.shape return for a 3x4 matrix?**
- A: (3, 4) - tuple with (rows, columns)

**Q5: How many bytes does a (5, 5) int64 array occupy?**
- A: 5 × 5 × 8 = 200 bytes

**Q6: Create a 2x3 array of all 7s.**
- A: `np.full((2, 3), 7)`

**Q7: What does np.random.rand(3, 4) create?**
- A: 3x4 array of random floats between 0 and 1

**Q8: How do you convert string "1 2 3 4" to NumPy array?**
- A: `np.fromstring("1 2 3 4", dtype=int, sep=' ')`

---

# LECTURE 3.02: Array vs List

## Core Differences

| Feature | Python List | NumPy Array |
|---------|-------------|------------|
| Speed | Slow (interpreted) | 50-100x FASTER |
| Memory | High | 4-8x LESS memory |
| Data Types | Mixed (heterogeneous) | Same type (homogeneous) |
| Operations | Limited, need loops | Vectorized (no loops!) |
| Broadcasting | NO | YES |

---

## Speed Comparison

```python
import numpy as np
import time

# Python List - requires explicit loop
list1 = list(range(1000000))
list2 = list(range(1000000))

start = time.time()
result_list = [list1[i] + list2[i] for i in range(len(list1))]
list_time = time.time() - start

# NumPy Array - vectorized, no loop!
arr1 = np.arange(1000000)
arr2 = np.arange(1000000)

start = time.time()
result_arr = arr1 + arr2
numpy_time = time.time() - start

print(f"NumPy is {list_time/numpy_time:.1f}x FASTER!")
# Output: NumPy is 50.3x FASTER!
```

---

## Memory Comparison

```python
import sys

# Python list
py_list = list(range(1000))
list_memory = sys.getsizeof(py_list)
print(f"List: {list_memory} bytes")

# NumPy array
np_arr = np.arange(1000)
arr_memory = np_arr.nbytes
print(f"Array: {arr_memory} bytes")

# Lists use more memory due to type information
```

---

## Vectorized Operations

```python
# List - Need explicit loop
prices = [10, 20, 30, 40, 50]
discounted_list = [price * 0.9 for price in prices]
# Output: [9.0, 18.0, 27.0, 36.0, 45.0]

# NumPy - ONE LINE!
prices_arr = np.array([10, 20, 30, 40, 50])
discounted_arr = prices_arr * 0.9
# Output: [ 9. 18. 27. 36. 45.]
```

---

## Lecture 3.02: QUIZ QUESTIONS & ANSWERS

**Q1: Why is NumPy array faster than Python list?**
- A: Homogeneous types + vectorized C-level operations without loops

**Q2: How much faster can NumPy be?**
- A: 50-100x faster

**Q3: Can NumPy arrays have mixed data types?**
- A: No, all elements must be same type

**Q4: What's the main advantage of NumPy for AI/ML?**
- A: Vectorized operations, efficient memory, rich math functions

**Q5: How to multiply all NumPy array elements by 2?**
- A: `arr * 2`

---

# LECTURE 3.03: Basic Operations

## Section 3.1: Arithmetic Operations

```python
import numpy as np

a = np.array([10, 20, 30, 40])
b = np.array([2, 4, 5, 8])

# Addition
print(a + b)           # [12 24 35 48]

# Subtraction
print(a - b)           # [8 16 25 32]

# Multiplication
print(a * b)           # [20 80 150 320]

# Division
print(a / b)           # [5. 5. 6. 5.]

# Power
print(a ** 2)          # [100 400 900 1600]

# Modulo
print(a % 3)           # [1 2 0 1]

# Example: Calculate profit margin
cost = np.array([100, 150, 200, 250])
price = np.array([150, 200, 280, 350])
profit = price - cost
margin_percent = (profit / cost) * 100
print(f"Margins: {margin_percent}%")
# Output: [50. 33.33 40. 40.]%
```

---

## Mathematical Functions

```python
arr = np.array([1, 4, 9, 16, 25])

print(np.sqrt(arr))      # [1. 2. 3. 4. 5.]
print(np.exp(np.array([0, 1, 2])))  # [1. 2.71828183 7.3890561]
print(np.log(np.array([1, 2.718, 10])))  # [0. 1. 2.30258509]

abs_arr = np.array([-5, -3, 2, -8, 4])
print(np.abs(abs_arr))   # [5 3 2 8 4]

print(np.round(np.array([1.234, 2.567]), 1))  # [1.2 2.6]
print(np.ceil(np.array([1.2, 2.5])))          # [2. 3.]
print(np.floor(np.array([1.9, 2.1])))         # [1. 2.]
```

---

## Statistical Functions

```python
data = np.array([23, 45, 12, 67, 34, 89, 56, 78, 90, 34])

print(f"Sum: {np.sum(data)}")           # 528
print(f"Mean: {np.mean(data)}")         # 52.8
print(f"Median: {np.median(data)}")     # 50.0
print(f"Std Dev: {np.std(data):.2f}")   # 27.82
print(f"Min: {np.min(data)}")           # 12
print(f"Max: {np.max(data)}")           # 90
print(f"Min idx: {np.argmin(data)}")    # 2
print(f"Max idx: {np.argmax(data)}")    # 8
```

---

## Aggregation with Axis

```python
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print(f"Total: {np.sum(matrix)}")                # 45
print(f"Column sum: {np.sum(matrix, axis=0)}")   # [12 15 18]
print(f"Row sum: {np.sum(matrix, axis=1)}")      # [6 15 24]
print(f"Col mean: {np.mean(matrix, axis=0)}")    # [4. 5. 6.]
print(f"Row max: {np.max(matrix, axis=1)}")      # [3 6 9]
```

---

## Boolean Indexing

```python
arr = np.array([10, 25, 15, 30, 5, 40, 20])

print(arr > 15)                      # [F F F T F T T]
print(arr[arr > 15])                 # [25 30 40 20]
print(arr[arr == 20])                # [20]
print(arr[arr % 2 == 0])             # [10 30 40 20]
print(arr[(arr >= 15) & (arr <= 35)])  # [25 15 30 20]

# Pass/Fail example
scores = np.array([45, 78, 92, 58, 88, 72])
passing = scores >= 70
print(f"Pass rate: {np.mean(passing)*100:.1f}%")
# Output: 66.7%
```

---

## Lecture 3.03: QUIZ QUESTIONS & ANSWERS

**Q1: How do you add two NumPy arrays element-wise?**
- A: `arr1 + arr2`

**Q2: What does np.sqrt(np.array([1, 4, 9, 16])) return?**
- A: [1. 2. 3. 4.]

**Q3: How to get mean of 2D array column-wise?**
- A: `np.mean(arr, axis=0)`

**Q4: What does arr > 50 return?**
- A: A boolean array of True/False

**Q5: How many elements > 5?**
- A: `np.sum(arr > 5)`

**Q6: Get elements < 10:**
- A: `arr[arr < 10]`

**Q7: Standard deviation of [1,2,3,4,5]?**
- A: `np.std([1,2,3,4,5])` → 1.41

---

# LECTURE 3.04: Indexing and Slicing

## 1D Array Indexing

```python
arr = np.array([10, 20, 30, 40, 50])

print(arr[0])           # 10 (first)
print(arr[2])           # 30 (third)
print(arr[-1])          # 50 (last)
print(arr[-2])          # 40 (second-to-last)

arr[1] = 999
print(arr)              # [10 999 30 40 50]
```

---

## 2D Array Indexing

```python
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print(matrix[0, 1])     # 2
print(matrix[2, 2])     # 9
print(matrix[1])        # [4 5 6]
print(matrix[0, :])     # [1 2 3]

matrix[1, 1] = 99
print(matrix)
# [[1  2  3]
#  [4 99  6]
#  [7  8  9]]
```

---

## 1D Slicing

```python
arr = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

print(arr[2:5])         # [20 30 40]
print(arr[:3])          # [0 10 20]
print(arr[5:])          # [50 60 70 80 90]
print(arr[::2])         # [0 20 40 60 80]
print(arr[::-1])        # [90 80 70 60 50 40 30 20 10 0]
print(arr[-3:])         # [70 80 90]
```

---

## 2D Slicing

```python
matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

print(matrix[0:2, :])           # First 2 rows
print(matrix[:, 1:3])           # Columns 1-2
print(matrix[0:2, 1:3])         # 2x2 block
print(matrix[::2])              # Every other row
print(matrix[:, -1])            # Last column: [4 8 12]
```

---

## Boolean Indexing

```python
arr = np.array([10, 25, 15, 30, 5, 40, 20])

mask = arr > 15
filtered = arr[mask]
print(f"Elements > 15: {filtered}")
# Output: [25 30 40 20]

print(arr[arr == 20])           # [20]
print(arr[arr != 10])           # [25 15 30 5 40 20]
print(arr[arr % 2 == 0])        # [10 30 40 20]
print(arr[(arr >= 15) & (arr <= 35)])  # [25 15 30 20]
```

---

## Lecture 3.04: QUIZ QUESTIONS & ANSWERS

**Q1: Get 3rd element:**
- A: `arr[2]`

**Q2: Get last element:**
- A: `arr[-1]`

**Q3: Get elements from index 2 to 5:**
- A: `arr[2:5]`

**Q4: Get every 2nd element:**
- A: `arr[::2]`

**Q5: Reverse an array:**
- A: `arr[::-1]`

**Q6: Get all elements > 50:**
- A: `arr[arr > 50]`

**Q7: Get middle column of 3x3:**
- A: `matrix[:, 1]`

**Q8: Get 2x2 top-left block:**
- A: `matrix[0:2, 0:2]`

---

# LECTURE 3.05: Broadcasting and Reshaping

## Broadcasting Examples

```python
# Example 1: Array + scalar
arr = np.array([1, 2, 3])
result = arr + 5
# Output: [6 7 8]

# Example 2: 2D array + 1D array
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
vector = np.array([10, 20, 30])
result = matrix + vector
# Output:
# [[11 22 33]
#  [14 25 36]]

# Example 3: Different shapes
a = np.array([[1], [2], [3]])    # (3, 1)
b = np.array([10, 20, 30])        # (3,)
result = a + b
# Output:
# [[11 21 31]
#  [12 22 32]
#  [13 23 33]]
```

---

## Reshaping Arrays

```python
# 1D to 2D
arr_1d = np.arange(12)
arr_2d = arr_1d.reshape(3, 4)
# Output:
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# Auto-infer dimension
arr = np.arange(24)
reshaped = arr.reshape(4, -1)   # (4, 6)

# Column vector
col_vector = np.array([1, 2, 3]).reshape(-1, 1)
# Output:
# [[1]
#  [2]
#  [3]]
```

---

## Flatten vs Ravel

```python
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

# flatten() - returns COPY
flat_copy = matrix.flatten()
flat_copy[0] = 999
print(matrix[0, 0])         # 1 (unchanged)

# ravel() - returns VIEW
rav = matrix.ravel()
rav[0] = 888
print(matrix[0, 0])         # 888 (changed)
```

---

## Transpose

```python
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

transposed = matrix.T
# Output:
# [[1 4]
#  [2 5]
#  [3 6]]
```

---

## Sorting

```python
arr = np.array([23, 5, 45, 12, 67, 34])
sorted_arr = np.sort(arr)
# Output: [5 12 23 34 45 67]

# Descending
print(sorted_arr[::-1])
# Output: [67 45 34 23 12 5]

# Indices of sorted array
sorted_indices = np.argsort(arr)
# Output: [1 3 0 5 2 4]
```

---

## Lecture 3.05: QUIZ QUESTIONS & ANSWERS

**Q1: Reshape (2,5) to (5,2):**
- A: `arr.reshape(5, 2)`

**Q2: What does arr.reshape(-1) do?**
- A: Flattens to 1D

**Q3: Difference between flatten() and ravel():**
- A: flatten() copy, ravel() view

**Q4: Transpose a matrix:**
- A: `matrix.T`

**Q5: Sort descending:**
- A: `np.sort(arr)[::-1]`

**Q6: Get sort indices:**
- A: `np.argsort(arr)`

**Q7: Broadcasting (2,3) + (3,):**
- A: (3,) repeated for each row

---

# LECTURE 3.06: Array Manipulation

## Adding/Removing Elements

```python
# Append
arr = np.array([1, 2, 3])
new_arr = np.append(arr, [4, 5])
# Output: [1 2 3 4 5]

# Insert
arr = np.array([1, 2, 3, 4])
result = np.insert(arr, 2, 99)
# Output: [1 2 99 3 4]

# Delete
arr = np.array([10, 20, 30, 40, 50])
result = np.delete(arr, 2)
# Output: [10 20 40 50]
```

---

## Repeating Elements

```python
# repeat() - each element
arr = np.array([1, 2, 3])
repeated = np.repeat(arr, 3)
# Output: [1 1 1 2 2 2 3 3 3]

# tile() - whole array
tiled = np.tile(arr, 3)
# Output: [1 2 3 1 2 3 1 2 3]

# KEY: repeat vs tile
# repeat([1,2], 2) → [1 1 2 2]
# tile([1,2], 2)   → [1 2 1 2]
```

---

## Unique & Counting

```python
arr = np.array([3, 2, 3, 1, 2, 4])

unique_vals = np.unique(arr)
# Output: [1 2 3 4]

unique_vals, counts = np.unique(arr, return_counts=True)
# Values: [1 2 3 4]
# Counts: [1 2 2 1]
```

---

## Copy vs View

```python
original = np.array([1, 2, 3, 4, 5])

# Deep copy - independent
deep_copy = original.copy()
deep_copy[0] = 999
print(original[0])          # 1 (unchanged)

# View - linked
view = original.view()
view[0] = 888
print(original[0])          # 888 (changed)

# Slicing creates view
slice_ref = original[1:4]
slice_ref[0] = 777
print(original)             # [1 777 3 4 5]
```

---

## Concatenating

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 1D
result = np.concatenate([a, b])
# Output: [1 2 3 4 5 6]

# 2D vertical
a_2d = np.array([[1, 2], [3, 4]])
b_2d = np.array([[5, 6], [7, 8]])
result_v = np.concatenate([a_2d, b_2d], axis=0)
# 4x2 array

# 2D horizontal
result_h = np.concatenate([a_2d, b_2d], axis=1)
# 2x4 array
```

---

## Lecture 3.06: QUIZ QUESTIONS & ANSWERS

**Q1: Add [7,8] to end:**
- A: `np.append(arr, [7, 8])`

**Q2: Insert 99 at index 2:**
- A: `np.insert(arr, 2, 99)`

**Q3: Delete index 3:**
- A: `np.delete(arr, 3)`

**Q4: Difference repeat vs tile:**
- A: repeat [1,2]→[1 1 2 2], tile [1,2]→[1 2 1 2]

**Q5: Get unique elements:**
- A: `np.unique(arr)`

**Q6: Difference copy() vs view():**
- A: copy independent, view linked

**Q7: Concatenate vertically:**
- A: `np.concatenate([arr1, arr2], axis=0)`

**Q8: Is slicing view or copy?**
- A: View (linked)

---

# LECTURE 3.07: Stacking and Splitting

## Stacking Functions

```python
# vstack - vertical (adds rows)
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = np.vstack((a, b))
# Output:
# [[1 2 3]
#  [4 5 6]]

# hstack - horizontal (adds columns)
result = np.hstack((a, b))
# Output: [1 2 3 4 5 6]

# dstack - depth (creates 3D)
mat1 = np.array([[1, 2], [3, 4]])
mat2 = np.array([[5, 6], [7, 8]])
result = np.dstack((mat1, mat2))
# Shape: (2, 2, 2)

# stack - flexible axis
result0 = np.stack((a, b), axis=0)   # like vstack
result1 = np.stack((a, b), axis=1)   # like hstack
```

---

## Splitting Functions

```python
# split() - equal parts
arr = np.array([1, 2, 3, 4, 5, 6])
split_result = np.split(arr, 3)
# Output: [array([1, 2]), array([3, 4]), array([5, 6])]

# vsplit() - split rows
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [10, 11, 12]])
split = np.vsplit(matrix, 2)
# Part 1: 2x3, Part 2: 2x3

# hsplit() - split columns
split = np.hsplit(matrix, 3)
# Each: 4x1

# array_split() - unequal
arr = np.array([1, 2, 3, 4, 5, 6, 7])
result = np.array_split(arr, 3)
# [1,2,3], [4,5], [6,7]
```

---

## Lecture 3.07: QUIZ QUESTIONS & ANSWERS

**Q1: Stack vertically:**
- A: `np.vstack((arr1, arr2))`

**Q2: Stack horizontally:**
- A: `np.hstack((arr1, arr2))`

**Q3: vstack does:**
- A: Adds rows

**Q4: hstack does:**
- A: Adds columns

**Q5: Shape vstack (2,3)+(2,3):**
- A: (4, 3)

**Q6: Shape hstack (2,3)+(2,3):**
- A: (2, 6)

**Q7: Split into 3 parts:**
- A: `np.split(arr, 3)`

**Q8: vsplit vs hsplit:**
- A: vsplit rows, hsplit columns

---

# LECTURE 3.08: IO Operations

## Binary Format

```python
# Save single
arr = np.array([1, 2, 3, 4, 5])
np.save('myarray.npy', arr)

# Load single
loaded = np.load('myarray.npy')

# Save multiple
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
np.savez('multiple.npz', first=arr1, second=arr2)

# Load multiple
data = np.load('multiple.npz')
loaded_first = data['first']
loaded_second = data['second']
```

---

## Text Format

```python
# Save as text
arr = np.array([1, 2, 3, 4, 5])
np.savetxt('array.txt', arr, fmt='%d')

# Save as CSV
matrix = np.array([[1.234, 2.567],
                   [3.891, 4.123]])
np.savetxt('matrix.csv', matrix, fmt='%.2f', delimiter=',')

# Load text
loaded = np.loadtxt('array.txt')

# Load CSV
loaded_csv = np.loadtxt('matrix.csv', delimiter=',')

# Load with header skip
loaded = np.loadtxt('file.csv', delimiter=',', skiprows=1)

# Handle missing data
loaded = np.genfromtxt('missing.csv', delimiter=',', filling_values=0)
```

---

## Format Comparison

| Aspect | Binary (.npy) | Text (.csv) |
|--------|---------------|------------|
| Speed | FAST (50-100x) | Slow |
| File Size | Small | Large |
| Readable | NO | YES |
| Excel | NO | YES |

---

## Lecture 3.08: QUIZ QUESTIONS & ANSWERS

**Q1: Save single array:**
- A: `np.save('file.npy', arr)`

**Q2: Load single array:**
- A: `np.load('file.npy')`

**Q3: Save multiple:**
- A: `np.savez('file.npz', arr1=x, arr2=y)`

**Q4: Load multiple:**
- A: `data=np.load('file.npz'); data['arr1']`

**Q5: Save as CSV:**
- A: `np.savetxt('file.csv', arr, delimiter=',')`

**Q6: Load CSV:**
- A: `np.loadtxt('file.csv', delimiter=',')`

**Q7: Faster format:**
- A: Binary (.npy, .npz)

**Q8: Excel format:**
- A: Text (.csv, .txt)

---

# FINAL REVISION CHECKLIST

## 3.01: Array Creation
- ✓ Know all functions (array, zeros, ones, full, eye, arange, linspace, random)
- ✓ Understand attributes (ndim, shape, size, dtype)
- ✓ Calculate nbytes = size × itemsize

## 3.02: Array vs List
- ✓ 50-100x faster
- ✓ Less memory usage
- ✓ Vectorized operations (no explicit loops needed)

## 3.03: Basic Operations
- ✓ Arithmetic element-wise
- ✓ Math functions (sqrt, exp, log, etc.)
- ✓ Aggregates with axis parameter
- ✓ Boolean indexing and filtering

## 3.04: Indexing & Slicing
- ✓ 0-based indexing
- ✓ Slicing syntax: [start:stop:step]
- ✓ 2D indexing: [row, col]
- ✓ Boolean masks for filtering
- ✓ Slicing creates views (linked to original)

## 3.05: Broadcasting & Reshaping
- ✓ reshape(-1) auto-infers dimension
- ✓ flatten() creates copy, ravel() creates view
- ✓ Transpose with .T
- ✓ argsort() returns sorted indices

## 3.06: Manipulation
- ✓ append, insert, delete operations
- ✓ repeat vs tile differences
- ✓ unique() for finding distinct values
- ✓ copy() vs view() for memory management
- ✓ concatenate for joining arrays

## 3.07: Stacking & Splitting
- ✓ vstack adds rows
- ✓ hstack adds columns
- ✓ dstack creates 3D arrays
- ✓ split, vsplit, hsplit break arrays

## 3.08: IO Operations
- ✓ save/load for single array (.npy)
- ✓ savez for multiple arrays (.npz)
- ✓ savetxt/loadtxt for text/CSV
- ✓ Binary format is fast, text is readable

---

# PRACTICE QUESTIONS (20 QUESTIONS)

## Basic Level

**Q1:** Create 3x3 identity matrix
- A: `np.eye(3)`

**Q2:** Create array from 0-10 with step 2
- A: `np.arange(0, 10, 2)`

**Q3:** Shape of `np.random.rand(4, 5)`?
- A: `(4, 5)` - 4 rows, 5 columns

**Q4:** Get 5th element (0-indexed)
- A: `arr[4]`

**Q5:** All elements greater than 50?
- A: `arr[arr > 50]`

## Intermediate Level

**Q6:** Reshape (2,6) to (3,4)
- A: `arr.reshape(3, 4)`

**Q7:** Sum all values column-wise
- A: `np.sum(arr, axis=0)`

**Q8:** Transpose a matrix
- A: `arr.T`

**Q9:** Repeat [1,2,3] each element twice
- A: `np.repeat([1,2,3], 2)` → [1 1 2 2 3 3]

**Q10:** Repeat [1,2,3] as whole array twice
- A: `np.tile([1,2,3], 2)` → [1 2 3 1 2 3]

## Advanced Level

**Q11:** Get unique values from array
- A: `np.unique(arr)`

**Q12:** Stack two arrays vertically
- A: `np.vstack((arr1, arr2))`

**Q13:** Stack two arrays horizontally
- A: `np.hstack((arr1, arr2))`

**Q14:** Split array into 3 equal parts
- A: `np.split(arr, 3)`

**Q15:** Save array to .npy file
- A: `np.save('file.npy', arr)`

## Expert Level

**Q16:** Load array from .npy file
- A: `np.load('file.npy')`

**Q17:** Save array as CSV
- A: `np.savetxt('file.csv', arr, delimiter=',')`

**Q18:** Which is faster - binary or text format?
- A: Binary format (.npy, .npz)

**Q19:** Difference between copy() and view()
- A: copy() is independent, view() is linked to original

**Q20:** Get mean of each row in 2D array
- A: `np.mean(arr, axis=1)`

---

# IMPORTANT FORMULAS & CONCEPTS

## Memory Calculation
- Total memory = size × itemsize
- For (5,5) int64 array: 5 × 5 × 8 = 200 bytes

## Array Operations
- Element-wise: all arithmetic operators work element-by-element
- Broadcasting: smaller arrays expand to match larger shapes
- Vectorization: eliminates explicit loops for better performance

## Key Differences
- arange: specify STEP size
- linspace: specify NUMBER of elements
- repeat: repeats each element
- tile: repeats whole array

## Important Concepts
- Slicing creates VIEWS (linked to original)
- copy() creates independent COPIES
- axis=0 operates on rows (column-wise)
- axis=1 operates on columns (row-wise)

---

# QUICK REFERENCE TABLE

| Task | Code | Output Type |
|------|------|-------------|
| Create zeros | `np.zeros((3,3))` | 3x3 matrix of 0s |
| Create ones | `np.ones((2,4))` | 2x4 matrix of 1s |
| Create range | `np.arange(5)` | [0 1 2 3 4] |
| Create linspace | `np.linspace(0,10,5)` | [0 2.5 5 7.5 10] |
| Sum column-wise | `np.sum(arr, axis=0)` | Vector of column sums |
| Sum row-wise | `np.sum(arr, axis=1)` | Vector of row sums |
| Transpose | `arr.T` | Transposed array |
| Flatten | `arr.flatten()` | 1D copy |
| Ravel | `arr.ravel()` | 1D view |
| Stack vertical | `np.vstack()` | Adds rows |
| Stack horizontal | `np.hstack()` | Adds columns |
| Save binary | `np.save()` | .npy file |
| Load binary | `np.load()` | From .npy file |

---

## Best of luck with your quiz! 💪

[translate:Jo cheez samajh nahi aaye - code run karke dekho. Practice se confidence aayega! Tum ready ho!]

