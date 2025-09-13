# Data Structures and Algorithms in Python: Key Patterns and Implementations

## Core Data Structures in Python

### 1. Arrays and Lists

```python
# Basic list operations
nums = [1, 2, 3, 4, 5]
nums.append(6)        # Add element to end
nums.insert(0, 0)     # Insert at specific position
nums.pop()            # Remove and return last element
nums.remove(3)        # Remove specific element
nums[2] = 10          # Update element

# List slicing
sub_list = nums[1:4]  # Elements from index 1 to 3

```

### 2. Linked Lists

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, val):
        if not self.head:
            self.head = ListNode(val)
            return
        
        current = self.head
        while current.next:
            current = current.next
        current.next = ListNode(val)
    
    def print_list(self):
        current = self.head
        while current:
            print(current.val, end=" -> ")
            current = current.next
        print("None")

```

### 3. Stacks and Queues

```python
# Stack implementation using list
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
    
    def peek(self):
        if not self.is_empty():
            return self.items[-1]
    
    def is_empty(self):
        return len(self.items) == 0

# Queue implementation using collections.deque
from collections import deque

class Queue:
    def __init__(self):
        self.items = deque()
    
    def enqueue(self, item):
        self.items.append(item)
    
    def dequeue(self):
        if not self.is_empty():
            return self.items.popleft()
    
    def is_empty(self):
        return len(self.items) == 0

```

### 4. Hash Tables (Dictionaries)

```python
# Dictionary operations
student = {
    "name": "John",
    "age": 21,
    "courses": ["Math", "CS"]
}

# Access, modify, add
print(student["name"])
student["age"] = 22
student["gpa"] = 3.8

# Check if key exists
if "name" in student:
    print("Name exists")

# Dictionary comprehension
squares = {x: x*x for x in range(6)}

```

### 5. Trees

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class BinarySearchTree:
    def __init__(self):
        self.root = None
    
    def insert(self, val):
        if not self.root:
            self.root = TreeNode(val)
            return
        
        def _insert(node, val):
            if val < node.val:
                if node.left:
                    _insert(node.left, val)
                else:
                    node.left = TreeNode(val)
            else:
                if node.right:
                    _insert(node.right, val)
                else:
                    node.right = TreeNode(val)
        
        _insert(self.root, val)
    
    # In-order traversal
    def inorder(self):
        result = []
        
        def _inorder(node):
            if node:
                _inorder(node.left)
                result.append(node.val)
                _inorder(node.right)
        
        _inorder(self.root)
        return result

```

### 6. Heaps

```python
import heapq

# Min heap operations
min_heap = []
heapq.heappush(min_heap, 5)
heapq.heappush(min_heap, 3)
heapq.heappush(min_heap, 7)
print(heapq.heappop(min_heap))  # Returns 3 (smallest element)

# Max heap simulation (negate values)
max_heap = []
heapq.heappush(max_heap, -5)
heapq.heappush(max_heap, -3)
heapq.heappush(max_heap, -7)
print(-heapq.heappop(max_heap))  # Returns 7 (largest element)

# Heapify an existing list
nums = [5, 7, 9, 1, 3]
heapq.heapify(nums)  # Converts nums to a min heap

```

### 7. Graphs

```python
# Adjacency list representation
class Graph:
    def __init__(self):
        self.graph = {}
    
    def add_edge(self, u, v, directed=False):
        if u not in self.graph:
            self.graph[u] = []
        if v not in self.graph:
            self.graph[v] = []
        
        self.graph[u].append(v)
        if not directed:
            self.graph[v].append(u)
    
    def bfs(self, start):
        visited = set()
        queue = deque([start])
        visited.add(start)
        result = []
        
        while queue:
            vertex = queue.popleft()
            result.append(vertex)
            
            for neighbor in self.graph[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return result
    
    def dfs(self, start):
        visited = set()
        result = []
        
        def _dfs(vertex):
            visited.add(vertex)
            result.append(vertex)
            
            for neighbor in self.graph[vertex]:
                if neighbor not in visited:
                    _dfs(neighbor)
        
        _dfs(start)
        return result

```

## Common Algorithm Patterns

### 1. Two Pointers

Two pointers technique involves using multiple pointers to solve problems, often reducing time complexity.

```python
# Example: Find if array has a pair with target sum
def has_pair_with_sum(arr, target):
    left, right = 0, len(arr) - 1
    
    while left < right:
        current_sum = arr[left] + arr[right]
        
        if current_sum == target:
            return True
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return False

# Example: Remove duplicates from sorted array
def remove_duplicates(nums):
    if not nums:
        return 0
        
    slow = 0
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    
    return slow + 1

```

### 2. Sliding Window

Sliding window pattern is used to process sequential data by maintaining a "window" of elements.

```python
# Example: Find maximum sum subarray of size k
def max_sum_subarray(arr, k):
    max_sum = 0
    window_sum = 0
    start = 0
    
    for end in range(len(arr)):
        window_sum += arr[end]
        
        if end >= k - 1:
            max_sum = max(max_sum, window_sum)
            window_sum -= arr[start]
            start += 1
    
    return max_sum

# Example: Longest substring with k distinct characters
def longest_substring_with_k_distinct(s, k):
    char_frequency = {}
    max_length = 0
    start = 0
    
    for end in range(len(s)):
        right_char = s[end]
        
        if right_char not in char_frequency:
            char_frequency[right_char] = 0
        char_frequency[right_char] += 1
        
        while len(char_frequency) > k:
            left_char = s[start]
            char_frequency[left_char] -= 1
            if char_frequency[left_char] == 0:
                del char_frequency[left_char]
            start += 1
        
        max_length = max(max_length, end - start + 1)
    
    return max_length

```

### 3. Binary Search

Binary search is used to efficiently search sorted arrays.

```python
# Example: Standard binary search
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

# Example: Find first and last position of element
def search_range(nums, target):
    def find_first():
        left, right = 0, len(nums) - 1
        first = -1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if nums[mid] == target:
                first = mid
                right = mid - 1
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return first
    
    def find_last():
        left, right = 0, len(nums) - 1
        last = -1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if nums[mid] == target:
                last = mid
                left = mid + 1
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return last
    
    return [find_first(), find_last()]

```

### 4. Depth-First Search (DFS)

DFS explores as far as possible along each branch before backtracking.

```python
# Example: Number of islands
def num_islands(grid):
    if not grid:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    count = 0
    
    def dfs(r, c):
        if (r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == '0'):
            return
        
        # Mark as visited
        grid[r][c] = '0'
        
        # Explore neighbors
        dfs(r+1, c)
        dfs(r-1, c)
        dfs(r, c+1)
        dfs(r, c-1)
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                dfs(r, c)
    
    return count

```

### 5. Breadth-First Search (BFS)

BFS explores all neighbors at the present depth before moving to nodes at the next depth level.

```python
# Example: Level order traversal of binary tree
def level_order_traversal(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(current_level)
    
    return result

```

### 6. Dynamic Programming

Dynamic programming breaks down complex problems into simpler subproblems.

```python
# Example: Fibonacci sequence (memoization)
def fibonacci(n, memo={}):
    if n in memo:
        return memo[n]
    
    if n <= 1:
        return n
    
    memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)
    return memo[n]

# Example: 0/1 Knapsack problem
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(
                    values[i-1] + dp[i-1][w-weights[i-1]],
                    dp[i-1][w]
                )
            else:
                dp[i][w] = dp[i-1][w]
    
    return dp[n][capacity]

```

### 7. Backtracking

Backtracking explores all possible solutions by trying different paths and backtracking when needed.

```python
# Example: Generate all permutations
def permute(nums):
    result = []
    
    def backtrack(start):
        if start == len(nums):
            result.append(nums[:])
            return
        
        for i in range(start, len(nums)):
            # Swap to create a new combination
            nums[start], nums[i] = nums[i], nums[start]
            
            # Recurse with the new combination
            backtrack(start + 1)
            
            # Backtrack (undo the swap)
            nums[start], nums[i] = nums[i], nums[start]
    
    backtrack(0)
    return result

# Example: N-Queens problem
def solve_n_queens(n):
    result = []
    board = [['.'] * n for _ in range(n)]
    
    def is_safe(row, col):
        # Check column
        for i in range(row):
            if board[i][col] == 'Q':
                return False
        
        # Check upper left diagonal
        for i, j in zip(range(row-1, -1, -1), range(col-1, -1, -1)):
            if board[i][j] == 'Q':
                return False
        
        # Check upper right diagonal
        for i, j in zip(range(row-1, -1, -1), range(col+1, n)):
            if board[i][j] == 'Q':
                return False
        
        return True
    
    def backtrack(row):
        if row == n:
            result.append([''.join(row) for row in board])
            return
        
        for col in range(n):
            if is_safe(row, col):
                board[row][col] = 'Q'
                backtrack(row + 1)
                board[row][col] = '.'  # Backtrack
    
    backtrack(0)
    return result

```

### 8. Greedy Algorithms

Greedy algorithms make locally optimal choices at each step with the hope of finding a global optimum.

```python
# Example: Activity selection problem
def activity_selection(start, finish):
    # Sort activities by finish time
    activities = sorted(zip(start, finish), key=lambda x: x[1])
    
    selected = [activities[0]]
    last_finish_time = activities[0][1]
    
    for i in range(1, len(activities)):
        if activities[i][0] >= last_finish_time:
            selected.append(activities[i])
            last_finish_time = activities[i][1]
    
    return selected

# Example: Coin change (greedy approach - works only for certain coin systems)
def min_coins(coins, amount):
    coins.sort(reverse=True)
    count = 0
    
    for coin in coins:
        while amount >= coin:
            amount -= coin
            count += 1
    
    return count if amount == 0 else -1

```

## Time and Space Complexity Analysis

| **Data Structure** | **Access** | **Search** | **Insert** | **Delete** | **Space** |
| --- | --- | --- | --- | --- | --- |
| Array | O(1) | O(n) | O(n) | O(n) | O(n) |
| Linked List | O(n) | O(n) | O(1) | O(1) | O(n) |
| Stack | O(n) | O(n) | O(1) | O(1) | O(n) |
| Queue | O(n) | O(n) | O(1) | O(1) | O(n) |
| Hash Table | N/A | O(1)* | O(1)* | O(1)* | O(n) |
| Binary Search Tree | O(log n)* | O(log n)* | O(log n)* | O(log n)* | O(n) |
| Heap | O(1)** | O(n) | O(log n) | O(log n) | O(n) |
- Average case, can be O(n) in worst case

** Only for min/max element

## Problem-Solving Strategies

### 1. Understand the Problem

- Read the problem statement carefully
- Identify input and output requirements
- Consider edge cases and constraints
- Ask clarifying questions if needed

### 2. Develop a Plan

- Start with a brute force approach
- Identify applicable patterns and algorithms
- Consider space-time tradeoffs
- Optimize your approach

### 3. Implement the Solution

- Write clean, readable code
- Use meaningful variable names
- Modularize your code
- Add comments for complex logic

### 4. Test and Debug

- Start with simple test cases
- Include edge cases
- Check for off-by-one errors
- Trace through your algorithm with examples

## Advanced DSA Topics

### 1. Disjoint Set (Union-Find)

```python
class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

```

### 2. Segment Trees

```python
class SegmentTree:
    def __init__(self, arr):
        self.n = len(arr)
        # Size of segment tree array
        self.tree = [0] * (4 * self.n)
        if self.n > 0:
            self._build_tree(arr, 0, 0, self.n - 1)
    
    def _build_tree(self, arr, node, start, end):
        if start == end:
            self.tree[node] = arr[start]
            return
        
        mid = (start + end) // 2
        # Build left and right subtrees
        self._build_tree(arr, 2 * node + 1, start, mid)
        self._build_tree(arr, 2 * node + 2, mid + 1, end)
        # Current node value is sum of children
        self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]
    
    def query(self, start, end):
        return self._query(0, 0, self.n - 1, start, end)
    
    def _query(self, node, node_start, node_end, query_start, query_end):
        if query_start > node_end or query_end < node_start:
            return 0  # Outside range
        
        if query_start <= node_start and node_end <= query_end:
            return self.tree[node]  # Inside range
        
        # Partial overlap
        mid = (node_start + node_end) // 2
        left_sum = self._query(2 * node + 1, node_start, mid, query_start, query_end)
        right_sum = self._query(2 * node + 2, mid + 1, node_end, query_start, query_end)
        return left_sum + right_sum
    
    def update(self, index, value):
        self._update(0, 0, self.n - 1, index, value)
    
    def _update(self, node, start, end, index, value):
        if start == end:
            self.tree[node] = value
            return
        
        mid = (start + end) // 2
        if index <= mid:
            self._update(2 * node + 1, start, mid, index, value)
        else:
            self._update(2 * node + 2, mid + 1, end, index, value)
        
        self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]

```

### 3. Trie (Prefix Tree)

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
    
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word
    
    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

```

## Common Coding Interview Questions by Pattern

### 1. Array/String Problems

- Two Sum / Three Sum / Four Sum
- Container With Most Water
- Longest Substring Without Repeating Characters
- String to Integer (atoi)
- Valid Palindrome

### 2. Linked List Problems

- Reverse Linked List
- Detect Cycle in Linked List
- Merge Two Sorted Lists
- LRU Cache
- Remove Nth Node From End of List

### 3. Tree Problems

- Maximum Depth of Binary Tree
- Validate Binary Search Tree
- Binary Tree Level Order Traversal
- Lowest Common Ancestor
- Serialize and Deserialize Binary Tree

### 4. Graph Problems

- Clone Graph
- Course Schedule (Topological Sort)
- Network Delay Time (Shortest Path)
- Number of Islands
- Word Ladder

### 5. Dynamic Programming Problems

- Climbing Stairs
- Coin Change
- Longest Increasing Subsequence
- Edit Distance
- Maximum Subarray

## Practice Resources

### 1. Online Platforms

- LeetCode - Extensive collection of problems with difficulty ratings
- HackerRank - Problems organized by domains and skills
- CodeSignal - Interactive coding challenges and assessments
- AlgoExpert - Curated list of interview questions with video explanations

### 2. Books

- "Cracking the Coding Interview" by Gayle Laakmann McDowell
- "Introduction to Algorithms" by Cormen, Leiserson, Rivest, and Stein
- "Algorithms" by Robert Sedgewick and Kevin Wayne
- "Elements of Programming Interviews in Python" by Aziz, Lee, and Prakash

### 3. GitHub Repositories

- TheAlgorithms/Python - Collection of algorithms implemented in Python
- keon/algorithms - Minimal examples of data structures and algorithms
- donnemartin/interactive-coding-challenges - Interactive Python challenges
