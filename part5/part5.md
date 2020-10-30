\# 排序，搜索和计数

\## 排序



\- `numpy.sort(a[, axis=-1, kind='quicksort', order=None])` Return a sorted ****copy**** of an array.

  \- axis：排序沿数组的（轴）方向，0表示按行，1表示按列，None表示展开来排序，默认为-1，表示沿最后的轴排序。

  \- kind：排序的算法，提供了快排'quicksort'、混排'mergesort'、堆排'heapsort'， 默认为‘quicksort'。

  \- order：排序的字段名，可指定字段排序，默认为None。





【例】

\```python

import numpy as np



np.random.seed(20200612)

x = np.random.rand(5, 5) * 10

x = np.around(x, 2)

print(x)

\# [[2.32 7.54 9.78 1.73 6.22]

\# [6.93 5.17 9.28 9.76 8.25]

\# [0.01 4.23 0.19 1.73 9.27]

\# [7.99 4.97 0.88 7.32 4.29]

\# [9.05 0.07 8.95 7.9 6.99]]



y = np.sort(x)

print(y)

\# [[1.73 2.32 6.22 7.54 9.78]

\# [5.17 6.93 8.25 9.28 9.76]

\# [0.01 0.19 1.73 4.23 9.27]

\# [0.88 4.29 4.97 7.32 7.99]

\# [0.07 6.99 7.9 8.95 9.05]]



y = np.sort(x, axis=0)

print(y)

\# [[0.01 0.07 0.19 1.73 4.29]

\# [2.32 4.23 0.88 1.73 6.22]

\# [6.93 4.97 8.95 7.32 6.99]

\# [7.99 5.17 9.28 7.9 8.25]

\# [9.05 7.54 9.78 9.76 9.27]]



y = np.sort(x, axis=1)

print(y)

\# [[1.73 2.32 6.22 7.54 9.78]

\# [5.17 6.93 8.25 9.28 9.76]

\# [0.01 0.19 1.73 4.23 9.27]

\# [0.88 4.29 4.97 7.32 7.99]

\# [0.07 6.99 7.9 8.95 9.05]]

\```



【例】

\```python

import numpy as np



dt = np.dtype([('name', 'S10'), ('age', np.int)])

a = np.array([("Mike", 21), ("Nancy", 25), ("Bob", 17), ("Jane", 27)], dtype=dt)

b = np.sort(a, order='name')

print(b)

\# [(b'Bob', 17) (b'Jane', 27) (b'Mike', 21) (b'Nancy', 25)]



b = np.sort(a, order='age')

print(b)

\# [(b'Bob', 17) (b'Mike', 21) (b'Nancy', 25) (b'Jane', 27)]

\```





如果排序后，想用元素的索引位置替代排序后的实际结果，该怎么办呢？



\- `numpy.argsort(a[, axis=-1, kind='quicksort', order=None])` Returns the indices that would sort an array.







【例】对数组沿给定轴执行间接排序，并使用指定排序类型返回数据的索引数组。这个索引数组用于构造排序后的数组。

\```python

import numpy as np



np.random.seed(20200612)

x = np.random.randint(0, 10, 10)

print(x)

\# [6 1 8 5 5 4 1 2 9 1]



y = np.argsort(x)

print(y)

\# [1 6 9 7 5 3 4 0 2 8]



print(x[y])

\# [1 1 1 2 4 5 5 6 8 9]



y = np.argsort(-x)

print(y)

\# [8 2 0 3 4 5 7 1 6 9]



print(x[y])

\# [9 8 6 5 5 4 2 1 1 1]

\```



【例】

\```python

import numpy as np



np.random.seed(20200612)

x = np.random.rand(5, 5) * 10

x = np.around(x, 2)

print(x)

\# [[2.32 7.54 9.78 1.73 6.22]

\# [6.93 5.17 9.28 9.76 8.25]

\# [0.01 4.23 0.19 1.73 9.27]

\# [7.99 4.97 0.88 7.32 4.29]

\# [9.05 0.07 8.95 7.9 6.99]]



y = np.argsort(x)

print(y)

\# [[3 0 4 1 2]

\# [1 0 4 2 3]

\# [0 2 3 1 4]

\# [2 4 1 3 0]

\# [1 4 3 2 0]]



y = np.argsort(x, axis=0)

print(y)

\# [[2 4 2 0 3]

\# [0 2 3 2 0]

\# [1 3 4 3 4]

\# [3 1 1 4 1]

\# [4 0 0 1 2]]



y = np.argsort(x, axis=1)

print(y)

\# [[3 0 4 1 2]

\# [1 0 4 2 3]

\# [0 2 3 1 4]

\# [2 4 1 3 0]

\# [1 4 3 2 0]]



y = np.array([np.take(x[i], np.argsort(x[i])) for i in range(5)])  

\#numpy.take(a, indices, axis=None, out=None, mode='raise')沿轴从数组中获取元素。

print(y)

\# [[1.73 2.32 6.22 7.54 9.78]

\# [5.17 6.93 8.25 9.28 9.76]

\# [0.01 0.19 1.73 4.23 9.27]

\# [0.88 4.29 4.97 7.32 7.99]

\# [0.07 6.99 7.9 8.95 9.05]]

\```



如何将数据按照某一指标进行排序呢？



\- `numpy.lexsort(keys[, axis=-1])` Perform an indirect stable sort using a sequence of keys.（使用键序列执行间接稳定排序。）



\- 给定多个可以在电子表格中解释为列的排序键，lexsort返回一个整数索引数组，该数组描述了按多个列排序的顺序。序列中的最后一个键用于主排序顺序，倒数第二个键用于辅助排序顺序，依此类推。keys参数必须是可以转换为相同形状的数组的对象序列。如果为keys参数提供了2D数组，则将其行解释为排序键，并根据最后一行，倒数第二行等进行排序。







【例】按照第一列的升序或者降序对整体数据进行排序。

\```python

import numpy as np



np.random.seed(20200612)

x = np.random.rand(5, 5) * 10

x = np.around(x, 2)

print(x)

\# [[2.32 7.54 9.78 1.73 6.22]

\# [6.93 5.17 9.28 9.76 8.25]

\# [0.01 4.23 0.19 1.73 9.27]

\# [7.99 4.97 0.88 7.32 4.29]

\# [9.05 0.07 8.95 7.9 6.99]]



index = np.lexsort([x[:, 0]])

print(index)

\# [2 0 1 3 4]



y = x[index]

print(y)

\# [[0.01 4.23 0.19 1.73 9.27]

\# [2.32 7.54 9.78 1.73 6.22]

\# [6.93 5.17 9.28 9.76 8.25]

\# [7.99 4.97 0.88 7.32 4.29]

\# [9.05 0.07 8.95 7.9 6.99]]



index = np.lexsort([-1 * x[:, 0]])

print(index)

\# [4 3 1 0 2]



y = x[index]

print(y)

\# [[9.05 0.07 8.95 7.9 6.99]

\# [7.99 4.97 0.88 7.32 4.29]

\# [6.93 5.17 9.28 9.76 8.25]

\# [2.32 7.54 9.78 1.73 6.22]

\# [0.01 4.23 0.19 1.73 9.27]]

\```



【例】

\```python

import numpy as np



x = np.array([1, 5, 1, 4, 3, 4, 4])

y = np.array([9, 4, 0, 4, 0, 2, 1])

a = np.lexsort([x])

b = np.lexsort([y])

print(a)

\# [0 2 4 3 5 6 1]

print(x[a])

\# [1 1 3 4 4 4 5]



print(b)

\# [2 4 6 5 1 3 0]

print(y[b])

\# [0 0 1 2 4 4 9]



z = np.lexsort([y, x])

print(z)

\# [2 0 4 6 5 3 1]

print(x[z])

\# [1 1 3 4 4 4 5]



z = np.lexsort([x, y])

print(z)

\# [2 4 6 5 3 1 0]

print(y[z])

\# [0 0 1 2 4 4 9]

\```



\- `numpy.partition(a, kth, axis=-1, kind='introselect', order=None)` Return a partitioned copy of an array.



Creates a copy of the array with its elements rearranged in such a way that the value of the element in k-th position is in the position it would be in a sorted array. All elements smaller than the k-th element are moved before this element and all equal or greater are moved behind it. The ordering of the elements in the two partitions is undefined.





【例】以索引是 kth 的元素为基准，将元素分成两部分，即大于该元素的放在其后面，小于该元素的放在其前面，这里有点类似于快排。



\```python

import numpy as np



np.random.seed(100)

x = np.random.randint(1, 30, [8, 3])

print(x)

\# [[ 9 25 4]

\# [ 8 24 16]

\# [17 11 21]

\# [ 3 22 3]

\# [ 3 15 3]

\# [18 17 25]

\# [16 5 12]

\# [29 27 17]]



y = np.sort(x, axis=0)

print(y)

\# [[ 3 5 3]

\# [ 3 11 3]

\# [ 8 15 4]

\# [ 9 17 12]

\# [16 22 16]

\# [17 24 17]

\# [18 25 21]

\# [29 27 25]]



z = np.partition(x, kth=2, axis=0)

print(z)

\# [[ 3 5 3]

\# [ 3 11 3]

\# [ 8 15 4]

\# [ 9 22 21]

\# [17 24 16]

\# [18 17 25]

\# [16 25 12]

\# [29 27 17]]

\```

【例】选取每一列第三小的数



\```python

import numpy as np



np.random.seed(100)

x = np.random.randint(1, 30, [8, 3])

print(x)

\# [[ 9 25 4]

\# [ 8 24 16]

\# [17 11 21]

\# [ 3 22 3]

\# [ 3 15 3]

\# [18 17 25]

\# [16 5 12]

\# [29 27 17]]

z = np.partition(x, kth=2, axis=0)

print(z[2])

\# [ 8 15 4]

\```



【例】选取每一列第三大的数据



\```python

import numpy as np



np.random.seed(100)

x = np.random.randint(1, 30, [8, 3])

print(x)

\# [[ 9 25 4]

\# [ 8 24 16]

\# [17 11 21]

\# [ 3 22 3]

\# [ 3 15 3]

\# [18 17 25]

\# [16 5 12]

\# [29 27 17]]

z = np.partition(x, kth=-3, axis=0)

print(z[-3])

\# [17 24 17]

\```





\- `numpy.argpartition(a, kth, axis=-1, kind='introselect', order=None)`



Perform an indirect partition along the given axis using the algorithm specified by the `kind` keyword. It returns an array of indices of the same shape as `a` that index data along the given axis in partitioned order.



【例】



\```python

import numpy as np



np.random.seed(100)

x = np.random.randint(1, 30, [8, 3])

print(x)

\# [[ 9 25 4]

\# [ 8 24 16]

\# [17 11 21]

\# [ 3 22 3]

\# [ 3 15 3]

\# [18 17 25]

\# [16 5 12]

\# [29 27 17]]



y = np.argsort(x, axis=0)

print(y)

\# [[3 6 3]

\# [4 2 4]

\# [1 4 0]

\# [0 5 6]

\# [6 3 1]

\# [2 1 7]

\# [5 0 2]

\# [7 7 5]]



z = np.argpartition(x, kth=2, axis=0)

print(z)

\# [[3 6 3]

\# [4 2 4]

\# [1 4 0]

\# [0 3 2]

\# [2 1 1]

\# [5 5 5]

\# [6 0 6]

\# [7 7 7]]

\```



【例】选取每一列第三小的数的索引



\```python

import numpy as np



np.random.seed(100)

x = np.random.randint(1, 30, [8, 3])

print(x)

\# [[ 9 25 4]

\# [ 8 24 16]

\# [17 11 21]

\# [ 3 22 3]

\# [ 3 15 3]

\# [18 17 25]

\# [16 5 12]

\# [29 27 17]]



z = np.argpartition(x, kth=2, axis=0)

print(z[2])

\# [1 4 0]

\```



【例】选取每一列第三大的数的索引

\```python

import numpy as np



np.random.seed(100)

x = np.random.randint(1, 30, [8, 3])

print(x)

\# [[ 9 25 4]

\# [ 8 24 16]

\# [17 11 21]

\# [ 3 22 3]

\# [ 3 15 3]

\# [18 17 25]

\# [16 5 12]

\# [29 27 17]]



z = np.argpartition(x, kth=-3, axis=0)

print(z[-3])

\# [2 1 7]

\```





\---

\## 搜索



\- `numpy.argmax(a[, axis=None, out=None])`Returns the indices of the maximum values along an axis.



【例】

\```python

import numpy as np



np.random.seed(20200612)

x = np.random.rand(5, 5) * 10

x = np.around(x, 2)

print(x)

\# [[2.32 7.54 9.78 1.73 6.22]

\# [6.93 5.17 9.28 9.76 8.25]

\# [0.01 4.23 0.19 1.73 9.27]

\# [7.99 4.97 0.88 7.32 4.29]

\# [9.05 0.07 8.95 7.9 6.99]]



y = np.argmax(x)

print(y) # 2



y = np.argmax(x, axis=0)

print(y)

\# [4 0 0 1 2]



y = np.argmax(x, axis=1)

print(y)

\# [2 3 4 0 0]

\```



\- `numpy.argmin(a[, axis=None, out=None])`Returns the indices of the minimum values along an axis.





【例】

\```python

import numpy as np



np.random.seed(20200612)

x = np.random.rand(5, 5) * 10

x = np.around(x, 2)

print(x)

\# [[2.32 7.54 9.78 1.73 6.22]

\# [6.93 5.17 9.28 9.76 8.25]

\# [0.01 4.23 0.19 1.73 9.27]

\# [7.99 4.97 0.88 7.32 4.29]

\# [9.05 0.07 8.95 7.9 6.99]]



y = np.argmin(x)

print(y) # 10



y = np.argmin(x, axis=0)

print(y)

\# [2 4 2 0 3]



y = np.argmin(x, axis=1)

print(y)

\# [3 1 0 2 1]

\```





\- `numppy.nonzero(a)` Return the indices of the elements that are non-zero.



，其值为非零元素的下标在对应轴上的值。



\1. 只有`a`中非零元素才会有索引值，那些零值元素没有索引值。

\2. 返回一个长度为`a.ndim`的元组（tuple），元组的每个元素都是一个整数数组（array）。

\3. 每一个array均是从一个维度上来描述其索引值。比如，如果`a`是一个二维数组，则tuple包含两个array，第一个array从行维度来描述索引值；第二个array从列维度来描述索引值。

\4. 该 `np.transpose(np.nonzero(x))` 函数能够描述出每一个非零元素在不同维度的索引值。

\5. 通过`a[nonzero(a)]`得到所有`a`中的非零值。



【例】一维数组

\```python

import numpy as np



x = np.array([0, 2, 3])

print(x) # [0 2 3]

print(x.shape) # (3,)

print(x.ndim) # 1



y = np.nonzero(x)

print(y) # (array([1, 2], dtype=int64),)

print(np.array(y)) # [[1 2]]

print(np.array(y).shape) # (1, 2)

print(np.array(y).ndim) # 2

print(np.transpose(y))

\# [[1]

\# [2]]

print(x[np.nonzero(x)])

\#[2, 3]

\```



【例】二维数组

\```python

import numpy as np



x = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]])

print(x)

\# [[3 0 0]

\# [0 4 0]

\# [5 6 0]]

print(x.shape) # (3, 3)

print(x.ndim) # 2



y = np.nonzero(x)

print(y)

\# (array([0, 1, 2, 2], dtype=int64), array([0, 1, 0, 1], dtype=int64))

print(np.array(y))

\# [[0 1 2 2]

\# [0 1 0 1]]

print(np.array(y).shape) # (2, 4)

print(np.array(y).ndim) # 2



y = x[np.nonzero(x)]

print(y) # [3 4 5 6]



y = np.transpose(np.nonzero(x))

print(y)

\# [[0 0]

\# [1 1]

\# [2 0]

\# [2 1]]

\```

【例】三维数组

\```python

import numpy as np



x = np.array([[[0, 1], [1, 0]], [[0, 1], [1, 0]], [[0, 0], [1, 0]]])

print(x)

\# [[[0 1]

\#  [1 0]]

\#

\# [[0 1]

\#  [1 0]]

\#

\# [[0 0]

\#  [1 0]]]

print(np.shape(x)) # (3, 2, 2)

print(x.ndim) # 3



y = np.nonzero(x)

print(np.array(y))

\# [[0 0 1 1 2]

\# [0 1 0 1 1]

\# [1 0 1 0 0]]

print(np.array(y).shape) # (3, 5)

print(np.array(y).ndim) # 2

print(y)

\# (array([0, 0, 1, 1, 2], dtype=int64), array([0, 1, 0, 1, 1], dtype=int64), array([1, 0, 1, 0, 0], dtype=int64))

print(x[np.nonzero(x)])

\#[1 1 1 1 1]

\```



【例】`nonzero()`将布尔数组转换成整数数组进行操作。

\```python

import numpy as np



x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(x)

\# [[1 2 3]

\# [4 5 6]

\# [7 8 9]]



y = x > 3

print(y)

\# [[False False False]

\# [ True True True]

\# [ True True True]]



y = np.nonzero(x > 3)

print(y)

\# (array([1, 1, 1, 2, 2, 2], dtype=int64), array([0, 1, 2, 0, 1, 2], dtype=int64))



y = x[np.nonzero(x > 3)]

print(y)

\# [4 5 6 7 8 9]



y = x[x > 3]

print(y)

\# [4 5 6 7 8 9]

\```



\- `numpy.where(condition, [x=None, y=None])` Return elements chosen from `x` or `y` depending on `condition`.



【例】满足条件`condition`，输出`x`，不满足输出`y`。

\```python

import numpy as np



x = np.arange(10)

print(x)

\# [0 1 2 3 4 5 6 7 8 9]



y = np.where(x < 5, x, 10 * x)

print(y)

\# [ 0 1 2 3 4 50 60 70 80 90]



x = np.array([[0, 1, 2],

​       [0, 2, 4],

​       [0, 3, 6]])

y = np.where(x < 4, x, -1)

print(y)

\# [[ 0 1 2]

\# [ 0 2 -1]

\# [ 0 3 -1]]

\```



【例】只有`condition`，没有`x`和`y`，则输出满足条件 (即非0) 元素的坐标 (等价于`numpy.nonzero`)。这里的坐标以tuple的形式给出，通常原数组有多少维，输出的tuple中就包含几个数组，分别对应符合条件元素的各维坐标。

\```python

import numpy as np



x = np.array([1, 2, 3, 4, 5, 6, 7, 8])

y = np.where(x > 5)

print(y)

\# (array([5, 6, 7], dtype=int64),)

print(x[y])

\# [6 7 8]



y = np.nonzero(x > 5)

print(y)

\# (array([5, 6, 7], dtype=int64),)

print(x[y])

\# [6 7 8]



x = np.array([[11, 12, 13, 14, 15],

​       [16, 17, 18, 19, 20],

​       [21, 22, 23, 24, 25],

​       [26, 27, 28, 29, 30],

​       [31, 32, 33, 34, 35]])

y = np.where(x > 25)

print(y)

\# (array([3, 3, 3, 3, 3, 4, 4, 4, 4, 4], dtype=int64), array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=int64))



print(x[y])

\# [26 27 28 29 30 31 32 33 34 35]



y = np.nonzero(x > 25)

print(y)

\# (array([3, 3, 3, 3, 3, 4, 4, 4, 4, 4], dtype=int64), array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=int64))

print(x[y])

\# [26 27 28 29 30 31 32 33 34 35]

\```









\- `numpy.searchsorted(a, v[, side='left', sorter=None])` Find indices where elements should be inserted to maintain order.

  \- a：一维输入数组。当`sorter`参数为`None`的时候，`a`必须为升序数组；否则，`sorter`不能为空，存放`a`中元素的`index`，用于反映`a`数组的升序排列方式。

  \- v：插入`a`数组的值，可以为单个元素，`list`或者`ndarray`。

  \- side：查询方向，当为`left`时，将返回第一个符合条件的元素下标；当为`right`时，将返回最后一个符合条件的元素下标。

  \- sorter：一维数组存放`a`数组元素的 index，index 对应元素为升序。







【例】

\```python

import numpy as np



x = np.array([0, 1, 5, 9, 11, 18, 26, 33])

y = np.searchsorted(x, 15)

print(y) # 5



y = np.searchsorted(x, 15, side='right')

print(y) # 5



y = np.searchsorted(x, -1)

print(y) # 0



y = np.searchsorted(x, -1, side='right')

print(y) # 0



y = np.searchsorted(x, 35)

print(y) # 8



y = np.searchsorted(x, 35, side='right')

print(y) # 8



y = np.searchsorted(x, 11)

print(y) # 4



y = np.searchsorted(x, 11, side='right')

print(y) # 5



y = np.searchsorted(x, 0)

print(y) # 0



y = np.searchsorted(x, 0, side='right')

print(y) # 1



y = np.searchsorted(x, 33)

print(y) # 7



y = np.searchsorted(x, 33, side='right')

print(y) # 8

\```



【例】

\```python

import numpy as np



x = np.array([0, 1, 5, 9, 11, 18, 26, 33])

y = np.searchsorted(x, [-1, 0, 11, 15, 33, 35])

print(y) # [0 0 4 5 7 8]



y = np.searchsorted(x, [-1, 0, 11, 15, 33, 35], side='right')

print(y) # [0 1 5 5 8 8]

\```



【例】

\```python

import numpy as np



x = np.array([0, 1, 5, 9, 11, 18, 26, 33])

np.random.shuffle(x)

print(x) # [33 1 9 18 11 26 0 5]



x_sort = np.argsort(x)

print(x_sort) # [6 1 7 2 4 3 5 0]



y = np.searchsorted(x, [-1, 0, 11, 15, 33, 35], sorter=x_sort)

print(y) # [0 0 4 5 7 8]



y = np.searchsorted(x, [-1, 0, 11, 15, 33, 35], side='right', sorter=x_sort)

print(y) # [0 1 5 5 8 8]

\```





\---

\## 计数



\- `numpy.count_nonzero(a, axis=None)` Counts the number of non-zero values in the array a.



【例】返回数组中的非0元素个数。

\```python

import numpy as np



x = np.count_nonzero(np.eye(4))

print(x) # 4



x = np.count_nonzero([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]])

print(x) # 5



x = np.count_nonzero([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]], axis=0)

print(x) # [1 1 1 1 1]



x = np.count_nonzero([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]], axis=1)

print(x) # [2 3]

\```



\---

****参考文献****



\- https://blog.csdn.net/u013698770/article/details/54632047

\- https://www.cnblogs.com/massquantity/p/8908859.html





\# 集合操作



\## 构造集合



\- `numpy.unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None)` Find the unique elements of an array.

  \- `return_index=True` 表示返回新列表元素在旧列表中的位置。

  \- `return_inverse=True`表示返回旧列表元素在新列表中的位置。

  \- `return_counts=True`表示返回新列表元素在旧列表中出现的次数。

   

   

【例】找出数组中的唯一值并返回已排序的结果。

\```python

import numpy as np



x = np.unique([1, 1, 3, 2, 3, 3])

print(x) # [1 2 3]



x = sorted(set([1, 1, 3, 2, 3, 3]))

print(x) # [1, 2, 3]



x = np.array([[1, 1], [2, 3]])

u = np.unique(x)

print(u) # [1 2 3]



x = np.array([[1, 0, 0], [1, 0, 0], [2, 3, 4]])

y = np.unique(x, axis=0)

print(y)

\# [[1 0 0]

\# [2 3 4]]



x = np.array(['a', 'b', 'b', 'c', 'a'])

u, index = np.unique(x, return_index=True)

print(u) # ['a' 'b' 'c']

print(index) # [0 1 3]

print(x[index]) # ['a' 'b' 'c']



x = np.array([1, 2, 6, 4, 2, 3, 2])

u, index = np.unique(x, return_inverse=True)

print(u) # [1 2 3 4 6]

print(index) # [0 1 4 3 1 2 1]

print(u[index]) # [1 2 6 4 2 3 2]



u, count = np.unique(x, return_counts=True)

print(u) # [1 2 3 4 6]

print(count) # [1 3 1 1 1]

\```



\## 布尔运算



\- `numpy.in1d(ar1, ar2, assume_unique=False, invert=False)` Test whether each element of a 1-D array is also present in a second array.



Returns a boolean array the same length as `ar1` that is True where an element of `ar1` is in `ar2` and False otherwise.



【例】前面的数组是否包含于后面的数组，返回布尔值。返回的值是针对第一个参数的数组的，所以维数和第一个参数一致，布尔值与数组的元素位置也一一对应。

\```python

import numpy as np



test = np.array([0, 1, 2, 5, 0])

states = [0, 2]

mask = np.in1d(test, states)

print(mask) # [ True False True False True]

print(test[mask]) # [0 2 0]



mask = np.in1d(test, states, invert=True)

print(mask) # [False True False True False]

print(test[mask]) # [1 5]

\```





\### 求两个集合的交集：



\- `numpy.intersect1d(ar1, ar2, assume_unique=False, return_indices=False)` Find the intersection of two arrays.



Return the sorted, unique values that are in both of the input arrays.



【例】求两个数组的唯一化+求交集+排序函数。

\```python

import numpy as np

from functools import reduce



x = np.intersect1d([1, 3, 4, 3], [3, 1, 2, 1])

print(x) # [1 3]



x = np.array([1, 1, 2, 3, 4])

y = np.array([2, 1, 4, 6])

xy, x_ind, y_ind = np.intersect1d(x, y, return_indices=True)

print(x_ind) # [0 2 4]

print(y_ind) # [1 0 2]

print(xy) # [1 2 4]

print(x[x_ind]) # [1 2 4]

print(y[y_ind]) # [1 2 4]



x = reduce(np.intersect1d, ([1, 3, 4, 3], [3, 1, 2, 1], [6, 3, 4, 2]))

print(x) # [3]

\```



\### 求两个集合的并集：



\- `numpy.union1d(ar1, ar2)` Find the union of two arrays.



Return the unique, sorted array of values that are in either of the two input arrays.



【例】计算两个集合的并集，唯一化并排序。

\```python

import numpy as np

from functools import reduce



x = np.union1d([-1, 0, 1], [-2, 0, 2])

print(x) # [-2 -1 0 1 2]

x = reduce(np.union1d, ([1, 3, 4, 3], [3, 1, 2, 1], [6, 3, 4, 2]))

print(x) # [1 2 3 4 6]

'''

functools.reduce(function, iterable[, initializer])

将两个参数的 function 从左至右积累地应用到 iterable 的条目，以便将该可迭代对象缩减为单一的值。 例如，reduce(lambda x, y: x+y, [1, 2, 3, 4, 5]) 是计算 ((((1+2)+3)+4)+5) 的值。 左边的参数 x 是积累值而右边的参数 y 则是来自 iterable 的更新值。 如果存在可选项 initializer，它会被放在参与计算的可迭代对象的条目之前，并在可迭代对象为空时作为默认值。 如果没有给出 initializer 并且 iterable 仅包含一个条目，则将返回第一项。



大致相当于：

def reduce(function, iterable, initializer=None):

  it = iter(iterable)

  if initializer is None:

​    value = next(it)

  else:

​    value = initializer

  for element in it:

​    value = function(value, element)

  return value

'''

\```



\### 求两个集合的差集：



\- `numpy.setdiff1d(ar1, ar2, assume_unique=False)` Find the set difference of two arrays.





Return the unique values in `ar1` that are not in `ar2`.



【例】集合的差，即元素存在于第一个函数不存在于第二个函数中。

\```python

import numpy as np



a = np.array([1, 2, 3, 2, 4, 1])

b = np.array([3, 4, 5, 6])

x = np.setdiff1d(a, b)

print(x) # [1 2]

\```



\### 求两个集合的异或：



\- `setxor1d(ar1, ar2, assume_unique=False)` Find the set exclusive-or of two arrays.



【例】集合的对称差，即两个集合的交集的补集。简言之，就是两个数组中各自独自拥有的元素的集合。



\```python

import numpy as np



a = np.array([1, 2, 3, 2, 4, 1])

b = np.array([3, 4, 5, 6])

x = np.setxor1d(a, b)

print(x) # [1 2 5 6]

\```







\---

****参考文献****

\- https://www.jianshu.com/p/3bfe21aa1adb