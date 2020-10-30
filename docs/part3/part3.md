# 数组操作

## 更改形状

在对数组进行操作时，为了满足格式和计算的要求通常会改变其形状。

- `numpy.ndarray.shape`表示数组的维度，返回一个元组，这个元组的长度就是维度的数目，即 `ndim` 属性(秩)。

【例】通过修改 shap 属性来改变数组的形状。
```python
import numpy as np

x = np.array([1, 2, 9, 4, 5, 6, 7, 8])
print(x.shape)  # (8,)
x.shape = [2, 4]
print(x)
# [[1 2 9 4]
#  [5 6 7 8]]
```

- `numpy.ndarray.flat` 将数组转换为一维的迭代器，可以用for访问数组每一个元素。

【例】
```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = x.flat
print(y)
# <numpy.flatiter object at 0x0000020F9BA10C60>
for i in y:
    print(i, end=' ')
# 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35

y[3] = 0
print(end='\n')
print(x)
# [[11 12 13  0 15]
#  [16 17 18 19 20]
#  [21 22 23 24 25]
#  [26 27 28 29 30]
#  [31 32 33 34 35]]
```

- `numpy.ndarray.flatten([order='C'])` 将数组的副本转换为一维数组，并返回。
    - order：'C' -- 按行，'F' -- 按列，'A' -- 原顺序，'k' -- 元素在内存中的出现顺序。(简记)
    - order：{'C / F，'A，K}，可选使用此索引顺序读取a的元素。'C'意味着以行大的C风格顺序对元素进行索引，最后一个轴索引会更改F表示以列大的Fortran样式顺序索引元素，其中第一个索引变化最快，最后一个索引变化最快。请注意，'C'和'F'选项不考虑基础数组的内存布局，仅引用轴索引的顺序.A'表示如果a为Fortran，则以类似Fortran的索引顺序读取元素在内存中连续，否则类似C的顺序。“ K”表示按照步序在内存中的顺序读取元素，但步幅为负时反转数据除外。默认情况下，使用Cindex顺序。

【例】`flatten()`函数返回的是拷贝。
```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = x.flatten()
print(y)
# [11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34
#  35]

y[3] = 0
print(x)
# [[11 12 13 14 15]
#  [16 17 18 19 20]
#  [21 22 23 24 25]
#  [26 27 28 29 30]
#  [31 32 33 34 35]]

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])

y = x.flatten(order='F')
print(y)
# [11 16 21 26 31 12 17 22 27 32 13 18 23 28 33 14 19 24 29 34 15 20 25 30
#  35]

y[3] = 0
print(x)
# [[11 12 13 14 15]
#  [16 17 18 19 20]
#  [21 22 23 24 25]
#  [26 27 28 29 30]
#  [31 32 33 34 35]]
```



- `numpy.ravel(a, order='C')`Return a contiguous flattened array.

【例】`ravel()`返回的是视图。
```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = np.ravel(x)
print(y)
# [11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34
#  35]

y[3] = 0
print(x)
# [[11 12 13  0 15]
#  [16 17 18 19 20]
#  [21 22 23 24 25]
#  [26 27 28 29 30]
#  [31 32 33 34 35]]

【例】order=F 就是拷贝

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])

y = np.ravel(x, order='F')
print(y)
# [11 16 21 26 31 12 17 22 27 32 13 18 23 28 33 14 19 24 29 34 15 20 25 30
#  35]

y[3] = 0
print(x)
# [[11 12 13 14 15]
#  [16 17 18 19 20]
#  [21 22 23 24 25]
#  [26 27 28 29 30]
#  [31 32 33 34 35]]
```

- `numpy.reshape(a, newshape[, order='C'])`在不更改数据的情况下为数组赋予新的形状。

【例】`reshape()`函数当参数`newshape = [rows,-1]`时，将根据行数自动确定列数。
```python
import numpy as np

x = np.arange(12)
y = np.reshape(x, [3, 4])
print(y.dtype)  # int32
print(y)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

y = np.reshape(x, [3, -1])
print(y)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

y = np.reshape(x,[-1,3])
print(y)
# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 9 10 11]]

y[0, 1] = 10
print(x)
# [ 0 10  2  3  4  5  6  7  8  9 10 11]（改变x去reshape后y中的值，x对应元素也改变）
```

【例】`reshape()`函数当参数`newshape = -1`时，表示将数组降为一维。

```python
import numpy as np

x = np.random.randint(12, size=[2, 2, 3])
print(x)
# [[[11  9  1]
#   [ 1 10  3]]
# 
#  [[ 0  6  1]
#   [ 4 11  3]]]
y = np.reshape(x, -1)
print(y)
# [11  9  1  1 10  3  0  6  1  4 11  3]
```



## 数组转置


- `numpy.transpose(a, axes=None)` Permute the dimensions of an array.
- `numpy.ndarray.T` Same as `self.transpose()`, except that self is returned if `self.ndim < 2`.

【例】
```python
import numpy as np

x = np.random.rand(5, 5) * 10
x = np.around(x, 2)
print(x)
# [[6.74 8.46 6.74 5.45 1.25]
#  [3.54 3.49 8.62 1.94 9.92]
#  [5.03 7.22 1.6  8.7  0.43]
#  [7.5  7.31 5.69 9.67 7.65]
#  [1.8  9.52 2.78 5.87 4.14]]
y = x.T
print(y)
# [[6.74 3.54 5.03 7.5  1.8 ]
#  [8.46 3.49 7.22 7.31 9.52]
#  [6.74 8.62 1.6  5.69 2.78]
#  [5.45 1.94 8.7  9.67 5.87]
#  [1.25 9.92 0.43 7.65 4.14]]
y = np.transpose(x)
print(y)
# [[6.74 3.54 5.03 7.5  1.8 ]
#  [8.46 3.49 7.22 7.31 9.52]
#  [6.74 8.62 1.6  5.69 2.78]
#  [5.45 1.94 8.7  9.67 5.87]
#  [1.25 9.92 0.43 7.65 4.14]]
```


## 更改维度

当创建一个数组之后，还可以给它增加一个维度，这在矩阵计算中经常会用到。

- `numpy.newaxis = None` `None`的别名，对索引数组很有用。

【例】很多工具包在进行计算时都会先判断输入数据的维度是否满足要求，如果输入数据达不到指定的维度时，可以使用`newaxis`参数来增加一个维度。
```python
import numpy as np

x = np.array([1, 2, 9, 4, 5, 6, 7, 8])
print(x.shape)  # (8,)
print(x)  # [1 2 9 4 5 6 7 8]

y = x[np.newaxis, :]
print(y.shape)  # (1, 8)
print(y)  # [[1 2 9 4 5 6 7 8]]

y = x[:, np.newaxis]
print(y.shape)  # (8, 1)
print(y)
# [[1]
#  [2]
#  [9]
#  [4]
#  [5]
#  [6]
#  [7]
#  [8]]
```

- `numpy.squeeze(a, axis=None)` 从数组的形状中删除单维度条目，即把shape中为1的维度去掉。
    - `a`表示输入的数组；
    - `axis`用于指定需要删除的维度，但是指定的维度必须为单维度，否则将会报错；

在机器学习和深度学习中，通常算法的结果是可以表示向量的数组（即包含两对或以上的方括号形式[[]]），如果直接利用这个数组进行画图可能显示界面为空（见后面的示例）。我们可以利用`squeeze()`函数将表示向量的数组转换为秩为1的数组，这样利用 matplotlib 库函数画图时，就可以正常的显示结果了。

【例】
```python
import numpy as np

x = np.arange(10)
print(x.shape)  # (10,)
x = x[np.newaxis, :]
print(x.shape)  # (1, 10)
y = np.squeeze(x)
print(y.shape)  # (10,)
```

【例】
```python
import numpy as np

x = np.array([[[0], [1], [2]]])
print(x.shape)  # (1, 3, 1)
print(x)
# [[[0]
#   [1]
#   [2]]]

y = np.squeeze(x)
print(y.shape)  # (3,)
print(y)  # [0 1 2]

y = np.squeeze(x, axis=0)
print(y.shape)  # (3, 1)
print(y)
# [[0]
#  [1]
#  [2]]

y = np.squeeze(x, axis=2)
print(y.shape)  # (1, 3)
print(y)  # [[0 1 2]]

y = np.squeeze(x, axis=1)
# ValueError: cannot select an axis to squeeze out which has size not equal to one
```

【例】
```python
import numpy as np
import matplotlib.pyplot as plt

x = np.array([[1, 4, 9, 16, 25]])
print(x.shape)  # (1, 5)
plt.plot(x)
plt.show()
```
![](https://img-blog.csdnimg.cn/20200528095957317.png)

【例】
```python
import numpy as np
import matplotlib.pyplot as plt

x = np.array([[1, 4, 9, 16, 25]])
x = np.squeeze(x)
print(x.shape)  # (5, )
plt.plot(x)
plt.show()
```

![](https://img-blog.csdnimg.cn/20200528100221464.png)

## 数组组合

如果要将两份数据组合到一起，就需要拼接操作。

- `numpy.concatenate((a1, a2, ...), axis=0, out=None)` Join a sequence of arrays along an existing axis.

【例】连接沿现有轴的数组序列（原来x，y都是一维的，拼接后的结果也是一维的）。
```python
import numpy as np

x = np.array([1, 2, 3])
y = np.array([7, 8, 9])
z = np.concatenate([x, y])
print(z)
# [1 2 3 7 8 9]

z = np.concatenate([x, y], axis=0)
print(z)
# [1 2 3 7 8 9]
```

【例】原来x，y都是二维的，拼接后的结果也是二维的。
```python
import numpy as np

x = np.array([1, 2, 3]).reshape(1, 3)
y = np.array([7, 8, 9]).reshape(1, 3)
z = np.concatenate([x, y])
print(z)
# [[ 1  2  3]
#  [ 7  8  9]]
z = np.concatenate([x, y], axis=0)
print(z)
# [[ 1  2  3]
#  [ 7  8  9]]
z = np.concatenate([x, y], axis=1)
print(z)
# [[ 1  2  3  7  8  9]]
```

【例】x，y在原来的维度上进行拼接。

```python
import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.array([[7, 8, 9], [10, 11, 12]])
z = np.concatenate([x, y])
print(z)
# [[ 1  2  3]
#  [ 4  5  6]
#  [ 7  8  9]
#  [10 11 12]]
z = np.concatenate([x, y], axis=0)
print(z)
# [[ 1  2  3]
#  [ 4  5  6]
#  [ 7  8  9]
#  [10 11 12]]
z = np.concatenate([x, y], axis=1)
print(z)
# [[ 1  2  3  7  8  9]
#  [ 4  5  6 10 11 12]]
```

- `numpy.stack(arrays, axis=0, out=None)`Join a sequence of arrays along a new axis.


【例】沿着新的轴加入一系列数组（stack为增加维度的拼接）。
```python
import numpy as np

x = np.array([1, 2, 3])
y = np.array([7, 8, 9])
z = np.stack([x, y])
print(z.shape)  # (2, 3)
print(z)
# [[1 2 3]
#  [7 8 9]]

z = np.stack([x, y], axis=1)
print(z.shape)  # (3, 2)
print(z)
# [[1 7]
#  [2 8]
#  [3 9]]
```
【例】
```python
import numpy as np

x = np.array([1, 2, 3]).reshape(1, 3)
y = np.array([7, 8, 9]).reshape(1, 3)
z = np.stack([x, y])
print(z.shape)  # (2, 1, 3)
print(z)
# [[[1 2 3]]
#
#  [[7 8 9]]]

z = np.stack([x, y], axis=1)
print(z.shape)  # (1, 2, 3)
print(z)
# [[[1 2 3]
#   [7 8 9]]]

z = np.stack([x, y], axis=2)
print(z.shape)  # (1, 3, 2)
print(z)
# [[[1 7]
#   [2 8]
#   [3 9]]]
```

【例】
```python
import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.array([[7, 8, 9], [10, 11, 12]])
z = np.stack([x, y])
print(z.shape)  # (2, 2, 3)
print(z)
# [[[ 1  2  3]
#   [ 4  5  6]]
# 
#  [[ 7  8  9]
#   [10 11 12]]]

z = np.stack([x, y], axis=1)
print(z.shape)  # (2, 2, 3)
print(z)
# [[[ 1  2  3]
#   [ 7  8  9]]
# 
#  [[ 4  5  6]
#   [10 11 12]]]

z = np.stack([x, y], axis=2)
print(z.shape)  # (2, 3, 2)
print(z)
# [[[ 1  7]
#   [ 2  8]
#   [ 3  9]]
# 
#  [[ 4 10]
#   [ 5 11]
#   [ 6 12]]]
```


- `numpy.vstack(tup)`Stack arrays in sequence vertically (row wise).
- `numpy.hstack(tup)`Stack arrays in sequence horizontally (column wise). 


【例】一维的情况。
```python
import numpy as np

x = np.array([1, 2, 3])
y = np.array([7, 8, 9])
z = np.vstack((x, y))
print(z.shape)  # (2, 3)
print(z)
# [[1 2 3]
#  [7 8 9]]

z = np.stack([x, y])
print(z.shape)  # (2, 3)
print(z)
# [[1 2 3]
#  [7 8 9]]

z = np.hstack((x, y))
print(z.shape)  # (6,)
print(z)
# [1  2  3  7  8  9]

z = np.concatenate((x, y))
print(z.shape)  # (6,)
print(z)  # [1 2 3 7 8 9]
```



【例】二维的情况。
```python
import numpy as np

x = np.array([1, 2, 3]).reshape(1, 3)
y = np.array([7, 8, 9]).reshape(1, 3)
z = np.vstack((x, y))
print(z.shape)  # (2, 3)
print(z)
# [[1 2 3]
#  [7 8 9]]

z = np.concatenate((x, y), axis=0)
print(z.shape)  # (2, 3)
print(z)
# [[1 2 3]
#  [7 8 9]]

z = np.hstack((x, y))
print(z.shape)  # (1, 6)
print(z)
# [[ 1  2  3  7  8  9]]

z = np.concatenate((x, y), axis=1)
print(z.shape)  # (1, 6)
print(z)
# [[1 2 3 7 8 9]]
```

【例】二维的情况。
```python
import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.array([[7, 8, 9], [10, 11, 12]])
z = np.vstack((x, y))
print(z.shape)  # (4, 3)
print(z)
# [[ 1  2  3]
#  [ 4  5  6]
#  [ 7  8  9]
#  [10 11 12]]

z = np.concatenate((x, y), axis=0)
print(z.shape)  # (4, 3)
print(z)
# [[ 1  2  3]
#  [ 4  5  6]
#  [ 7  8  9]
#  [10 11 12]]

z = np.hstack((x, y))
print(z.shape)  # (2, 6)
print(z)
# [[ 1  2  3  7  8  9]
#  [ 4  5  6 10 11 12]]

z = np.concatenate((x, y), axis=1)
print(z.shape)  # (2, 6)
print(z)
# [[ 1  2  3  7  8  9]
#  [ 4  5  6 10 11 12]]
```

`hstack(),vstack()`分别表示水平和竖直的拼接方式。在数据维度等于1时，比较特殊。而当维度大于或等于2时，它们的作用相当于`concatenate`，用于在已有轴上进行操作。

【例】
```python
import numpy as np

a = np.hstack([np.array([1, 2, 3, 4]), 5])
print(a)  # [1 2 3 4 5]

a = np.concatenate([np.array([1, 2, 3, 4]), 5])
print(a)
# all the input arrays must have same number of dimensions, but the array at index 0 has 1 dimension(s) and the array at index 1 has 0 dimension(s)
```


## 数组拆分

- `numpy.split(ary, indices_or_sections, axis=0)` Split an array into multiple sub-arrays as views into ary.

【例】拆分数组。
```python
import numpy as np

x = np.array([[11, 12, 13, 14],
              [16, 17, 18, 19],
              [21, 22, 23, 24]])
y = np.split(x, [1, 3])
print(y)
# [array([[11, 12, 13, 14]]), array([[16, 17, 18, 19],
#        [21, 22, 23, 24]]), array([], shape=(0, 4), dtype=int32)]

y = np.split(x, [1, 3], axis=1)
print(y)
# [array([[11],
#        [16],
#        [21]]), array([[12, 13],
#        [17, 18],
#        [22, 23]]), array([[14],
#        [19],
#        [24]])]
```



- `numpy.vsplit(ary, indices_or_sections)` Split an array into multiple sub-arrays vertically (row-wise).

【例】垂直切分是把数组按照高度切分
```python
import numpy as np

x = np.array([[11, 12, 13, 14],
              [16, 17, 18, 19],
              [21, 22, 23, 24]])
y = np.vsplit(x, 3)
print(y)
# [array([[11, 12, 13, 14]]), array([[16, 17, 18, 19]]), array([[21, 22, 23, 24]])]

y = np.split(x, 3)
print(y)
# [array([[11, 12, 13, 14]]), array([[16, 17, 18, 19]]), array([[21, 22, 23, 24]])]


y = np.vsplit(x, [1])
print(y)
# [array([[11, 12, 13, 14]]), array([[16, 17, 18, 19],
#        [21, 22, 23, 24]])]

y = np.split(x, [1])
print(y)
# [array([[11, 12, 13, 14]]), array([[16, 17, 18, 19],
#        [21, 22, 23, 24]])]


y = np.vsplit(x, [1, 3])
print(y)
# [array([[11, 12, 13, 14]]), array([[16, 17, 18, 19],
#        [21, 22, 23, 24]]), array([], shape=(0, 4), dtype=int32)]
y = np.split(x, [1, 3], axis=0)
print(y)
# [array([[11, 12, 13, 14]]), array([[16, 17, 18, 19],
#        [21, 22, 23, 24]]), array([], shape=(0, 4), dtype=int32)]
```

- `numpy.hsplit(ary, indices_or_sections)` Split an array into multiple sub-arrays horizontally (column-wise).


【例】水平切分是把数组按照宽度切分。
```python
import numpy as np

x = np.array([[11, 12, 13, 14],
              [16, 17, 18, 19],
              [21, 22, 23, 24]])
y = np.hsplit(x, 2)
print(y)
# [array([[11, 12],
#        [16, 17],
#        [21, 22]]), array([[13, 14],
#        [18, 19],
#        [23, 24]])]

y = np.split(x, 2, axis=1)
print(y)
# [array([[11, 12],
#        [16, 17],
#        [21, 22]]), array([[13, 14],
#        [18, 19],
#        [23, 24]])]

y = np.hsplit(x, [3])
print(y)
# [array([[11, 12, 13],
#        [16, 17, 18],
#        [21, 22, 23]]), array([[14],
#        [19],
#        [24]])]

y = np.split(x, [3], axis=1)
print(y)
# [array([[11, 12, 13],
#        [16, 17, 18],
#        [21, 22, 23]]), array([[14],
#        [19],
#        [24]])]

y = np.hsplit(x, [1, 3])
print(y)
# [array([[11],
#        [16],
#        [21]]), array([[12, 13],
#        [17, 18],
#        [22, 23]]), array([[14],
#        [19],
#        [24]])]

y = np.split(x, [1, 3], axis=1)
print(y)
# [array([[11],
#        [16],
#        [21]]), array([[12, 13],
#        [17, 18],
#        [22, 23]]), array([[14],
#        [19],
#        [24]])]
```





## 数组平铺

- `numpy.tile(A, reps)` Construct an array by repeating A the number of times given by reps.

`tile`是瓷砖的意思，顾名思义，这个函数就是把数组像瓷砖一样铺展开来。

【例】将原矩阵横向、纵向地复制。
```python
import numpy as np

x = np.array([[1, 2], [3, 4]])
print(x)
# [[1 2]
#  [3 4]]

y = np.tile(x, (1, 3))
print(y)
# [[1 2 1 2 1 2]
#  [3 4 3 4 3 4]]

y = np.tile(x, (3, 1))
print(y)
# [[1 2]
#  [3 4]
#  [1 2]
#  [3 4]
#  [1 2]
#  [3 4]]

y = np.tile(x, (3, 3))
print(y)
# [[1 2 1 2 1 2]
#  [3 4 3 4 3 4]
#  [1 2 1 2 1 2]
#  [3 4 3 4 3 4]
#  [1 2 1 2 1 2]
#  [3 4 3 4 3 4]]
```

- `numpy.repeat(a, repeats, axis=None)` Repeat elements of an array.
    - `axis=0`，沿着y轴复制，实际上增加了行数。
    - `axis=1`，沿着x轴复制，实际上增加了列数。
    - `repeats`，可以为一个数，也可以为一个矩阵。
    - `axis=None`时就会flatten当前矩阵，实际上就是变成了一个行向量。

【例】重复数组的元素。
```python
import numpy as np

x = np.repeat(3, 4)
print(x)  # [3 3 3 3]

x = np.array([[1, 2], [3, 4]])
y = np.repeat(x, 2)
print(y)
# [1 1 2 2 3 3 4 4]

y = np.repeat(x, 2, axis=0)
print(y)
# [[1 2]
#  [1 2]
#  [3 4]
#  [3 4]]

y = np.repeat(x, 2, axis=1)
print(y)
# [[1 1 2 2]
#  [3 3 4 4]]

y = np.repeat(x, [2, 3], axis=0)
print(y)
# [[1 2]
#  [1 2]
#  [3 4]
#  [3 4]
#  [3 4]]

y = np.repeat(x, [2, 3], axis=1)
print(y)
# [[1 1 2 2 2]
#  [3 3 4 4 4]]
```

---
## 添加和删除元素

- `numpy.unique(ar, return_index=False, return_inverse=False,return_counts=False, axis=None)` Find the unique elements of an array.
    - return_index：the indices of the input array that give the unique values
    - return_inverse：the indices of the unique array that reconstruct the input array
    - return_counts：the number of times each unique value comes up in the input array




【例】查找数组的唯一元素。
```python
a=np.array([1,1,2,3,3,4,4])
b=np.unique(a,return_counts=True)
print(b[0][list(b[1]).index(1)])
#2
```

---
**参考文献**
- https://blog.csdn.net/csdn15698845876/article/details/73380803


