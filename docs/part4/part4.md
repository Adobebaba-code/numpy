# 数学函数


## 算数运算
### numpy.add
### numpy.subtract
### numpy.multiply
### numpy.divide
### numpy.floor_divide
### numpy.power
- `numpy.add(x1, x2, *args, **kwargs)` Add arguments element-wise.
- `numpy.subtract(x1, x2, *args, **kwargs)` Subtract arguments element-wise.
- `numpy.multiply(x1, x2, *args, **kwargs)` Multiply arguments element-wise.
- `numpy.divide(x1, x2, *args, **kwargs)` Returns a true division of the inputs, element-wise.
- `numpy.floor_divide(x1, x2, *args, **kwargs)` Return the largest integer smaller or equal to the division of the inputs.
- `numpy.power(x1, x2, *args, **kwargs)` First array elements raised to powers from second array, element-wise.

在 numpy 中对以上函数进行了运算符的重载，且运算符为 **元素级**。也就是说，它们只用于位置相同的元素之间，所得到的运算结果组成一个新的数组。

【例】注意 numpy 的广播规则。
```python
import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = x + 1
print(y)
print(np.add(x, 1))
# [2 3 4 5 6 7 8 9]

y = x - 1
print(y)
print(np.subtract(x, 1))
# [0 1 2 3 4 5 6 7]

y = x * 2
print(y)
print(np.multiply(x, 2))
# [ 2  4  6  8 10 12 14 16]

y = x / 2
print(y)
print(np.divide(x, 2))
# [0.5 1.  1.5 2.  2.5 3.  3.5 4. ]

y = x // 2
print(y)
print(np.floor_divide(x, 2))
# [0 1 1 2 2 3 3 4]

y = x ** 2
print(y)
print(np.power(x, 2))
# [ 1  4  9 16 25 36 49 64]
```




【例】注意 numpy 的广播规则。
```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = x + 1
print(y)
print(np.add(x, 1))
# [[12 13 14 15 16]
#  [17 18 19 20 21]
#  [22 23 24 25 26]
#  [27 28 29 30 31]
#  [32 33 34 35 36]]

y = x - 1
print(y)
print(np.subtract(x, 1))
# [[10 11 12 13 14]
#  [15 16 17 18 19]
#  [20 21 22 23 24]
#  [25 26 27 28 29]
#  [30 31 32 33 34]]

y = x * 2
print(y)
print(np.multiply(x, 2))
# [[22 24 26 28 30]
#  [32 34 36 38 40]
#  [42 44 46 48 50]
#  [52 54 56 58 60]
#  [62 64 66 68 70]]

y = x / 2
print(y)
print(np.divide(x, 2))
# [[ 5.5  6.   6.5  7.   7.5]
#  [ 8.   8.5  9.   9.5 10. ]
#  [10.5 11.  11.5 12.  12.5]
#  [13.  13.5 14.  14.5 15. ]
#  [15.5 16.  16.5 17.  17.5]]

y = x // 2
print(y)
print(np.floor_divide(x, 2))
# [[ 5  6  6  7  7]
#  [ 8  8  9  9 10]
#  [10 11 11 12 12]
#  [13 13 14 14 15]
#  [15 16 16 17 17]]

y = x ** 2
print(y)
print(np.power(x, 2))
# [[ 121  144  169  196  225]
#  [ 256  289  324  361  400]
#  [ 441  484  529  576  625]
#  [ 676  729  784  841  900]
#  [ 961 1024 1089 1156 1225]]
```

【例】注意 numpy 的广播规则。
```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])

y = np.arange(1, 6)
print(y)
# [1 2 3 4 5]

z = x + y
print(z)
print(np.add(x, y))
# [[12 14 16 18 20]
#  [17 19 21 23 25]
#  [22 24 26 28 30]
#  [27 29 31 33 35]
#  [32 34 36 38 40]]

z = x - y
print(z)
print(np.subtract(x, y))
# [[10 10 10 10 10]
#  [15 15 15 15 15]
#  [20 20 20 20 20]
#  [25 25 25 25 25]
#  [30 30 30 30 30]]

z = x * y
print(z)
print(np.multiply(x, y))
# [[ 11  24  39  56  75]
#  [ 16  34  54  76 100]
#  [ 21  44  69  96 125]
#  [ 26  54  84 116 150]
#  [ 31  64  99 136 175]]

z = x / y
print(z)
print(np.divide(x, y))
# [[11.          6.          4.33333333  3.5         3.        ]
#  [16.          8.5         6.          4.75        4.        ]
#  [21.         11.          7.66666667  6.          5.        ]
#  [26.         13.5         9.33333333  7.25        6.        ]
#  [31.         16.         11.          8.5         7.        ]]

z = x // y
print(z)
print(np.floor_divide(x, y))
# [[11  6  4  3  3]
#  [16  8  6  4  4]
#  [21 11  7  6  5]
#  [26 13  9  7  6]
#  [31 16 11  8  7]]

z = x ** np.full([1, 5], 2)
print(z)
print(np.power(x, np.full([5, 5], 2)))
# [[ 121  144  169  196  225]
#  [ 256  289  324  361  400]
#  [ 441  484  529  576  625]
#  [ 676  729  784  841  900]
#  [ 961 1024 1089 1156 1225]]
```


【例】
```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])

y = np.arange(1, 26).reshape([5, 5])
print(y)
# [[ 1  2  3  4  5]
#  [ 6  7  8  9 10]
#  [11 12 13 14 15]
#  [16 17 18 19 20]
#  [21 22 23 24 25]]

z = x + y
print(z)
print(np.add(x, y))
# [[12 14 16 18 20]
#  [22 24 26 28 30]
#  [32 34 36 38 40]
#  [42 44 46 48 50]
#  [52 54 56 58 60]]

z = x - y
print(z)
print(np.subtract(x, y))
# [[10 10 10 10 10]
#  [10 10 10 10 10]
#  [10 10 10 10 10]
#  [10 10 10 10 10]
#  [10 10 10 10 10]]

z = x * y
print(z)
print(np.multiply(x, y))
# [[ 11  24  39  56  75]
#  [ 96 119 144 171 200]
#  [231 264 299 336 375]
#  [416 459 504 551 600]
#  [651 704 759 816 875]]

z = x / y
print(z)
print(np.divide(x, y))
# [[11.          6.          4.33333333  3.5         3.        ]
#  [ 2.66666667  2.42857143  2.25        2.11111111  2.        ]
#  [ 1.90909091  1.83333333  1.76923077  1.71428571  1.66666667]
#  [ 1.625       1.58823529  1.55555556  1.52631579  1.5       ]
#  [ 1.47619048  1.45454545  1.43478261  1.41666667  1.4       ]]

z = x // y
print(z)
print(np.floor_divide(x, y))
# [[11  6  4  3  3]
#  [ 2  2  2  2  2]
#  [ 1  1  1  1  1]
#  [ 1  1  1  1  1]
#  [ 1  1  1  1  1]]

z = x ** np.full([5, 5], 2)
print(z)
print(np.power(x, np.full([5, 5], 2)))
# [[ 121  144  169  196  225]
#  [ 256  289  324  361  400]
#  [ 441  484  529  576  625]
#  [ 676  729  784  841  900]
#  [ 961 1024 1089 1156 1225]]
```

### numpy.sqrt
### numpy.square
- `numpy.sqrt(x, *args, **kwargs)` Return the non-negative square-root of an array, element-wise.
- `numpy.square(x, *args, **kwargs)` Return the element-wise square of the input.

【例】
```python
import numpy as np

x = np.arange(1, 5)
print(x)  # [1 2 3 4]

y = np.sqrt(x)
print(y)
# [1.         1.41421356 1.73205081 2.        ]
print(np.power(x, 0.5))
# [1.         1.41421356 1.73205081 2.        ]

y = np.square(x)
print(y)
# [ 1  4  9 16]
print(np.power(x, 2))
# [ 1  4  9 16]
```


---
## 三角函数

### numpy.sin
### numpy.cos
### numpy.tan
### numpy.arcsin
### numpy.arccos
### numpy.arctan

- `numpy.sin(x, *args, **kwargs)` Trigonometric sine, element-wise.
- `numpy.cos(x, *args, **kwargs)` Cosine element-wise.
- `numpy.tan(x, *args, **kwargs)` Compute tangent element-wise.
- `numpy.arcsin(x, *args, **kwargs)` Inverse sine, element-wise.
- `numpy.arccos(x, *args, **kwargs)` Trigonometric inverse cosine, element-wise.
- `numpy.arctan(x, *args, **kwargs)` Trigonometric inverse tangent, element-wise.


**通用函数**（universal function）通常叫作ufunc，它对数组中的各个元素逐一进行操作。这表明，通用函数分别处理输入数组的每个元素，生成的结果组成一个新的输出数组。输出数组的大小跟输入数组相同。

三角函数等很多数学运算符合通用函数的定义，例如，计算平方根的`sqrt()`函数、用来取对数的`log()`函数和求正弦值的`sin()`函数。

【例】
```python
import numpy as np

x = np.linspace(start=0, stop=np.pi / 2, num=10)
print(x)
# [0.         0.17453293 0.34906585 0.52359878 0.6981317  0.87266463
#  1.04719755 1.22173048 1.3962634  1.57079633]

y = np.sin(x)
print(y)
# [0.         0.17364818 0.34202014 0.5        0.64278761 0.76604444
#  0.8660254  0.93969262 0.98480775 1.        ]

z = np.arcsin(y)
print(z)
# [0.         0.17453293 0.34906585 0.52359878 0.6981317  0.87266463
#  1.04719755 1.22173048 1.3962634  1.57079633]

y = np.cos(x)
print(y)
# [1.00000000e+00 9.84807753e-01 9.39692621e-01 8.66025404e-01
#  7.66044443e-01 6.42787610e-01 5.00000000e-01 3.42020143e-01
#  1.73648178e-01 6.12323400e-17]

z = np.arccos(y)
print(z)
# [0.         0.17453293 0.34906585 0.52359878 0.6981317  0.87266463
#  1.04719755 1.22173048 1.3962634  1.57079633]

y = np.tan(x)
print(y)
# [0.00000000e+00 1.76326981e-01 3.63970234e-01 5.77350269e-01
#  8.39099631e-01 1.19175359e+00 1.73205081e+00 2.74747742e+00
#  5.67128182e+00 1.63312394e+16]

z = np.arctan(y)
print(z)
# [0.         0.17453293 0.34906585 0.52359878 0.6981317  0.87266463
#  1.04719755 1.22173048 1.3962634  1.57079633]
```

---
## 指数和对数
### numpy.exp
### numpy.log
### numpy.exp2
### numpy.log2
### numpy.log10

- `numpy.exp(x, *args, **kwargs)` Calculate the exponential of all elements in the input array.
- `numpy.log(x, *args, **kwargs)` Natural logarithm, element-wise.
- `numpy.exp2(x, *args, **kwargs)` Calculate `2**p` for all `p` in the input array.
- `numpy.log2(x, *args, **kwargs)` Base-2 logarithm of `x`.
- `numpy.log10(x, *args, **kwargs)` Return the base 10 logarithm of the input array, element-wise.




【例】The natural logarithm `log` is the inverse of the exponential function, so that `log(exp(x)) = x`. The natural logarithm is logarithm in base `e`.
```python
import numpy as np

x = np.arange(1, 5)
print(x)
# [1 2 3 4]
y = np.exp(x)
print(y)
# [ 2.71828183  7.3890561  20.08553692 54.59815003]
z = np.log(y)
print(z)
# [1. 2. 3. 4.]
```



---
## 加法函数、乘法函数
### numpy.sum

- `numpy.sum(a[, axis=None, dtype=None, out=None, …])` Sum of array elements over a given axis.

通过不同的 `axis`，numpy 会沿着不同的方向进行操作：如果不设置，那么对所有的元素操作；如果`axis=0`，则沿着纵轴进行操作；`axis=1`，则沿着横轴进行操作。但这只是简单的二位数组，如果是多维的呢？可以总结为一句话：设`axis=i`，则 numpy 沿着第`i`个下标变化的方向进行操作。

### numpy.cumsum

- `numpy.cumsum(a, axis=None, dtype=None, out=None)` Return the cumulative sum of the elements along a given axis.

**聚合函数** 是指对一组值（比如一个数组）进行操作，返回一个单一值作为结果的函数。因而，求数组所有元素之和的函数就是聚合函数。`ndarray`类实现了多个这样的函数。

【例】返回给定轴上的数组元素的总和。
```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = np.sum(x)
print(y)  # 575

y = np.sum(x, axis=0)
print(y)  # [105 110 115 120 125]

y = np.sum(x, axis=1)
print(y)  # [ 65  90 115 140 165]
```


【例】返回给定轴上的数组元素的累加和。
```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = np.cumsum(x)
print(y)
# [ 11  23  36  50  65  81  98 116 135 155 176 198 221 245 270 296 323 351
#  380 410 441 473 506 540 575]

y = np.cumsum(x, axis=0)
print(y)
# [[ 11  12  13  14  15]
#  [ 27  29  31  33  35]
#  [ 48  51  54  57  60]
#  [ 74  78  82  86  90]
#  [105 110 115 120 125]]

y = np.cumsum(x, axis=1)
print(y)
# [[ 11  23  36  50  65]
#  [ 16  33  51  70  90]
#  [ 21  43  66  90 115]
#  [ 26  53  81 110 140]
#  [ 31  63  96 130 165]]
```
### numpy.prod 乘积
- `numpy.prod(a[, axis=None, dtype=None, out=None, …])` Return the product of array elements over a given axis.

### numpy.cumprod 累乘
- `numpy.cumprod(a, axis=None, dtype=None, out=None)` Return the cumulative product of elements along a given axis.

【例】返回给定轴上数组元素的乘积。
```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = np.prod(x)
print(y)  # 788529152

y = np.prod(x, axis=0)
print(y)
# [2978976 3877632 4972968 6294624 7875000]

y = np.prod(x, axis=1)
print(y)
# [  360360  1860480  6375600 17100720 38955840]
```


【例】返回给定轴上数组元素的累乘。
```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = np.cumprod(x)
print(y)
# [         11         132        1716       24024      360360     5765760
#     98017920  1764322560  -837609728   427674624   391232512    17180672
#    395155456   893796352   870072320  1147043840   905412608  -418250752
#    755630080  1194065920 -1638662144  -897581056   444596224 -2063597568
#    788529152]

y = np.cumprod(x, axis=0)
print(y)
# [[     11      12      13      14      15]
#  [    176     204     234     266     300]
#  [   3696    4488    5382    6384    7500]
#  [  96096  121176  150696  185136  225000]
#  [2978976 3877632 4972968 6294624 7875000]]

y = np.cumprod(x, axis=1)
print(y)
# [[      11      132     1716    24024   360360]
#  [      16      272     4896    93024  1860480]
#  [      21      462    10626   255024  6375600]
#  [      26      702    19656   570024 17100720]
#  [      31      992    32736  1113024 38955840]]
```
### numpy.diff 差值

- `numpy.diff(a, n=1, axis=-1, prepend=np._NoValue, append=np._NoValue)` Calculate the n-th discrete difference along the given axis.
    - a：输入矩阵
    - n：可选，代表要执行几次差值
    - axis：默认是最后一个


The first difference is given by `out[i] = a[i+1] - a[i]` along the given axis, higher differences are calculated by using `diff` recursively.

【例】沿着指定轴计算第N维的离散差值。

```python
import numpy as np

A = np.arange(2, 14).reshape((3, 4))
A[1, 1] = 8
print(A)
# [[ 2  3  4  5]
#  [ 6  8  8  9]
#  [10 11 12 13]]
print(np.diff(A))
# [[1 1 1]
#  [2 0 1]
#  [1 1 1]]
print(np.diff(A, axis=0))
# [[4 5 4 4]
#  [4 3 4 4]]
```



---
## 四舍五入
### numpy.around 舍入

- `numpy.around(a, decimals=0, out=None)` Evenly round to the given number of decimals.


【例】将数组舍入到给定的小数位数。
```python
import numpy as np

x = np.random.rand(3, 3) * 10
print(x)
# [[6.59144457 3.78566113 8.15321227]
#  [1.68241475 3.78753332 7.68886328]
#  [2.84255822 9.58106727 7.86678037]]

y = np.around(x)
print(y)
# [[ 7.  4.  8.]
#  [ 2.  4.  8.]
#  [ 3. 10.  8.]]

y = np.around(x, decimals=2)
print(y)
# [[6.59 3.79 8.15]
#  [1.68 3.79 7.69]
#  [2.84 9.58 7.87]]
```
### numpy.ceil 上限
### numpy.floor 下限

- `numpy.ceil(x, *args, **kwargs)` Return the ceiling of the input, element-wise.
- `numpy.floor(x, *args, **kwargs)` Return the floor of the input, element-wise.


【例】
```python
import numpy as np

x = np.random.rand(3, 3) * 10
print(x)
# [[0.67847795 1.33073923 4.53920122]
#  [7.55724676 5.88854047 2.65502046]
#  [8.67640444 8.80110812 5.97528726]]

y = np.ceil(x)
print(y)
# [[1. 2. 5.]
#  [8. 6. 3.]
#  [9. 9. 6.]]

y = np.floor(x)
print(y)
# [[0. 1. 4.]
#  [7. 5. 2.]
#  [8. 8. 5.]]
```

---
## 杂项

### numpy.clip 裁剪

- `numpy.clip(a, a_min, a_max, out=None, **kwargs):` Clip (limit) the values in an array.

Given an interval, values outside the interval are clipped to the interval edges.  For example, if an interval of `[0, 1]` is specified, values smaller than 0 become 0, and values larger than 1 become 1.


【例】裁剪（限制）数组中的值。
```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = np.clip(x, a_min=20, a_max=30)
print(y)
# [[20 20 20 20 20]
#  [20 20 20 20 20]
#  [21 22 23 24 25]
#  [26 27 28 29 30]
#  [30 30 30 30 30]]
```
### numpy.absolute 绝对值
### numpy.abs

- `numpy.absolute(x, *args, **kwargs)` Calculate the absolute value element-wise. 
- `numpy.abs(x, *args, **kwargs)` is a shorthand for this function.

【例】
```python
import numpy as np

x = np.arange(-5, 5)
print(x)
# [-5 -4 -3 -2 -1  0  1  2  3  4]

y = np.abs(x)
print(y)
# [5 4 3 2 1 0 1 2 3 4]

y = np.absolute(x)
print(y)
# [5 4 3 2 1 0 1 2 3 4]
```
### numpy.sign 返回数字符号的逐元素指示
- `numpy.sign(x, *args, **kwargs)` Returns an element-wise indication of the sign of a number.

【例】

```python
x = np.arange(-5, 5)
print(x)
#[-5 -4 -3 -2 -1  0  1  2  3  4]
print(np.sign(x))
#[-1 -1 -1 -1 -1  0  1  1  1  1]
```




---
**参考文献**
- https://mp.weixin.qq.com/s/RWsGvvmw4ptf7d8zPIDEJw
- https://blog.csdn.net/hanshuobest/article/details/78558826?utm_medium=distribute.pc_relevant.none-task-blog-baidujs-1


# 逻辑函数

## 真值测试
### numpy.all
### numpy.any

- `numpy.all(a, axis=None, out=None, keepdims=np._NoValue)` Test whether all array elements along a given axis evaluate to True.
- `numpy.any(a, axis=None, out=None, keepdims=np._NoValue)` Test whether any array element along a given axis evaluates to True.




【例】
```python
import numpy as np

a = np.array([0, 4, 5])
b = np.copy(a)
print(np.all(a == b))  # True
print(np.any(a == b))  # True

b[0] = 1
print(np.all(a == b))  # False
print(np.any(a == b))  # True

print(np.all([1.0, np.nan]))  # True
print(np.any([1.0, np.nan]))  # True

a = np.eye(3)
print(np.all(a, axis=0))  # [False False False]
print(np.any(a, axis=0))  # [ True  True  True]
```


---
## 数组内容
### numpy.isnan

- `numpy.isnan(x, *args, **kwargs)` Test element-wise for NaN and return result as a boolean array.


【例】
```python
a=np.array([1,2,np.nan])
print(np.isnan(a))
#[False False  True]

```




---
## 逻辑运算
### numpy.logical_not
### numpy.logical_and
### numpy.logical_or
### numpy.logical_xor


- `numpy.logical_not(x, *args, **kwargs)`Compute the truth value of NOT x element-wise.
- `numpy.logical_and(x1, x2, *args, **kwargs)` Compute the truth value of x1 AND x2 element-wise.
- `numpy.logical_or(x1, x2, *args, **kwargs)`Compute the truth value of x1 OR x2 element-wise.
- `numpy.logical_xor(x1, x2, *args, **kwargs)`Compute the truth value of x1 XOR x2, element-wise.




```python
【例】计算非x元素的真值。

import numpy as np

print(np.logical_not(3))  
# False
print(np.logical_not([True, False, 0, 1]))
# [False  True  True False]

x = np.arange(5)
print(np.logical_not(x < 3))
# [False False False  True  True]

【例】计算x1 AND x2元素的真值。

print(np.logical_and(True, False))  
# False
print(np.logical_and([True, False], [True, False]))
# [ True False]
print(np.logical_and(x > 1, x < 4))
# [False False  True  True False]

【例】逐元素计算x1 OR x2的真值。


print(np.logical_or(True, False))
# True
print(np.logical_or([True, False], [False, False]))
# [ True False]
print(np.logical_or(x < 1, x > 3))
# [ True False False False  True]

【例】计算x1 XOR x2的真值，按元素计算。

print(np.logical_xor(True, False))
# True
print(np.logical_xor([True, True, False, False], [True, False, True, False]))
# [False  True  True False]
print(np.logical_xor(x < 1, x > 3))
# [ True False False False  True]
print(np.logical_xor(0, np.eye(2)))
# [[ True False]
#  [False  True]]
```


## 对照
### numpy.greater
### numpy.greater_equal
### numpy.equal
### numpy.not_equal
### numpy.less
### numpy.less_equal


- `numpy.greater(x1, x2, *args, **kwargs)` Return the truth value of (x1 > x2) element-wise.
- `numpy.greater_equal(x1, x2, *args, **kwargs)` Return the truth value of (x1 >= x2) element-wise.
- `numpy.equal(x1, x2, *args, **kwargs)` Return (x1 == x2) element-wise.
- `numpy.not_equal(x1, x2, *args, **kwargs)` Return (x1 != x2) element-wise.
- `numpy.less(x1, x2, *args, **kwargs)` Return the truth value of (x1 < x2) element-wise.
- `numpy.less_equal(x1, x2, *args, **kwargs)` Return the truth value of (x1 =< x2) element-wise.



【例】numpy对以上对照函数进行了运算符的重载。
```python
import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])

y = x > 2
print(y)
print(np.greater(x, 2))
# [False False  True  True  True  True  True  True]

y = x >= 2
print(y)
print(np.greater_equal(x, 2))
# [False  True  True  True  True  True  True  True]

y = x == 2
print(y)
print(np.equal(x, 2))
# [False  True False False False False False False]

y = x != 2
print(y)
print(np.not_equal(x, 2))
# [ True False  True  True  True  True  True  True]

y = x < 2
print(y)
print(np.less(x, 2))
# [ True False False False False False False False]

y = x <= 2
print(y)
print(np.less_equal(x, 2))
# [ True  True False False False False False False]
```

【例】
```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = x > 20
print(y)
print(np.greater(x, 20))
# [[False False False False False]
#  [False False False False False]
#  [ True  True  True  True  True]
#  [ True  True  True  True  True]
#  [ True  True  True  True  True]]

y = x >= 20
print(y)
print(np.greater_equal(x, 20))
# [[False False False False False]
#  [False False False False  True]
#  [ True  True  True  True  True]
#  [ True  True  True  True  True]
#  [ True  True  True  True  True]]

y = x == 20
print(y)
print(np.equal(x, 20))
# [[False False False False False]
#  [False False False False  True]
#  [False False False False False]
#  [False False False False False]
#  [False False False False False]]

y = x != 20
print(y)
print(np.not_equal(x, 20))
# [[ True  True  True  True  True]
#  [ True  True  True  True False]
#  [ True  True  True  True  True]
#  [ True  True  True  True  True]
#  [ True  True  True  True  True]]


y = x < 20
print(y)
print(np.less(x, 20))
# [[ True  True  True  True  True]
#  [ True  True  True  True False]
#  [False False False False False]
#  [False False False False False]
#  [False False False False False]]

y = x <= 20
print(y)
print(np.less_equal(x, 20))
# [[ True  True  True  True  True]
#  [ True  True  True  True  True]
#  [False False False False False]
#  [False False False False False]
#  [False False False False False]]
```

【例】
```python
import numpy as np

np.random.seed(20200611)
x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])

y = np.random.randint(10, 40, [5, 5])
print(y)
# [[32 28 31 33 37]
#  [23 37 37 30 29]
#  [32 24 10 33 15]
#  [27 17 10 36 16]
#  [25 32 23 39 34]]

z = x > y
print(z)
print(np.greater(x, y))
# [[False False False False False]
#  [False False False False False]
#  [False False  True False  True]
#  [False  True  True False  True]
#  [ True False  True False  True]]

z = x >= y
print(z)
print(np.greater_equal(x, y))
# [[False False False False False]
#  [False False False False False]
#  [False False  True False  True]
#  [False  True  True False  True]
#  [ True  True  True False  True]]

z = x == y
print(z)
print(np.equal(x, y))
# [[False False False False False]
#  [False False False False False]
#  [False False False False False]
#  [False False False False False]
#  [False  True False False False]]

z = x != y
print(z)
print(np.not_equal(x, y))
# [[ True  True  True  True  True]
#  [ True  True  True  True  True]
#  [ True  True  True  True  True]
#  [ True  True  True  True  True]
#  [ True False  True  True  True]]

z = x < y
print(z)
print(np.less(x, y))
# [[ True  True  True  True  True]
#  [ True  True  True  True  True]
#  [ True  True False  True False]
#  [ True False False  True False]
#  [False False False  True False]]

z = x <= y
print(z)
print(np.less_equal(x, y))
# [[ True  True  True  True  True]
#  [ True  True  True  True  True]
#  [ True  True False  True False]
#  [ True False False  True False]
#  [False  True False  True False]]
```

【例】注意 numpy 的广播规则。
```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])

np.random.seed(20200611)
y = np.random.randint(10, 50, 5)

print(y)
# [32 37 30 24 10]

z = x > y
print(z)
print(np.greater(x, y))
# [[False False False False  True]
#  [False False False False  True]
#  [False False False False  True]
#  [False False False  True  True]
#  [False False  True  True  True]]

z = x >= y
print(z)
print(np.greater_equal(x, y))
# [[False False False False  True]
#  [False False False False  True]
#  [False False False  True  True]
#  [False False False  True  True]
#  [False False  True  True  True]]

z = x == y
print(z)
print(np.equal(x, y))
# [[False False False False False]
#  [False False False False False]
#  [False False False  True False]
#  [False False False False False]
#  [False False False False False]]

z = x != y
print(z)
print(np.not_equal(x, y))
# [[ True  True  True  True  True]
#  [ True  True  True  True  True]
#  [ True  True  True False  True]
#  [ True  True  True  True  True]
#  [ True  True  True  True  True]]

z = x < y
print(z)
print(np.less(x, y))
# [[ True  True  True  True False]
#  [ True  True  True  True False]
#  [ True  True  True False False]
#  [ True  True  True False False]
#  [ True  True False False False]]

z = x <= y
print(z)
print(np.less_equal(x, y))
# [[ True  True  True  True False]
#  [ True  True  True  True False]
#  [ True  True  True  True False]
#  [ True  True  True False False]
#  [ True  True False False False]]
```

### numpy.isclose
### numpy.allclose
- `numpy.isclose(a, b, rtol=1.e-5, atol=1.e-8, equal_nan=False)` Returns a boolean array where two arrays are element-wise equal within a tolerance.
- `numpy.allclose(a, b, rtol=1.e-5, atol=1.e-8, equal_nan=False)` Returns True if two arrays are element-wise equal within a tolerance. 

`numpy.allclose()` 等价于 `numpy.all(isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))`。

The tolerance values are positive, typically very small numbers.  The relative difference (`rtol * abs(b)`) and the absolute difference `atol` are added together to compare against the absolute difference between `a` and `b`.

判断是否为True的计算依据：

```python
np.absolute(a - b) <= (atol + rtol * absolute(b))

- atol：float，绝对公差。
- rtol：float，相对公差。
```

NaNs are treated as equal if they are in the same place and if `equal_nan=True`.  Infs are treated as equal if they are in the same place and of the same sign in both arrays.

【例】比较两个数组是否可以认为相等。
```python
import numpy as np

x = np.isclose([1e10, 1e-7], [1.00001e10, 1e-8])
print(x)  # [ True False]

x = np.allclose([1e10, 1e-7], [1.00001e10, 1e-8])
print(x)  # False

x = np.isclose([1e10, 1e-8], [1.00001e10, 1e-9])
print(x)  # [ True  True]

x = np.allclose([1e10, 1e-8], [1.00001e10, 1e-9])
print(x)  # True

x = np.isclose([1e10, 1e-8], [1.0001e10, 1e-9])
print(x)  # [False  True]

x = np.allclose([1e10, 1e-8], [1.0001e10, 1e-9])
print(x)  # False

x = np.isclose([1.0, np.nan], [1.0, np.nan])
print(x)  # [ True False]

x = np.allclose([1.0, np.nan], [1.0, np.nan])
print(x)  # False

x = np.isclose([1.0, np.nan], [1.0, np.nan], equal_nan=True)
print(x)  # [ True  True]

x = np.allclose([1.0, np.nan], [1.0, np.nan], equal_nan=True)
print(x)  # True
```


