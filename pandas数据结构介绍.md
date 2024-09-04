# pandas数据结构简介

我们将快速地概览pandas基本数据结构。数据类型、索引、轴标签和对齐的基本原理适用于所有对象。首先，导入NumPy和pandas到你的命名空间：

```python
In [1]: import numpy as np

In [2]: import pandas as pd
```

基本上，**数据对齐是内在固有的**。除非你明确地打破，否则标签和数据之间的链接不会断开。

我们将简要介绍数据结构，然后分别考虑所有广泛的功能类别和方法。

## Series

`Series`是一个一维的带标签数组，能够存储任何数据类型（整数、字符串、浮点数、Python 对象等）。轴标签统称为**索引**。创建`Series`的基本方法是调用：

```python
s = pd.Series(data, index=index)
```

这里的`data`可以是许多不同的东西：

- 一个 Python 字典
- 一个 ndarray
- 一个标量值（如 5）

`index`是轴标签的列表。根据`data`不同又分为不同的情况：

**data为ndarray**

如果`data`是一个ndarray，`index`必须与`data`长度相同。如果没有传递`index`，将自动创建一个具有值 `[0, ..., len(data) - 1]`的索引。

```python
In [3]: s = pd.Series(np.random.randn(5), index=["a", "b", "c", "d", "e"])

In [4]: s
Out[4]:
a    0.469112
b   -0.282863
c   -1.509059
d   -1.135632
e    1.212112
dtype: float64

In [5]: s.index
Out[5]: Index(['a', 'b', 'c', 'd', 'e'], dtype='object')

In [6]: pd.Series(np.random.randn(5))
Out[6]:
0   -0.173215
1    0.119209
2   -1.044236
3   -0.861849
4   -2.104569
dtype: float64
```

> **备注**
> pandas支持非唯一索引。如果尝试执行不支持重复索引的操作，执行时发生异常。

**data为字典**

可以从字典创建`Series`：

```python
In [7]: d = {"b": 1, "a": 0, "c": 2}

In [8]: pd.Series(d)
Out[8]:
b    1
a    0
c    2
dtype: int64
```

如果传递了索引，字典中与索引标签相对应的值将被设置到数据中。

```python
In [9]: d = {"a": 0.0, "b": 1.0, "c": 2.0}

In [10]: pd.Series(d)
Out[10]:
a    0.0
b    1.0
c    2.0
dtype: float64

In [11]: pd.Series(d, index=["b", "c", "d", "a"])
Out[11]:
b    1.0
c    2.0
d    NaN
a    0.0
dtype: float64
```

> **备注**
> NaN（非数字）是pandas中使用的标准缺失数据标记。

**data为标量**

如果`data`是一个标量值，则必须提供索引。该值将被设置到index长度的所有位置。

```python
In [12]: pd.Series(5.0, index=["a", "b", "c", "d", "e"])
Out[12]:
a    5.0
b    5.0
c    5.0
d    5.0
e    5.0
dtype: float64
```

### Series类似于ndarray

`Series`非常类似于`ndarray`，并且是大多数NumPy函数的有效参数。然而，切片操作也会切片索引。

```python
In [13]: s.iloc[0]
Out[13]: 0.4691122999071863

In [14]: s.iloc[:3]
Out[14]:
a    0.469112
b   -0.282863
c   -1.509059
dtype: float64

In [15]: s[s > s.median()]
Out[15]:
a    0.469112
e    1.212112
dtype: float64

In [16]: s.iloc[[4, 3, 1]]
Out[16]:
e    1.212112
d   -1.135632
b   -0.282863
dtype: float64

In [17]: np.exp(s)
Out[17]:
a    1.598575
b    0.753623
c    0.221118
d    0.321219
e    3.360575
dtype: float64
```

像NumPy数组一样，pandas `Series`有一个单一的 dtype。

```python
In [18]: s.dtype
Out[18]: dtype('float64')
```

这通常是一个NumPy dtype。然而，pandas和第三方库在某些地方扩展了NumPy的类型系统，在这种情况下，dtype将是一个`ExtensionDtype`。pandas中的一些例子是[分类数据](https://pandas.pydata.org/docs/user_guide/categorical.html#categorical "分类数据")和[可空整数数据类型](https://pandas.pydata.org/docs/user_guide/integer_na.html#integer-na "可空整数数据类型")。参见[数据类型](https://pandas.pydata.org/docs/user_guide/basics.html#basics-dtypes "数据类型")。

如果你需要`Series`背后的实际数组，请使用`Series.array`。

```python
In [19]: s.array
Out[19]:
<NumpyExtensionArray>
[ 0.4691122999071863, -0.2828633443286633, -1.5090585031735124,
 -1.1356323710171934,  1.2121120250208506]
Length: 5, dtype: float64
```

在你需要进行一些不带索引的操作时（例如，禁用自动对齐）访问array可能会很有用。

`Series.array`始终是一个`ExtensionArray`。简而言之，ExtensionArray是一个或多个具体数组（如 numpy.ndarray）的瘦包装（thin wrapper）。pandas知道如何使用一个ExtensionArray并将其存储在 `Series`或`DataFrame`的列中。参见[数据类型](https://pandas.pydata.org/docs/user_guide/basics.html#basics-dtypes "数据类型")。

虽然`Series`类似于ndarray，但如果你需要一个真正的ndarray，那么请使用`Series.to_numpy()`。

```python
In [20]: s.to_numpy()
Out[20]: array([ 0.4691, -0.2829, -1.5091, -1.1356,  1.2121])
```

即使`Series`内部是`ExtensionArray`，但`Series.to_numpy()`会返回一个 NumPy ndarray。

### Series类似于字典

`Series`也像一个固定大小的字典，你可以通过索引标签获取和设置值：

```python
In [21]: s["a"]
Out[21]: 0.4691122999071863

In [22]: s["e"] = 12.0

In [23]: s
Out[23]:
a     0.469112
b    -0.282863
c    -1.509059
d    -1.135632
e    12.000000
dtype: float64

In [24]: "e" in s
Out[24]: True

In [25]: "f" in s
Out[25]: False
```

如果访问的标签不在索引中，则会引发异常

```python
In [26]: s["f"]
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
File ~/work/pandas/pandas/pandas/core/indexes/base.py:3805, in Index.get_loc(self, key)
   3804 try:
-> 3805     return self._engine.get_loc(casted_key)
   3806 except KeyError as err:

File index.pyx:167, in pandas._libs.index.IndexEngine.get_loc()

File index.pyx:196, in pandas._libs.index.IndexEngine.get_loc()

File pandas/_libs/hashtable_class_helper.pxi:7081, in pandas._libs.hashtable.PyObjectHashTable.get_item()

File pandas/_libs/hashtable_class_helper.pxi:7089, in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'f'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
Cell In[26], line 1
----> 1 s["f"]

File ~/work/pandas/pandas/pandas/core/series.py:1121, in Series.__getitem__(self, key)
   1118     return self._values[key]
   1120 elif key_is_scalar:
-> 1121     return self._get_value(key)
   1123 # Convert generator to list before going through hashable part
   1124 # (We will iterate through the generator there to check for slices)
   1125 if is_iterator(key):

File ~/work/pandas/pandas/pandas/core/series.py:1237, in Series._get_value(self, label, takeable)
   1234     return self._values[label]
   1236 # Similar to Index.get_value, but we do not fall back to positional
-> 1237 loc = self.index.get_loc(label)
   1239 if is_integer(loc):
   1240     return self._values[loc]

File ~/work/pandas/pandas/pandas/core/indexes/base.py:3812, in Index.get_loc(self, key)
   3807     if isinstance(casted_key, slice) or (
   3808         isinstance(casted_key, abc.Iterable)
   3809         and any(isinstance(x, slice) for x in casted_key)
   3810     ):
   3811         raise InvalidIndexError(key)
-> 3812     raise KeyError(key) from err
   3813 except TypeError:
   3814     # If we have a listlike key, _check_indexing_error will raise
   3815     #  InvalidIndexError. Otherwise we fall through and re-raise
   3816     #  the TypeError.
   3817     self._check_indexing_error(key)

KeyError: 'f'
```

使用`Series.get()`方法，访问缺失的标签将返回None或指定的默认值

```python
In [27]: s.get("f")

In [28]: s.get("f", np.nan)
Out[28]: nan
```

这些标签也可以通过[属性](https://pandas.pydata.org/docs/user_guide/indexing.html#indexing-attribute-access "属性")访问。

### `Series`的向量化操作和标签对齐

当使用原始NumPy数组时，通常不需要逐值循环。pandas的`Series`也是如此。`Series` 也可以传递到大多数使用ndarray的NumPy方法中。

```python
In [29]: s + s
Out[29]:
a     0.938225
b    -0.565727
c    -3.018117
d    -2.271265
e    24.000000
dtype: float64

In [30]: s * 2
Out[30]:
a     0.938225
b    -0.565727
c    -3.018117
d    -2.271265
e    24.000000
dtype: float64

In [31]: np.exp(s)
Out[31]:
a         1.598575
b         0.753623
c         0.221118
d         0.321219
e    162754.791419
dtype: float64
```

`Series`与ndarray的一个关键区别是，`Series`之间的操作会自动根据标签对齐数据。因此，你可以编写计算代码而不必担心`Series`是否具有相同的标签。

```python
In [32]: s.iloc[1:] + s.iloc[:-1]
Out[32]:
a         NaN
b   -0.565727
c   -3.018117
d   -2.271265
e         NaN
dtype: float64
```

未对齐的`Series`之间的操作结果是涉及的索引的并集。如果在其中一个`Series`对应的找不到标签，结果将被标记为缺失`NaN`。能够编写无需进行任何显式数据对齐的代码，为交互式数据分析和研究提供了巨大的自由度和灵活性。pandas数据结构的集成数据对齐功能使pandas区别于大多数用于处理带标签数据的相关工具。

**备注**

通常，我们选择使不同索引对象之间的操作默认产生索引的并集，以避免信息丢失。尽管数据缺失，但保留索引标签通常是重要的信息，因为它是计算的一部分。当然，你可以通过`dropna`函数丢弃带有缺失数据的标签。

### name属性

`Series`还有一个`name`属性：

```python
In [33]: s = pd.Series(np.random.randn(5), name="something")

In [34]: s
Out[34]:
0   -0.494929
1    1.071804
2    0.721555
3   -0.706771
4   -1.039575
Name: something, dtype: float64

In [35]: s.name
Out[35]: 'something'
```

`Series`的 `name` 在许多情况下可以自动分配，特别是，当从 `DataFrame` 中选择单列时，`name` 将被设置为列标签。

你可以使用 `pandas.Series.rename()` 方法重命名一个 `Series`。

```python
In [36]: s2 = s.rename("different")

In [37]: s2.name
Out[37]: 'different'

```

注意 s 和 s2 引用不同的对象。

## DataFrame

`DataFrame` 是一个 2 维的带标签数据结构，列可能具有不同的类型。你可以将其视为电子表格或 SQL 表，或者 `Series` 对象的字典。它通常是最常用的 pandas 对象。像 `Series` 一样，`DataFrame` 接受许多不同类型的输入：

- '一维ndarray、列表、字典或Series'的字典
- 二维ndarray
- [结构化或记录](https://numpy.org/doc/stable/user/basics.rec.html "结构化或记录")的ndarray
- 一个`Series`
- 另一个 `DataFrame`

除了数据，你可以选择性地传递 `index`（行标签）和 `columns`（列标签）参数。如果你传递了`index`和/或`columns`，你自行确保结果 DataFrame 的`index`和/或`columns`。因此，一个包含Series的字典加上一个特定的`index`将丢弃所有不匹配传递的`index`的数据。

如果未传递轴标签，它们将根据输入数据基于常识规则构建。

### 从'Series或字典'的字典创建

结果的 `index` 将是各个 Series 的 `index` 的并集。如果有嵌套字典，嵌套字典将首先转换为 `Series`。如果没有传递`columns`，列标签将是字典键的有序列表。

```python
In [38]: d = {
    "one": pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"]),
    "two": pd.Series([1.0, 2.0, 3.0, 4.0], index=["a", "b", "c", "d"]),
}


In [39]: df = pd.DataFrame(d)

In [40]: df
Out[40]:
   one  two
a  1.0  1.0
b  2.0  2.0
c  3.0  3.0
d  NaN  4.0

In [41]: pd.DataFrame(d, index=["d", "b", "a"])
Out[41]:
   one  two
d  NaN  4.0
b  2.0  2.0
a  1.0  1.0

In [42]: pd.DataFrame(d, index=["d", "b", "a"], columns=["two", "three"])
Out[42]:
   two three
d  4.0   NaN
b  2.0   NaN
a  1.0   NaN
```

行和列标签可以通过分别访问 `index` 和 `columns` 属性来访问：

> **备注**
> 当传递`columns`和字典数据时，传递的`columns`会覆盖字典中的键。

```python
In [43]: df.index
Out[43]: Index(['a', 'b', 'c', 'd'], dtype='object')

In [44]: df.columns
Out[44]: Index(['one', 'two'], dtype='object')
```

### 从'ndarray或列表'的字典创建

所有 ndarray 必须具有相同的长度。如果传递了`index`，它也必须与数组长度相同。如果没有传递`index`，结果的索引将是 `range(n)`，其中 `n` 是数组长度。

```python
In [45]: d = {"one": [1.0, 2.0, 3.0, 4.0], "two": [4.0, 3.0, 2.0, 1.0]}

In [46]: pd.DataFrame(d)
Out[46]:
   one  two
0  1.0  4.0
1  2.0  3.0
2  3.0  2.0
3  4.0  1.0

In [47]: pd.DataFrame(d, index=["a", "b", "c", "d"])
Out[47]:
   one  two
a  1.0  4.0
b  2.0  3.0
c  3.0  2.0
d  4.0  1.0
```

### 从结构化或记录数组创建

这种情况的处理方式与字典数组相同。

```python
In [48]: data = np.zeros((2,), dtype=[("A", "i4"), ("B", "f4"), ("C", "a10")])

In [49]: data[:] = [(1, 2.0, "Hello"), (2, 3.0, "World")]

In [50]: pd.DataFrame(data)
Out[50]:
   A    B         C
0  1  2.0  b'Hello'
1  2  3.0  b'World'

In [51]: pd.DataFrame(data, index=["first", "second"])
Out[51]:
        A    B         C
first   1  2.0  b'Hello'
second  2  3.0  b'World'

In [52]: pd.DataFrame(data, columns=["C", "A", "B"])
Out[52]:
          C  A    B
0  b'Hello'  1  2.0
1  b'World'  2  3.0
```

> **备注**
> DataFrame不完全像2维NumPy ndarray那样工作。

### 从字典列表创建

```python
In [53]: data2 = [{"a": 1, "b": 2}, {"a": 5, "b": 10, "c": 20}]

In [54]: pd.DataFrame(data2)
Out[54]:
   a   b     c
0  1   2   NaN
1  5  10  20.0

In [55]: pd.DataFrame(data2, index=["first", "second"])
Out[55]:
        a   b     c
first   1   2   NaN
second  5  10  20.0

In [56]: pd.DataFrame(data2, columns=["a", "b"])
Out[56]:
   a   b
0  1   2
1  5  10
```

### 从一个字典的元组创建

你可以传递元组字典创建多索引的`DataFrame`.

```python
pd.DataFrame(
    {
        ("a", "b"): {("A", "B"): 1, ("A", "C"): 2},
        ("a", "a"): {("A", "C"): 3, ("A", "B"): 4},
        ("a", "c"): {("A", "B"): 5, ("A", "C"): 6},
        ("b", "a"): {("A", "C"): 7, ("A", "B"): 8},
        ("b", "b"): {("A", "D"): 9, ("A", "B"): 10},
    }
)
In [57]:
Out[57]:
       a              b
       b    a    c    a     b
A B  1.0  4.0  5.0  8.0  10.0
  C  2.0  3.0  6.0  7.0   NaN
  D  NaN  NaN  NaN  NaN   9.0
```

### 从`Series`创建

结果将是一个与输入 `Series` 具有相同索引的 `DataFrame`，并且只有一列，其列名称是原始 `Series` 的名称（仅在未提供其他列名称时）。

```python
In [58]: ser = pd.Series(range(3), index=list("abc"), name="ser")

In [59]: pd.DataFrame(ser)
Out[59]:
   ser
a    0
b    1
c    2
```

### 从namedtuple列表创建

第一个 `namedtuple` 的字段名决定了 `DataFrame` 的列。剩余的 `namedtuple`（或元组）被简单地解包，它们的值被放入 `DataFrame` 的行中。如果任何一个元组比第一个 `namedtuple` 短，则相应行的后面几列将被标记为缺失值。如果任何一个元组比第一个 `namedtuple` 长，则会引发 `ValueError` 。

```python
In [60]: from collections import namedtuple

In [61]: Point = namedtuple("Point", "x y")

In [62]: pd.DataFrame([Point(0, 0), Point(0, 3), (2, 3)])
Out[62]:
   x  y
0  0  0
1  0  3
2  2  3

In [63]: Point3D = namedtuple("Point3D", "x y z")

In [64]: pd.DataFrame([Point3D(0, 0, 0), Point3D(0, 3, 5), Point(2, 3)])
Out[64]:
   x  y    z
0  0  0  0.0
1  0  3  5.0
2  2  3  NaN
```

### 从数据类的列表创建

数据类是 [PEP557](https://www.python.org/dev/peps/pep-0557 "PEP557") 中引入的，可以传递到 DataFrame 构造函数中。传递数据类列表相当于传递字典列表。

请注意，列表中的所有值都应该是数据类，混合类型会导致 `TypeError`。

```python
In [65]: from dataclasses import make_dataclass

In [66]: Point = make_dataclass("Point", [("x", int), ("y", int)])

In [67]: pd.DataFrame([Point(0, 0), Point(0, 3), Point(2, 3)])
Out[67]:
   x  y
0  0  0
1  0  3
2  2  3
```

**缺失数据**

要构建带有缺失数据的 `DataFrame`，我们使用 `np.nan` 来表示缺失值。或者，你可以传递一个 numpy.MaskedArray 作为数据参数到 DataFrame 构造函数，它的掩码条目将被视为缺失。参见[缺失数据](https://pandas.pydata.org/docs/user_guide/missing_data.html#missing-data "缺失数据")。

### 备用构造函数

**DataFrame.from_dict**

`DataFrame.from_dict()` 接受一个字典的字典或数组类序列的字典，并返回一个 DataFrame。它类似于 DataFrame 构造函数，除了 `orient` 参数默认为 `'columns'`，但可以设置为`'index'`以使用字典键作为行标签。

```python
In [68]: pd.DataFrame.from_dict(dict([("A", [1, 2, 3]), ("B", [4, 5, 6])]))
Out[68]:
   A  B
0  1  4
1  2  5
2  3  6
```

如果你传递 `orient='index'`，键将是行标签。在这种情况下，你还可以传递所需的列名：

```python
pd.DataFrame.from_dict(
    dict([("A", [1, 2, 3]), ("B", [4, 5, 6])]),
    orient="index",
    columns=["one", "two", "three"],
)
In [69]:
Out[69]:
   one  two  three
A    1    2      3
B    4    5      6
```

**DataFrame.from_records**

`DataFrame.from_records()` 接受一个元组列表或具有结构化 dtype 的 ndarray。它类似于普通的 `DataFrame` 构造函数，除了结果 `DataFrame` 的索引可能是结构化 dtype 的特定字段。

```python
In [70]: data
Out[70]:
array([(1, 2., b'Hello'), (2, 3., b'World')],
      dtype=[('A', '<i4'), ('B', '<f4'), ('C', 'S10')])

In [71]: pd.DataFrame.from_records(data, index="C")
Out[71]:
          A    B
C
b'Hello'  1  2.0
b'World'  2  3.0
```

### 列的选择、添加、删除

你可以将 DataFrame 看作'具有类似索引的Series对象'的字典。获取、设置和删除与字典的操作相同：

```python
In [72]: df["one"]
Out[72]:
a    1.0
b    2.0
c    3.0
d    NaN
Name: one, dtype: float64

In [73]: df["three"] = df["one"] * df["two"]

In [74]: df["flag"] = df["one"] > 2

In [75]: df
Out[75]:
   one  two  three   flag
a  1.0  1.0    1.0  False
b  2.0  2.0    4.0  False
c  3.0  3.0    9.0   True
d  NaN  4.0    NaN  False
```

可以像字典一样删除或pop列：

```python
In [76]: del df["two"]

In [77]: three = df.pop("three")

In [78]: df
Out[78]:
   one   flag
a  1.0  False
b  2.0  False
c  3.0   True
d  NaN  False
```

当插入一个标量值时，它将填充到整列：

```python
In [79]: df["foo"] = "bar"

In [80]: df
Out[80]:
   one   flag  foo
a  1.0  False  bar
b  2.0  False  bar
c  3.0   True  bar
d  NaN  False  bar
```

当插入一个与 `DataFrame` 索引不同的 `Series` 时，它将被同型到 `DataFrame` 的索引：

```python
In [81]: df["one_trunc"] = df["one"][:2]

In [82]: df
Out[82]:
   one   flag  foo  one_trunc
a  1.0  False  bar        1.0
b  2.0  False  bar        2.0
c  3.0   True  bar        NaN
d  NaN  False  bar        NaN
```

你可以插入原始 ndarray，但它们的长度必须与 `DataFrame` 索引的长度匹配。

默认情况下，列会插入到末尾。`DataFrame.insert()` 在列中的特定位置插入：

```python
In [83]: df.insert(1, "bar", df["one"])

In [84]: df
Out[84]:
   one  bar   flag  foo  one_trunc
a  1.0  1.0  False  bar        1.0
b  2.0  2.0  False  bar        2.0
c  3.0  3.0   True  bar        NaN
d  NaN  NaN  False  bar        NaN
```

### 在方法链中赋值新列

受 [dplyr](https://dplyr.tidyverse.org/reference/mutate.html "dplyr") 的 `mutate` 动词启发，`DataFrame` 有一个 `assign()` 方法，允许你轻松创建源自现有列的新列。

```python
In [85]: iris = pd.read_csv("data/iris.data")

In [86]: iris.head()
Out[86]:
   SepalLength  SepalWidth  PetalLength  PetalWidth         Name
0          5.1         3.5          1.4         0.2  Iris-setosa
1          4.9         3.0          1.4         0.2  Iris-setosa
2          4.7         3.2          1.3         0.2  Iris-setosa
3          4.6         3.1          1.5         0.2  Iris-setosa
4          5.0         3.6          1.4         0.2  Iris-setosa

In [87]: iris.assign(sepal_ratio=iris["SepalWidth"] / iris["SepalLength"]).head()
Out[87]:
   SepalLength  SepalWidth  PetalLength  PetalWidth         Name  sepal_ratio
0          5.1         3.5          1.4         0.2  Iris-setosa     0.686275
1          4.9         3.0          1.4         0.2  Iris-setosa     0.612245
2          4.7         3.2          1.3         0.2  Iris-setosa     0.680851
3          4.6         3.1          1.5         0.2  Iris-setosa     0.673913
4          5.0         3.6          1.4         0.2  Iris-setosa     0.720000
```

在上面的例子中，我们插入了一个预计算的值。我们也可以传递一个单参数函数。参数是被赋值的DataFrame。

```python
In [88]: iris.assign(sepal_ratio=lambda x: (x["SepalWidth"] / x["SepalLength"])).head()
Out[88]:
   SepalLength  SepalWidth  PetalLength  PetalWidth         Name  sepal_ratio
0          5.1         3.5          1.4         0.2  Iris-setosa     0.686275
1          4.9         3.0          1.4         0.2  Iris-setosa     0.612245
2          4.7         3.2          1.3         0.2  Iris-setosa     0.680851
3          4.6         3.1          1.5         0.2  Iris-setosa     0.673913
4          5.0         3.6          1.4         0.2  Iris-setosa     0.720000
```

`assign()` 总是返回原始DataFrame的副本，不修改原始DataFrame。

当你手头没有 DataFrame 引用时，传递一个可调用对象，而不是要插入的实际值，很有用。这在使用 assign() 进行一系列操作时很常见。例如，我们可以限制 DataFrame 仅包含那些萼片长度大于 5 的观测值，计算比率，并绘制：
Passing a callable, as opposed to an actual value to be inserted, is useful when you don’t have a reference to the DataFrame at hand. This is common when using assign() in a chain of operations. For example, we can limit the DataFrame to just those observations with a Sepal Length greater than 5, calculate the ratio, and plot:

```python
(
    iris.query("SepalLength > 5")
    .assign(
        SepalRatio=lambda x: x.SepalWidth / x.SepalLength,
        PetalRatio=lambda x: x.PetalWidth / x.PetalLength,
    )
    .plot(kind="scatter", x="SepalRatio", y="PetalRatio")
)
In [89]:
Out[89]: <Axes: xlabel='SepalRatio', ylabel='PetalRatio'>
```

![](https://pandas.pydata.org/docs/_images/basics_assign.png)

由于传递了一个函数，该函数将在被赋值的 DataFrame 上计算。重要的是，这是首先被过滤为萼片长度大于 5 的那些行的 DataFrame，然后进行比率计算。这是我们没有 被过滤的 DataFrame 引用的一个例子。

assign() 的函数签名很简单，就是 \*\*kwargs。键是新列的列名，值要么是要插入的值（例如，一个 Series 或 NumPy 数组），要么是要在 DataFrame 上调用的单参数函数。返回插入了新列的原始 DataFrame 的副本。

\*\*kwargs的顺序将被保留。这允许依赖赋值， 同一个assign()中，\*\*kwargs中后面的表达式可使用前面的表达式创建的列。

```python
In [90]: dfa = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

In [91]: dfa.assign(C=lambda x: x["A"] + x["B"], D=lambda x: x["A"] + x["C"])
Out[91]:
   A  B  C   D
0  1  4  5   6
1  2  5  7   9
2  3  6  9  12
```

在第二个表达式中，x['C'] 将引用新创建的列，等于 dfa['A'] + dfa['B']。

### 索引(Indexing)/选择(selection)

索引的基础知识如下：

|操作|语法|结果|
|---|---|---|
|选择列|`df[col]`|Series|
|按标签选择行|`df.loc[label]`|Series|
|按整数位置选择行|`df.iloc[loc]`]|Series|
|行切片|`df[5:10]`|DataFrame|
|按布尔向量选择行|`df[bool_vec]`|DataFrame|

例如，选择行返回一个 Series，其索引是 DataFrame 的列：

```python
In [92]: df.loc["b"]
Out[92]:
one            2.0
bar            2.0
flag         False
foo            bar
one_trunc      2.0
Name: b, dtype: object

In [93]: df.iloc[2]
Out[93]:
one           3.0
bar           3.0
flag         True
foo           bar
one_trunc     NaN
Name: c, dtype: object
```

对于更详尽的基于标签的索引和切片的高级处理，参见[索引](https://pandas.pydata.org/docs/user_guide/indexing.html#indexing "索引")部分。我们将在[重索引](https://pandas.pydata.org/docs/user_guide/basics.html#basics-reindexing "重索引")部分讨论重索引/符合新标签集的基础知识。

### 数据对齐和计算

DataFrame 之间的数据对齐自动在 列和索引（行标签） 上对齐。同样，结果对象将是列和行标签的并集。

```python
In [94]: df = pd.DataFrame(np.random.randn(10, 4), columns=["A", "B", "C", "D"])

In [95]: df2 = pd.DataFrame(np.random.randn(7, 3), columns=["A", "B", "C"])

In [96]: df + df2
Out[96]:
          A         B         C   D
0  0.045691 -0.014138  1.380871 NaN
1 -0.955398 -1.501007  0.037181 NaN
2 -0.662690  1.534833 -0.859691 NaN
3 -2.452949  1.237274 -0.133712 NaN
4  1.414490  1.951676 -2.320422 NaN
5 -0.494922 -1.649727 -1.084601 NaN
6 -1.047551 -0.748572 -0.805479 NaN
7       NaN       NaN       NaN NaN
8       NaN       NaN       NaN NaN
9       NaN       NaN       NaN NaN
```

当执行 DataFrame 和 Series 之间的操作时，默认行为是将 Series 索引 对齐到 DataFrame 列上。例如：

```python
In [97]: df - df.iloc[0]
Out[97]:
          A         B         C         D
0  0.000000  0.000000  0.000000  0.000000
1 -1.359261 -0.248717 -0.453372 -1.754659
2  0.253128  0.829678  0.010026 -1.991234
3 -1.311128  0.054325 -1.724913 -1.620544
4  0.573025  1.500742 -0.676070  1.367331
5 -1.741248  0.781993 -1.241620 -2.053136
6 -1.240774 -0.869551 -0.153282  0.000430
7 -0.743894  0.411013 -0.929563 -0.282386
8 -1.194921  1.320690  0.238224 -1.482644
9  2.293786  1.856228  0.773289 -1.446531
```

对于显式的匹配和广播行为控制，参见[灵活的二元操作](https://pandas.pydata.org/docs/user_guide/basics.html#basics-binop "灵活的二元操作")部分。

标量与算术运算按元素进行：

```python
In [98]: df * 5 + 2
Out[98]:
           A         B         C          D
0   3.359299 -0.124862  4.835102   3.381160
1  -3.437003 -1.368449  2.568242  -5.392133
2   4.624938  4.023526  4.885230  -6.575010
3  -3.196342  0.146766 -3.789461  -4.721559
4   6.224426  7.378849  1.454750  10.217815
5  -5.346940  3.785103 -1.373001  -6.884519
6  -2.844569 -4.472618  4.068691   3.383309
7  -0.360173  1.930201  0.187285   1.969232
8  -2.615303  6.478587  6.026220  -4.032059
9  14.828230  9.156280  8.701544  -3.851494

In [99]: 1 / df
Out[99]:
          A          B         C           D
0  3.678365  -2.353094  1.763605    3.620145
1 -0.919624  -1.484363  8.799067   -0.676395
2  1.904807   2.470934  1.732964   -0.583090
3 -0.962215  -2.697986 -0.863638   -0.743875
4  1.183593   0.929567 -9.170108    0.608434
5 -0.680555   2.800959 -1.482360   -0.562777
6 -1.032084  -0.772485  2.416988    3.614523
7 -2.118489 -71.634509 -2.758294 -162.507295
8 -1.083352   1.116424  1.241860   -0.828904
9  0.389765   0.698687  0.746097   -0.854483

In [100]: df ** 4
Out[100]:
           A             B         C             D
0   0.005462  3.261689e-02  0.103370  5.822320e-03
1   1.398165  2.059869e-01  0.000167  4.777482e+00
2   0.075962  2.682596e-02  0.110877  8.650845e+00
3   1.166571  1.887302e-02  1.797515  3.265879e+00
4   0.509555  1.339298e+00  0.000141  7.297019e+00
5   4.661717  1.624699e-02  0.207103  9.969092e+00
6   0.881334  2.808277e+00  0.029302  5.858632e-03
7   0.049647  3.797614e-08  0.017276  1.433866e-09
8   0.725974  6.437005e-01  0.420446  2.118275e+00
9  43.329821  4.196326e+00  3.227153  1.875802e+00
```

布尔运算符也按元素进行：

```python
In [101]: df1 = pd.DataFrame({"a": [1, 0, 1], "b": [0, 1, 1]}, dtype=bool)

In [102]: df2 = pd.DataFrame({"a": [0, 1, 1], "b": [1, 1, 0]}, dtype=bool)

In [103]: df1 & df2
Out[103]:
       a      b
0  False  False
1  False   True
2   True  False

In [104]: df1 | df2
Out[104]:
      a     b
0  True  True
1  True  True
2  True  True

In [105]: df1 ^ df2
Out[105]:
       a      b
0   True   True
1   True  False
2  False   True

In [106]: -df1
Out[106]:
       a      b
0  False   True
1   True  False
2  False  False
```

### 转置

要转置，访问 T 属性或 DataFrame.transpose()，类似于 ndarray：

```python
# only show the first 5 rows
In [107]: df[:5].T
Out[107]:
          0         1         2         3         4
A  0.271860 -1.087401  0.524988 -1.039268  0.844885
B -0.424972 -0.673690  0.404705 -0.370647  1.075770
C  0.567020  0.113648  0.577046 -1.157892 -0.109050
D  0.276232 -1.478427 -1.715002 -1.344312  1.643563
```

### 用NumPy函数操作DataFrame

大多数 NumPy 函数可以直接在 Series 和 DataFrame 上调用。

```python
In [108]: np.exp(df)
Out[108]:
           A         B         C         D
0   1.312403  0.653788  1.763006  1.318154
1   0.337092  0.509824  1.120358  0.227996
2   1.690438  1.498861  1.780770  0.179963
3   0.353713  0.690288  0.314148  0.260719
4   2.327710  2.932249  0.896686  5.173571
5   0.230066  1.429065  0.509360  0.169161
6   0.379495  0.274028  1.512461  1.318720
7   0.623732  0.986137  0.695904  0.993865
8   0.397301  2.449092  2.237242  0.299269
9  13.009059  4.183951  3.820223  0.310274

In [109]: np.asarray(df)
Out[109]:
array([[ 0.2719, -0.425 ,  0.567 ,  0.2762],
       [-1.0874, -0.6737,  0.1136, -1.4784],
       [ 0.525 ,  0.4047,  0.577 , -1.715 ],
       [-1.0393, -0.3706, -1.1579, -1.3443],
       [ 0.8449,  1.0758, -0.109 ,  1.6436],
       [-1.4694,  0.357 , -0.6746, -1.7769],
       [-0.9689, -1.2945,  0.4137,  0.2767],
       [-0.472 , -0.014 , -0.3625, -0.0062],
       [-0.9231,  0.8957,  0.8052, -1.2064],
       [ 2.5656,  1.4313,  1.3403, -1.1703]])
```

DataFrame不能完全替代ndarray，因为其索引语义和数据模型在某些地方与n维数组有很大不同。

Series 实现了 `__array_ufunc__`，这使得它可以与 NumPy 的[通用函数](https://numpy.org/doc/stable/reference/ufuncs.html "通用函数")一起工作。

ufunc 应用于 Series 内部的数组。

```python
In [110]: ser = pd.Series([1, 2, 3, 4])

In [111]: np.exp(ser)
Out[111]:
0     2.718282
1     7.389056
2    20.085537
3    54.598150
dtype: float64
```

当多个 Series 被传递给 ufunc 时，它们会在执行操作之前进行对齐。

像库的其他部分一样，pandas 将自动对齐标签。例如，在两个标签顺序不同的 Series 上调用 numpy.remainder()， 将先对齐然后执行操作。

```python
In [112]: ser1 = pd.Series([1, 2, 3], index=["a", "b", "c"])

In [113]: ser2 = pd.Series([1, 3, 5], index=["b", "a", "c"])

In [114]: ser1
Out[114]:
a    1
b    2
c    3
dtype: int64

In [115]: ser2
Out[115]:
b    1
a    3
c    5
dtype: int64

In [116]: np.remainder(ser1, ser2)
Out[116]:
a    1
b    0
c    3
dtype: int64
```

像往常一样，将使用两个索引的并集，并且不重叠的值将用缺失值填充。

```python
In [117]: ser3 = pd.Series([2, 4, 6], index=["b", "c", "d"])

In [118]: ser3
Out[118]:
b    2
c    4
d    6
dtype: int64

In [119]: np.remainder(ser1, ser3)
Out[119]:
a    NaN
b    0.0
c    3.0
d    NaN
dtype: float64
```

当二元 ufunc 应用于 Series 和 Index 时，使用Series的索引，并返回一个 Series。

```python
In [120]: ser = pd.Series([1, 2, 3])

In [121]: idx = pd.Index([4, 5, 6])

In [122]: np.maximum(ser, idx)
Out[122]:
0    4
1    5
2    6
dtype: int64
```

NumPy ufuncs 可以安全地应用于内部非ndarray数组的Series，例如 `arrays.SparseArray`（见[稀疏计算](https://pandas.pydata.org/docs/user_guide/sparse.html#sparse-calculation "稀疏计算")）。如果可能，ufunc 将在不将底层数据转换为 ndarray 的情况下应用。

### 控制台显示

一个非常大的 DataFrame 将在控制台中被截断以显示。你也可以使用 info() 获取摘要。 (baseball数据集来自 plyr R 包):

```python
In [123]: baseball = pd.read_csv("data/baseball.csv")

In [124]: print(baseball)
       id     player  year  stint team  lg  ...    so  ibb  hbp   sh   sf  gidp
0   88641  womacto01  2006      2  CHN  NL  ...   4.0  0.0  0.0  3.0  0.0   0.0
1   88643  schilcu01  2006      1  BOS  AL  ...   1.0  0.0  0.0  0.0  0.0   0.0
..    ...        ...   ...    ...  ...  ..  ...   ...  ...  ...  ...  ...   ...
98  89533   aloumo01  2007      1  NYN  NL  ...  30.0  5.0  2.0  0.0  3.0  13.0
99  89534  alomasa02  2007      1  NYN  NL  ...   3.0  0.0  0.0  0.0  0.0   0.0

[100 rows x 23 columns]

In [125]: baseball.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 100 entries, 0 to 99
Data columns (total 23 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   id      100 non-null    int64
 1   player  100 non-null    object
 2   year    100 non-null    int64
 3   stint   100 non-null    int64
 4   team    100 non-null    object
 5   lg      100 non-null    object
 6   g       100 non-null    int64
 7   ab      100 non-null    int64
 8   r       100 non-null    int64
 9   h       100 non-null    int64
 10  X2b     100 non-null    int64
 11  X3b     100 non-null    int64
 12  hr      100 non-null    int64
 13  rbi     100 non-null    float64
 14  sb      100 non-null    float64
 15  cs      100 non-null    float64
 16  bb      100 non-null    int64
 17  so      100 non-null    float64
 18  ibb     100 non-null    float64
 19  hbp     100 non-null    float64
 20  sh      100 non-null    float64
 21  sf      100 non-null    float64
 22  gidp    100 non-null    float64
dtypes: float64(9), int64(11), object(3)
memory usage: 18.1+ KB
```

然而，使用 DataFrame.to_string() 将以表格形式返回 DataFrame 的字符串表示，虽然它可能不适合控制台宽度：

```python
In [126]: print(baseball.iloc[-20:, :12].to_string())
       id     player  year  stint team  lg    g   ab   r    h  X2b  X3b
80  89474  finlest01  2007      1  COL  NL   43   94   9   17    3    0
81  89480  embreal01  2007      1  OAK  AL    4    0   0    0    0    0
82  89481  edmonji01  2007      1  SLN  NL  117  365  39   92   15    2
83  89482  easleda01  2007      1  NYN  NL   76  193  24   54    6    0
84  89489  delgaca01  2007      1  NYN  NL  139  538  71  139   30    0
85  89493  cormirh01  2007      1  CIN  NL    6    0   0    0    0    0
86  89494  coninje01  2007      2  NYN  NL   21   41   2    8    2    0
87  89495  coninje01  2007      1  CIN  NL   80  215  23   57   11    1
88  89497  clemero02  2007      1  NYA  AL    2    2   0    1    0    0
89  89498  claytro01  2007      2  BOS  AL    8    6   1    0    0    0
90  89499  claytro01  2007      1  TOR  AL   69  189  23   48   14    0
91  89501  cirilje01  2007      2  ARI  NL   28   40   6    8    4    0
92  89502  cirilje01  2007      1  MIN  AL   50  153  18   40    9    2
93  89521  bondsba01  2007      1  SFN  NL  126  340  75   94   14    0
94  89523  biggicr01  2007      1  HOU  NL  141  517  68  130   31    3
95  89525  benitar01  2007      2  FLO  NL   34    0   0    0    0    0
96  89526  benitar01  2007      1  SFN  NL   19    0   0    0    0    0
97  89530  ausmubr01  2007      1  HOU  NL  117  349  38   82   16    3
98  89533   aloumo01  2007      1  NYN  NL   87  328  51  112   19    1
99  89534  alomasa02  2007      1  NYN  NL    8   22   1    3    1    0
```

宽 DataFrame 将默认打印在多行中：

```python
In [127]: pd.DataFrame(np.random.randn(3, 12))
Out[127]:
         0         1         2   ...        9         10        11
0 -1.226825  0.769804 -1.281247  ... -1.110336 -0.619976  0.149748
1 -0.732339  0.687738  0.176444  ...  1.462696 -1.743161 -0.826591
2 -0.345352  1.314232  0.690579  ...  0.896171 -0.487602 -0.082240

[3 rows x 12 columns]
```

你可以通过设置 display.width 选项来更改单行上打印的内容量：

```python
In [128]: pd.set_option("display.width", 40)  # default is 80

In [129]: pd.DataFrame(np.random.randn(3, 12))
Out[129]:
         0         1         2   ...        9         10        11
0 -2.182937  0.380396  0.084844  ... -0.023688  2.410179  1.450520
1  0.206053 -0.251905 -2.213588  ... -0.025747 -0.988387  0.094055
2  1.262731  1.289997  0.082423  ... -0.281461  0.030711  0.109121

[3 rows x 12 columns]
```

你可以通过设置 `display.max_colwidth`调整最大列宽：

```python
In [130]: datafile = {
    "filename": ["filename_01", "filename_02"],
    "path": [
        "media/user_name/storage/folder_01/filename_01",
        "media/user_name/storage/folder_02/filename_02",
    ],
}


In [131]: pd.set_option("display.max_colwidth", 30)

In [132]: pd.DataFrame(datafile)
Out[132]:
      filename                           path
0  filename_01  media/user_name/storage/fo...
1  filename_02  media/user_name/storage/fo...

In [133]: pd.set_option("display.max_colwidth", 100)

In [134]: pd.DataFrame(datafile)
Out[134]:
      filename                                           path
0  filename_01  media/user_name/storage/folder_01/filename_01
1  filename_02  media/user_name/storage/folder_02/filename_02
```

你也可以通过 expand_frame_repr 选项禁用此功能。

### DataFrame 列属性访问和 IPython 补全

如果 DataFrame 列标签是有效的 Python 变量名，可以像属性一样访问列：

```python
In [135]: df = pd.DataFrame({"foo1": np.random.randn(5), "foo2": np.random.randn(5)})

In [136]: df
Out[136]:
       foo1      foo2
0  1.126203  0.781836
1 -0.977349 -1.071357
2  1.474071  0.441153
3 -0.064034  2.353925
4 -1.282782  0.583787

In [137]: df.foo1
Out[137]:
0    1.126203
1   -0.977349
2    1.474071
3   -0.064034
4   -1.282782
Name: foo1, dtype: float64
```

列也连接到 IPython 补全机制，因此可以通过 Tab 补全：

```python
df.foo<TAB>  # noqa: E225, E999
df.foo1  df.foo2
```
