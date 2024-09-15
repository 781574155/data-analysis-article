# pandas基础功能

在这里，我们将讨论许多 pandas 数据结构共有的基本功能。首先，让我们像在[10分钟了解pandas](https://blog.openai36.com/2024/09/03/%e5%8d%81%e5%88%86%e9%92%9f%e4%ba%86%e8%a7%a3pandas/ "10分钟了解pandas")中那样创建一些示例对象：

```python
In [1]: index = pd.date_range("1/1/2000", periods=8)

In [2]: s = pd.Series(np.random.randn(5), index=["a", "b", "c", "d", "e"])

In [3]: df = pd.DataFrame(np.random.randn(8, 3), index=index, columns=["A", "B", "C"])
```

## 头部和尾部

可以使用 `head()` 和 `tail()` 方法查看 Series 或 DataFrame 对象的部分数据。默认显示五个元素，你可以传递一个自定义的数量。

```python
In [4]: long_series = pd.Series(np.random.randn(1000))

In [5]: long_series.head()
Out[5]:
0   -1.157892
1   -1.344312
2    0.844885
3    1.075770
4   -0.109050
dtype: float64

In [6]: long_series.tail(3)
Out[6]:
997   -0.289388
998   -1.020544
999    0.589993
dtype: float64
```

## 属性和底层数据

pandas 对象具有许多属性，使你能够访问元数据：

- shape：对象的轴尺寸，与 ndarray 一致
- 轴标签
	- Series：index
	- DataFrame：index 和 columns

备注，**可以对这些属性进行赋值**

```python
In [7]: df[:2]
Out[7]:
                   A         B         C
2000-01-01 -0.173215  0.119209 -1.044236
2000-01-02 -0.861849 -2.104569 -0.494929

In [8]: df.columns = [x.lower() for x in df.columns]

In [9]: df
Out[9]:
                   a         b         c
2000-01-01 -0.173215  0.119209 -1.044236
2000-01-02 -0.861849 -2.104569 -0.494929
2000-01-03  1.071804  0.721555 -0.706771
2000-01-04 -1.039575  0.271860 -0.424972
2000-01-05  0.567020  0.276232 -1.087401
2000-01-06 -0.673690  0.113648 -1.478427
2000-01-07  0.524988  0.404705  0.577046
2000-01-08 -1.715002 -1.039268 -0.370647
```

可以把pandas 对象 (Index, Series, DataFrame) 看作是数组的容器, 它们存储实际数据并执行实际计算. 对于许多类型，底层数组是 numpy.ndarray。然而，pandas 和第三方库可能会扩展NumPy的类型系统以支持自定义数组（参阅[dtypes](https://pandas.pydata.org/docs/user_guide/basics.html#basics-dtypes "dtypes")）。

要获取 Index 或 Series 内部的实际数据，使用 `.array` 属性

```python
In [10]: s.array
Out[10]:
<NumpyExtensionArray>
[ 0.4691122999071863, -0.2828633443286633, -1.5090585031735124,
 -1.1356323710171934,  1.2121120250208506]
Length: 5, dtype: float64

In [11]: s.index.array
Out[11]:
<NumpyExtensionArray>
['a', 'b', 'c', 'd', 'e']
Length: 5, dtype: object
```

array 总是一个 ExtensionArray。
ExtensionArray 的确切细节以及为什么 pandas 使用它们超出本文范围。参见 [dtypes](https://pandas.pydata.org/docs/user_guide/basics.html#basics-dtypes "dtypes")。

如果你需要一个 NumPy 数组，使用 `to_numpy()` 或 `numpy.asarray()`。

```python
In [12]: s.to_numpy()
Out[12]: array([ 0.4691, -0.2829, -1.5091, -1.1356,  1.2121])

In [13]: np.asarray(s)
Out[13]: array([ 0.4691, -0.2829, -1.5091, -1.1356,  1.2121])
```

当 Series 或 Index 内部是 `ExtensionArray` 时，to_numpy() 可能涉及复制数据和强制转换值。参见 [dtypes](https://pandas.pydata.org/docs/user_guide/basics.html#basics-dtypes "dtypes")。

to_numpy() 允许你对生成的 numpy.ndarray 的 dtype 进行一定控制。例如，考虑带有时区的日期时间。NumPy 没有一个 dtype 来表示带时区的日期时间，所以有两种可能有用的表示方法：

1. 一个 dtype 为 object 的 numpy.ndarray，具体类型为Timestamp，每个Timestamp对象都带有正确的 tz 时区
1. 一个 dtype 为 datetime64[ns] 的 numpy.ndarray，其中的值已经被转换为 UTC 并且丢弃了时区

时区可以通过 dtype=object 保留

```python
In [14]: ser = pd.Series(pd.date_range("2000", periods=2, tz="CET"))

In [15]: ser.to_numpy(dtype=object)
Out[15]:
array([Timestamp('2000-01-01 00:00:00+0100', tz='CET'),
       Timestamp('2000-01-02 00:00:00+0100', tz='CET')], dtype=object)
```

或者用 dtype='datetime64[ns]' 丢弃时区

```python
In [16]: ser.to_numpy(dtype="datetime64[ns]")
Out[16]:
array(['1999-12-31T23:00:00.000000000', '2000-01-01T23:00:00.000000000'],
      dtype='datetime64[ns]')
```

获取 DataFrame 内部的“原始数据”可能有点复杂。当 DataFrame 所有列均为同一数据类型时，DataFrame.to_numpy() 将返回底层数据：

```python
In [17]: df.to_numpy()
Out[17]:
array([[-0.1732,  0.1192, -1.0442],
       [-0.8618, -2.1046, -0.4949],
       [ 1.0718,  0.7216, -0.7068],
       [-1.0396,  0.2719, -0.425 ],
       [ 0.567 ,  0.2762, -1.0874],
       [-0.6737,  0.1136, -1.4784],
       [ 0.525 ,  0.4047,  0.577 ],
       [-1.715 , -1.0393, -0.3706]])
```

如果 DataFrame 包含同质类型数据， 可以修改获取的ndarray，并且修改会反映到Dataframe中。对于异质数据（例如，DataFrame 的某些列不是全部相同的 dtype），则不能这样修改。获取的ndarray值属性与轴标签不同，本身不能被赋值。

> **备注**

> 在处理异质数据时，是用可以容纳所有涉数据的 dtype 生成 ndarray。例如，如果涉及字符串，dtype 为 object。如果只有浮点数和整数，则 dtype 为 float。

过去，pandas 推荐使用 Series.values 或 DataFrame.values 从 Series 或 DataFrame 中提取数据。你仍然可以在旧的代码库找到对这些的引用。现在，我们建议避免使用 .values，而是使用 .array 或 .to_numpy()。.values 有以下缺点：

1. 当 Series 包含[扩展类型](https://pandas.pydata.org/docs/development/extending.html#extending-extension-types "扩展类型")时，不确定 Series.values 返回的是 NumPy 数组还是扩展数组。Series.array 总是返回一个 ExtensionArray，并且永远不会复制数据。Series.to_numpy() 总是返回一个 NumPy 数组，可能需要复制/强制转换值。
1. 当 DataFrame 包含混合数据类型时，DataFrame.values 可能涉及复制数据和强制转换值到一个公共 dtype，这是一个相对昂贵的操作。DataFrame.to_numpy() 明确它使返回的 NumPy 数组可能不是 DataFrame 中相同数据的视图。

## 加速操作

pandas 支持使用 numexpr 库和 bottleneck 库来加速某些类型的二元数值和布尔运算。

这些库在处理大型数据集时特别有用，能显著地提升速度。numexpr 使用智能分块、缓存和多核心。bottleneck 是一组专门的 cython 程序，特别擅长处理包含 nans 的数组。

这里有一个示例（使用 100 列 x 100,000 行的 DataFrame）：

|操作|0.11.0 (ms)|旧版本 (ms)|与旧版本的比率|
|---|---|---|---|
|df1 > df2|13.32|125.35|0.1063|
|df1 * df2|21.71|36.63|0.5928|
|df1 + df2|22.04|36.50|0.6039|

强烈建议安装这两个库。有关更多安装信息，请参阅[推荐的依赖](https://pandas.pydata.org/docs/getting_started/install.html#install-recommended-dependencies "推荐的依赖")部分。

这两个库默认启用，你可以通过设置选项来控制：

```python
pd.set_option("compute.use_bottleneck", False)
pd.set_option("compute.use_numexpr", False)
```

## 灵活的二元操作

pandas 数据结构之间的二元操作中，有两个关键点：

- 高维（例如 DataFrame）和低维（例如 Series）对象之间的广播(broadcast)行为。
- 计算中的缺失数据。

我们将分别演示如何单独处理这些问题，尽管它们可以同时处理。

### 匹配 / 广播行为

DataFrame 有方法 add(), sub(), mul(), div() 和相关函数 radd(), rsub(), ...
用于执行二元操作。对于广播行为，主要关注输入为Series时。你可以使用 axis 关键字指定按 index 或 columns 进行匹配：

```python
In [18]: df = pd.DataFrame(
    {
        "one": pd.Series(np.random.randn(3), index=["a", "b", "c"]),
        "two": pd.Series(np.random.randn(4), index=["a", "b", "c", "d"]),
        "three": pd.Series(np.random.randn(3), index=["b", "c", "d"]),
    }
)


In [19]: df
Out[19]:
        one       two     three
a  1.394981  1.772517       NaN
b  0.343054  1.912123 -0.050390
c  0.695246  1.478369  1.227435
d       NaN  0.279344 -0.613172

In [20]: row = df.iloc[1]

In [21]: column = df["two"]

In [22]: df.sub(row, axis="columns")
Out[22]:
        one       two     three
a  1.051928 -0.139606       NaN
b  0.000000  0.000000  0.000000
c  0.352192 -0.433754  1.277825
d       NaN -1.632779 -0.562782

In [23]: df.sub(row, axis=1)
Out[23]:
        one       two     three
a  1.051928 -0.139606       NaN
b  0.000000  0.000000  0.000000
c  0.352192 -0.433754  1.277825
d       NaN -1.632779 -0.562782

In [24]: df.sub(column, axis="index")
Out[24]:
        one  two     three
a -0.377535  0.0       NaN
b -1.569069  0.0 -1.962513
c -0.783123  0.0 -0.250933
d       NaN  0.0 -0.892516

In [25]: df.sub(column, axis=0)
Out[25]:
        one  two     three
a -0.377535  0.0       NaN
b -1.569069  0.0 -1.962513
c -0.783123  0.0 -0.250933
d       NaN  0.0 -0.892516
```

此外，你可以将Serie与MultiIndexed DataFrame 的一个级别对齐。

```python
In [26]: dfmi = df.copy()

In [27]: dfmi.index = pd.MultiIndex.from_tuples(
    [(1, "a"), (1, "b"), (1, "c"), (2, "a")], names=["first", "second"]
)


In [28]: dfmi.sub(column, axis=0, level="second")
Out[28]:
                   one       two     three
first second
1     a      -0.377535  0.000000       NaN
      b      -1.569069  0.000000 -1.962513
      c      -0.783123  0.000000 -0.250933
2     a            NaN -1.493173 -2.385688
```

Series 和 Index 也支持内置的 divmod() 函数。这个函数同时执行向下取整除法和求模运算，返回与左侧参数类型相同的两个元组。例如：

```python
In [29]: s = pd.Series(np.arange(10))

In [30]: s
Out[30]:
0    0
1    1
2    2
3    3
4    4
5    5
6    6
7    7
8    8
9    9
dtype: int64

In [31]: div, rem = divmod(s, 3)

In [32]: div
Out[32]:
0    0
1    0
2    0
3    1
4    1
5    1
6    2
7    2
8    2
9    3
dtype: int64

In [33]: rem
Out[33]:
0    0
1    1
2    2
3    0
4    1
5    2
6    0
7    1
8    2
9    0
dtype: int64

In [34]: idx = pd.Index(np.arange(10))

In [35]: idx
Out[35]: Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int64')

In [36]: div, rem = divmod(idx, 3)

In [37]: div
Out[37]: Index([0, 0, 0, 1, 1, 1, 2, 2, 2, 3], dtype='int64')

In [38]: rem
Out[38]: Index([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype='int64')
```

我们也可以进行逐元素（elementwise）的 divmod()：

```python
In [39]: div, rem = divmod(s, [2, 2, 3, 3, 4, 4, 5, 5, 6, 6])

In [40]: div
Out[40]:
0    0
1    0
2    0
3    1
4    1
5    1
6    1
7    1
8    1
9    1
dtype: int64

In [41]: rem
Out[41]:
0    0
1    1
2    2
3    0
4    0
5    1
6    1
7    2
8    2
9    3
dtype: int64
```

### 缺失数据 / 填充值的操作

在 Series 和 DataFrame 中，算术函数有个 fill_value 参数，在同一位置最多一个值缺失时替代该缺失值。例如，当两个 DataFrame 相加时，你可能希望将 NaN 视为 0，除非两个 DataFrame 都缺失该值，在这种情况下，结果将是 NaN（如果需要的话，你稍后可以使用 fillna 用其他值替换 NaN）。

```python
In [42]: df2 = df.copy()

In [43]: df2.loc["a", "three"] = 1.0

In [44]: df
Out[44]:
        one       two     three
a  1.394981  1.772517       NaN
b  0.343054  1.912123 -0.050390
c  0.695246  1.478369  1.227435
d       NaN  0.279344 -0.613172

In [45]: df2
Out[45]:
        one       two     three
a  1.394981  1.772517  1.000000
b  0.343054  1.912123 -0.050390
c  0.695246  1.478369  1.227435
d       NaN  0.279344 -0.613172

In [46]: df + df2
Out[46]:
        one       two     three
a  2.789963  3.545034       NaN
b  0.686107  3.824246 -0.100780
c  1.390491  2.956737  2.454870
d       NaN  0.558688 -1.226343

In [47]: df.add(df2, fill_value=0)
Out[47]:
        one       two     three
a  2.789963  3.545034  1.000000
b  0.686107  3.824246 -0.100780
c  1.390491  2.956737  2.454870
d       NaN  0.558688 -1.226343
```


### 灵活的比较

Series 和 DataFrame 有二元比较方法 `eq`, `ne`, `lt`, `gt`, `le`, 和 `ge`，其行为类似于上述的二元算术操作：

```python
In [48]: df.gt(df2)
Out[48]:
     one    two  three
a  False  False  False
b  False  False  False
c  False  False  False
d  False  False  False

In [49]: df2.ne(df)
Out[49]:
     one    two  three
a  False  False   True
b  False  False  False
c  False  False  False
d   True  False  False
```

这些操作产生与左侧变量相同类型的 pandas 对象，该对象的 dtype 为 bool。这些 布尔 对象可以用于索引操作，参见[布尔索引](https://pandas.pydata.org/docs/user_guide/indexing.html#indexing-boolean "布尔索引")部分。

### 布尔规约

你可以对布尔结果应用规约：empty, any(), all(), 和 bool()，来总结布尔结果。

```python
In [50]: (df > 0).all()
Out[50]:
one      False
two       True
three    False
dtype: bool

In [51]: (df > 0).any()
Out[51]:
one      True
two      True
three    True
dtype: bool
```

你可以规约到一个最终的布尔值。

```python
In [52]: (df > 0).any().any()
Out[52]: True
```

你可以通过 empty 属性测试 pandas 对象是否为空。

```python
In [53]: df.empty
Out[53]: False

In [54]: pd.DataFrame(columns=list("ABC")).empty
Out[54]: True
```

> **警告**
>
> 直接测试pandas对象的真假将引发错误，因为这是测试空还是测试值，模糊不清。
>
> ```
> In [55]: if df:
>     print(True)
>
> ---------------------------------------------------------------------------
> ValueError                                Traceback (most recent call last)
> <ipython-input-55-318d08b2571a> in ?()
> ----> 1 if df:
>       2     print(True)
>
> ~/work/pandas/pandas/pandas/core/generic.py in ?(self)
>    1575     @final
>    1576     def __nonzero__(self) -> NoReturn:
> -> 1577         raise ValueError(
>    1578             f"The truth value of a {type(self).__name__} is ambiguous. "
>    1579             "Use a.empty, a.bool(), a.item(), a.any() or a.all()."
>    1580         )
>
> ValueError: The truth value of a DataFrame is ambiguous. Use a.empty, a.bool(), > > a.item(), a.any() or a.all().
> ```
>
> ```
> In [56]: df and df2
> ---------------------------------------------------------------------------
> ValueError                                Traceback (most recent call last)
> <ipython-input-56-b241b64bb471> in ?()
> ----> 1 df and df2
>
> ~/work/pandas/pandas/pandas/core/generic.py in ?(self)
>    1575     @final
>    1576     def __nonzero__(self) -> NoReturn:
> -> 1577         raise ValueError(
>    1578             f"The truth value of a {type(self).__name__} is ambiguous. "
>    1579             "Use a.empty, a.bool(), a.item(), a.any() or a.all()."
>    1580         )
>
> ValueError: The truth value of a DataFrame is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
> ```

有关更详细的讨论，请参阅[陷阱](https://pandas.pydata.org/docs/user_guide/gotchas.html#gotchas-truth "陷阱")部分。

### 比较对象是否相等

你可能经常发现有多种方法可以计算相同的结果。作为一个简单的例子，考虑 `df + df` 和 `df * 2`。要测试这两个计算产生相同的结果，使用上面展示的工具，你可能会想到用 `(df + df == df * 2).all()`。但实际上，这个表达式是 False：

```python
In [57]: df + df == df * 2
Out[57]:
     one   two  three
a   True  True  False
b   True  True   True
c   True  True   True
d  False  True   True

In [58]: (df + df == df * 2).all()
Out[58]:
one      False
two       True
three    False
dtype: bool
```

注意，布尔 DataFrame df + df == df * 2 包含一些 False 值！这是因为 NaN 不被视为相等：

```python
In [59]: np.nan == np.nan
Out[59]: False
```

因此，NDFrames（如 Series 和 DataFrame）有一个 equals() 方法用于测试相等性，其将相同位置的 NaN 视为相等。

```python
In [60]: (df + df).equals(df * 2)
Out[60]: True
```

注意，Series 或 DataFrame 索引需按相同顺序排列，才能使相等性为 True：

```python
In [61]: df1 = pd.DataFrame({"col": ["foo", 0, np.nan]})

In [62]: df2 = pd.DataFrame({"col": [np.nan, 0, "foo"]}, index=[2, 1, 0])

In [63]: df1.equals(df2)
Out[63]: False

In [64]: df1.equals(df2.sort_index())
Out[64]: True
```

### 比较类数组对象

pandas 数据结构与标量比较时是逐元素（element-wise）比较：

```python
In [65]: pd.Series(["foo", "bar", "baz"]) == "foo"
Out[65]:
0     True
1    False
2    False
dtype: bool

In [66]: pd.Index(["foo", "bar", "baz"]) == "foo"
Out[66]: array([ True, False, False])
```

具有相同长度的两个不同的类数组对象之间的比较也是逐元素比较：

```python
In [67]: pd.Series(["foo", "bar", "baz"]) == pd.Index(["foo", "bar", "qux"])
Out[67]:
0     True
1     True
2    False
dtype: bool

In [68]: pd.Series(["foo", "bar", "baz"]) == np.array(["foo", "bar", "qux"])
Out[68]:
0     True
1     True
2    False
dtype: bool
```

比较不同长度的 Index 或 Series 对象会引发 ValueError：

```python
In [69]: pd.Series(['foo', 'bar', 'baz']) == pd.Series(['foo', 'bar'])
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[69], line 1
----> 1 pd.Series(['foo', 'bar', 'baz']) == pd.Series(['foo', 'bar'])

File ~/work/pandas/pandas/pandas/core/ops/common.py:76, in _unpack_zerodim_and_defer.<locals>.new_method(self, other)
     72             return NotImplemented
     74 other = item_from_zerodim(other)
---> 76 return method(self, other)

File ~/work/pandas/pandas/pandas/core/arraylike.py:40, in OpsMixin.__eq__(self, other)
     38 @unpack_zerodim_and_defer("__eq__")
     39 def __eq__(self, other):
---> 40     return self._cmp_method(other, operator.eq)

File ~/work/pandas/pandas/pandas/core/series.py:6114, in Series._cmp_method(self, other, op)
   6111 res_name = ops.get_op_result_name(self, other)
   6113 if isinstance(other, Series) and not self._indexed_same(other):
-> 6114     raise ValueError("Can only compare identically-labeled Series objects")
   6116 lvalues = self._values
   6117 rvalues = extract_array(other, extract_numpy=True, extract_range=True)

ValueError: Can only compare identically-labeled Series objects

In [70]: pd.Series(['foo', 'bar', 'baz']) == pd.Series(['foo'])
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[70], line 1
----> 1 pd.Series(['foo', 'bar', 'baz']) == pd.Series(['foo'])

File ~/work/pandas/pandas/pandas/core/ops/common.py:76, in _unpack_zerodim_and_defer.<locals>.new_method(self, other)
     72             return NotImplemented
     74 other = item_from_zerodim(other)
---> 76 return method(self, other)

File ~/work/pandas/pandas/pandas/core/arraylike.py:40, in OpsMixin.__eq__(self, other)
     38 @unpack_zerodim_and_defer("__eq__")
     39 def __eq__(self, other):
---> 40     return self._cmp_method(other, operator.eq)

File ~/work/pandas/pandas/pandas/core/series.py:6114, in Series._cmp_method(self, other, op)
   6111 res_name = ops.get_op_result_name(self, other)
   6113 if isinstance(other, Series) and not self._indexed_same(other):
-> 6114     raise ValueError("Can only compare identically-labeled Series objects")
   6116 lvalues = self._values
   6117 rvalues = extract_array(other, extract_numpy=True, extract_range=True)

ValueError: Can only compare identically-labeled Series objects
```

### 组合重叠的数据集

偶尔需要组合两个相似数据集，其中一个数据集中的值优先级更高。例如，两个表示特定经济指标的数据，其中一个被认为是“更高质量”的。然而，较低质量的可能包含更早期的数据或具有更完整的数据覆盖。因此，我们希望组合两个 DataFrame 对象，其中一个 DataFrame 中的缺失值用另一个 DataFrame 中值填充。实现此操作的函数是 `combine_first()`：

```python
In [71]: df1 = pd.DataFrame(
    {"A": [1.0, np.nan, 3.0, 5.0, np.nan], "B": [np.nan, 2.0, 3.0, np.nan, 6.0]}
)


In [72]: df2 = pd.DataFrame(
    {
        "A": [5.0, 2.0, 4.0, np.nan, 3.0, 7.0],
        "B": [np.nan, np.nan, 3.0, 4.0, 6.0, 8.0],
    }
)


In [73]: df1
Out[73]:
     A    B
0  1.0  NaN
1  NaN  2.0
2  3.0  3.0
3  5.0  NaN
4  NaN  6.0

In [74]: df2
Out[74]:
     A    B
0  5.0  NaN
1  2.0  NaN
2  4.0  3.0
3  NaN  4.0
4  3.0  6.0
5  7.0  8.0

In [75]: df1.combine_first(df2)
Out[75]:
     A    B
0  1.0  NaN
1  2.0  2.0
2  3.0  3.0
3  5.0  4.0
4  3.0  6.0
5  7.0  8.0
```

### 通用的 DataFrame 组合

上面的 combine_first() 方法调用了更通用的 `DataFrame.combine()`。此方法接受另一个 DataFrame 和一个组合函数，对齐输入的 DataFrame，然后将成对的 Series（即，名称相同的列）传递给组合函数。

因此，例如，要重现上述的 combine_first()：

```python
In [76]: def combiner(x, y):
    return np.where(pd.isna(x), y, x)

In [77]: df1.combine(df2, combiner)
Out[77]:
     A    B
0  1.0  NaN
1  2.0  2.0
2  3.0  3.0
3  5.0  4.0
4  3.0  6.0
5  7.0  8.0
```

## 描述性统计

Series 和 DataFrame 上存在大量的方法用于计算描述性统计和其他相关操作。它们大多数是聚合函数（因此产生较低维度的结果），如 sum(), mean(), 和 quantile()，但有些，如 cumsum() 和 cumprod()，产生大小相同的对象。一般来说，这些方法接受一个 axis 参数，就像 ndarray.{sum, std, …}，axis 可以是名称或整数：

- Series：不需要 axis 参数
- DataFrame："index" (axis=0, 默认), "columns" (axis=1)

例如：

```python
In [78]: df
Out[78]:
        one       two     three
a  1.394981  1.772517       NaN
b  0.343054  1.912123 -0.050390
c  0.695246  1.478369  1.227435
d       NaN  0.279344 -0.613172

In [79]: df.mean(0)
Out[79]:
one      0.811094
two      1.360588
three    0.187958
dtype: float64

In [80]: df.mean(1)
Out[80]:
a    1.583749
b    0.734929
c    1.133683
d   -0.166914
dtype: float64
```
所有这些方法都有一个 skipna 参数，表示是否排除缺失数据（默认为 True）：

```python
In [81]: df.sum(0, skipna=False)
Out[81]:
one           NaN
two      5.442353
three         NaN
dtype: float64

In [82]: df.sum(axis=1, skipna=True)
Out[82]:
a    3.167498
b    2.204786
c    3.401050
d   -0.333828
dtype: float64
```

结合广播 / 算术行为，可以简洁地描述各种统计过程，如标准化（使数据均值为零，标准差为 1），非常简洁：

```python
In [83]: ts_stand = (df - df.mean()) / df.std()

In [84]: ts_stand.std()
Out[84]:
one      1.0
two      1.0
three    1.0
dtype: float64

In [85]: xs_stand = df.sub(df.mean(1), axis=0).div(df.std(1), axis=0)

In [86]: xs_stand.std(1)
Out[86]:
a    1.0
b    1.0
c    1.0
d    1.0
dtype: float64
```
请注意，像 cumsum() 和 cumprod() 这些方法保留 NaN 值的位置。这与 expanding() 和 rolling() 稍有不同，因为 NaN 行为进一步由 min_periods 参数决定。

```python
In [87]: df.cumsum()
Out[87]:
        one       two     three
a  1.394981  1.772517       NaN
b  1.738035  3.684640 -0.050390
c  2.433281  5.163008  1.177045
d       NaN  5.442353  0.563873
```

这里有一个常见的函数快速参考表。每个函数还接受一个可选的 level 参数，这只适用于对象具有[分层索引](https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#advanced-hierarchical "分层索引")的情况。

| 函数 | 描述 |
| --- | --- |
| count | 非Nan值的数量 |
| sum | 总和 |
| mean | 平均值 |
| median | 中位数 |
| min | 最小值 |
| max | 最大值 |
| mode | 模式 |
| abs | 绝对值 |
| prod | 乘积 |
| std | 标准差 |
| var | 方差 |
| sem | 样本平均数的标准误差 |
| skew | 样本偏度（第三阶矩） |
| kurt | 样本峰度（第四阶矩） |
| quantile | 样本分位数（百分比值） |
| cumsum | 累计和 |
| cumprod | 累计乘积 |
| cummax | 累计最大值 |
| cummin | 累计最小值 |

注意，一些 NumPy 方法，如 mean, std, 和 sum，将默认排除 Series 输入中的 Nan：

```python
In [88]: np.mean(df["one"])
Out[88]: 0.8110935116651192

In [89]: np.mean(df["one"].to_numpy())
Out[89]: nan
```

Series.nunique() 将返回 Series 中非Nan的唯一值的数量：

```python
In [90]: series = pd.Series(np.random.randn(500))

In [91]: series[20:500] = np.nan

In [92]: series[10:20] = 5

In [93]: series.nunique()
Out[93]: 11
```

### 摘要数据：describe

有一个方便的 describe() 函数，它计算 Series 或 DataFrame 列的各种摘要统计信息（不包括Nan值）：

```python
In [94]: series = pd.Series(np.random.randn(1000))

In [95]: series[::2] = np.nan

In [96]: series.describe()
Out[96]:
count    500.000000
mean      -0.021292
std        1.015906
min       -2.683763
25%       -0.699070
50%       -0.069718
75%        0.714483
max        3.160915
dtype: float64

In [97]: frame = pd.DataFrame(np.random.randn(1000, 5), columns=["a", "b", "c", "d", "e"])

In [98]: frame.iloc[::2] = np.nan

In [99]: frame.describe()
Out[99]:
                a           b           c           d           e
count  500.000000  500.000000  500.000000  500.000000  500.000000
mean     0.033387    0.030045   -0.043719   -0.051686    0.005979
std      1.017152    0.978743    1.025270    1.015988    1.006695
min     -3.000951   -2.637901   -3.303099   -3.159200   -3.188821
25%     -0.647623   -0.576449   -0.712369   -0.691338   -0.691115
50%      0.047578   -0.021499   -0.023888   -0.032652   -0.025363
75%      0.729907    0.775880    0.618896    0.670047    0.649748
max      2.740139    2.752332    3.004229    2.728702    3.240991
```

你可以选择特定的百分位数包含在输出中：

```python
In [100]: series.describe(percentiles=[0.05, 0.25, 0.75, 0.95])
Out[100]:
count    500.000000
mean      -0.021292
std        1.015906
min       -2.683763
5%        -1.645423
25%       -0.699070
50%       -0.069718
75%        0.714483
95%        1.711409
max        3.160915
dtype: float64
```

默认情况下，总是包括中位数。

对于非数值 Series 对象，describe() 将提供简单摘要，包括唯一值的数量和最频繁出现的值：

```python
In [101]: s = pd.Series(["a", "a", "b", "b", "a", "a", np.nan, "c", "d", "a"])

In [102]: s.describe()
Out[102]:
count     9
unique    4
top       a
freq      5
dtype: object
```

注意，对于混合类型 DataFrame 对象，describe() 将限制摘要仅包括数值列，或者如果没有数值列，则只包括分类列：

```python
In [103]: frame = pd.DataFrame({"a": ["Yes", "Yes", "No", "No"], "b": range(4)})

In [104]: frame.describe()
Out[104]:
              b
count  4.000000
mean   1.500000
std    1.290994
min    0.000000
25%    0.750000
50%    1.500000
75%    2.250000
max    3.000000
```

注意，对于混合类型 DataFrame 对象，describe() 将限制摘要仅包括数值列，或者如果没有数值列，则只包括分类列：

```python
In [105]: frame.describe(include=["object"])
Out[105]:
          a
count     4
unique    2
top     Yes
freq      2

In [106]: frame.describe(include=["number"])
Out[106]:
              b
count  4.000000
mean   1.500000
std    1.290994
min    0.000000
25%    0.750000
50%    1.500000
75%    2.250000
max    3.000000

In [107]: frame.describe(include="all")
Out[107]:
          a         b
count     4  4.000000
unique    2       NaN
top     Yes       NaN
freq      2       NaN
mean    NaN  1.500000
std     NaN  1.290994
min     NaN  0.000000
25%     NaN  0.750000
50%     NaN  1.500000
75%     NaN  2.250000
max     NaN  3.000000
```

该功能依赖于 [select_dtypes](https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html#basics-selectdtypes "select_dtypes")。有关接受的输入的详细信息，请参考那里。

### 最小/最大值的索引

Series 和 DataFrame 上的 idxmin() 和 idxmax() 函数计算最小和最大值的索引标签：

```python
In [108]: s1 = pd.Series(np.random.randn(5))

In [109]: s1
Out[109]:
0    1.118076
1   -0.352051
2   -1.242883
3   -1.277155
4   -0.641184
dtype: float64

In [110]: s1.idxmin(), s1.idxmax()
Out[110]: (3, 0)

df1 = pd.DataFrame(np.random.randn(5, 3), columns=["A", "B", "C"])

In [112]: df1
Out[112]:
          A         B         C
0 -0.327863 -0.946180 -0.137570
1 -0.186235 -0.257213 -0.486567
2 -0.507027 -0.871259 -0.111110
3  2.000339 -2.430505  0.089759
4 -0.321434 -0.033695  0.096271

In [113]: df1.idxmin(axis=0)
Out[113]:
A    2
B    3
C    1
dtype: int64

In [114]: df1.idxmax(axis=1)
Out[114]:
0    C
1    A
2    C
3    A
4    C
dtype: object
```

当存在多行（或列）匹配最小或最大值时，idxmin() 和 idxmax() 返回第一个匹配的索引：

```python
In [115]: df3 = pd.DataFrame([2, 1, 1, 3, np.nan], columns=["A"], index=list("edcba"))

In [116]: df3
Out[116]:
     A
e  2.0
d  1.0
c  1.0
b  3.0
a  NaN

In [117]: df3["A"].idxmin()
Out[117]: 'd'
```

> **备注**
> idxmin 和 idxmax 在 NumPy 中被称为 argmin 和 argmax。

### 值计数（直方图）/ mode

Series的value_counts()方法计算一维数组值的直方图。它也可以作为函数对常规数组使用：

```python
In [118]: data = np.random.randint(0, 7, size=50)

In [119]: data
Out[119]:
array([6, 6, 2, 3, 5, 3, 2, 5, 4, 5, 4, 3, 4, 5, 0, 2, 0, 4, 2, 0, 3, 2,
       2, 5, 6, 5, 3, 4, 6, 4, 3, 5, 6, 4, 3, 6, 2, 6, 6, 2, 3, 4, 2, 1,
       6, 2, 6, 1, 5, 4])

In [120]: s = pd.Series(data)

In [121]: s.value_counts()
Out[121]:
6    10
2    10
4     9
3     8
5     8
0     3
1     2
Name: count, dtype: int64
```

`value_counts()` 方法可以用来计算多个列的组合。默认使用所有列，但可以通过 `subset` 参数选择子集：

```python
In [122]: data = {"a": [1, 2, 3, 4], "b": ["x", "x", "y", "y"]}

In [123]: frame = pd.DataFrame(data)

In [124]: frame.value_counts()
Out[124]:
a  b
1  x    1
2  x    1
3  y    1
4  y    1
Name: count, dtype: int64
```

同样，你可以得到最频繁出现的值（即mode），在 Series 或 DataFrame 中：

```python
In [125]: s5 = pd.Series([1, 1, 3, 3, 3, 5, 5, 7, 7, 7])

In [126]: s5.mode()
Out[126]:
0    3
1    7
dtype: int64

In [127]: df5 = pd.DataFrame(
    {
        "A": np.random.randint(0, 7, size=50),
        "B": np.random.randint(-10, 15, size=50),
    }
)


In [128]: df5.mode()
Out[128]:
     A   B
0  1.0  -9
1  NaN  10
2  NaN  13
```

### 离散化和分位数

可以使用 cut()（基于值的区间）和 qcut()（基于样本分位数的区间）函数将连续值离散化：

```python
In [129]: arr = np.random.randn(20)

In [130]: factor = pd.cut(arr, 4)

In [131]: factor
Out[131]:
[(-0.251, 0.464], (-0.968, -0.251], (0.464, 1.179], (-0.251, 0.464], (-0.968, -0.251], ..., (-0.251, 0.464], (-0.968, -0.251], (-0.968, -0.251], (-0.968, -0.251], (-0.968, -0.251]]
Length: 20
Categories (4, interval[float64, right]): [(-0.968, -0.251] < (-0.251, 0.464] < (0.464, 1.179] <
                                           (1.179, 1.893]]

In [132]: factor = pd.cut(arr, [-5, -1, 0, 1, 5])

In [133]: factor
Out[133]:
[(0, 1], (-1, 0], (0, 1], (0, 1], (-1, 0], ..., (-1, 0], (-1, 0], (-1, 0], (-1, 0], (-1, 0]]
Length: 20
Categories (4, interval[int64, right]): [(-5, -1] < (-1, 0] < (0, 1] < (1, 5]]
```

qcut() 计算样本分位数。例如，我们可以将一些正态分布的数据分割成等大小的四分位数，如下所示：

```python
In [134]: arr = np.random.randn(30)

In [135]: factor = pd.qcut(arr, [0, 0.25, 0.5, 0.75, 1])

In [136]: factor
Out[136]:
[(0.569, 1.184], (-2.278, -0.301], (-2.278, -0.301], (0.569, 1.184], (0.569, 1.184], ..., (-0.301, 0.569], (1.184, 2.346], (1.184, 2.346], (-0.301, 0.569], (-2.278, -0.301]]
Length: 30
Categories (4, interval[float64, right]): [(-2.278, -0.301] < (-0.301, 0.569] < (0.569, 1.184] <
                                           (1.184, 2.346]]
```

我们还可以传递无限值来定义箱：

```python
In [137]: arr = np.random.randn(20)

In [138]: factor = pd.cut(arr, [-np.inf, 0, np.inf])

In [139]: factor
Out[139]:
[(-inf, 0.0], (0.0, inf], (0.0, inf], (-inf, 0.0], (-inf, 0.0], ..., (-inf, 0.0], (-inf, 0.0], (-inf, 0.0], (0.0, inf], (0.0, inf]]
Length: 20
Categories (2, interval[float64, right]): [(-inf, 0.0] < (0.0, inf]]
```

## 函数应用

要将你自己的或另一个库的函数应用于 pandas 对象，你应该了解以下三种方法。要使用哪个方法取决于你的函数是否期望对整个 DataFrame 或 Series 进行操作，是按行或按列进行，还是逐元素进行。

1. [表格级函数应用](https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html#tablewise-function-application "表格级函数应用")：pipe()
1. [行(列)级函数应用](https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html#row-or-column-wise-function-application "行(列)级函数应用")：apply()
1. [聚合 API](https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html#aggregation-api "聚合 API")：agg() 和 transform()
1. [逐元素函数应用](https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html#applying-elementwise-functions "逐元素函数应用")：map()

### 表格级函数应用

DataFrames 和 Series 可以传递到函数中。然而，如果函数需要在链中调用，请考虑使用 pipe() 方法。

首先进行一些设置：

```python
In [140]: def extract_city_name(df):
    """
    Chicago, IL -> Chicago for city_name column
    """
    df["city_name"] = df["city_and_code"].str.split(",").str.get(0)
    return df


In [141]: def add_country_name(df, country_name=None):
    """
    Chicago -> Chicago-US for city_name column
    """
    col = "city_name"
    df["city_and_country"] = df[col] + country_name
    return df


In [142]: df_p = pd.DataFrame({"city_and_code": ["Chicago, IL"]})
```

extract_city_name 和 add_country_name 是接受和返回 DataFrame 的函数。

现在比较以下内容：

```python
In [143]: add_country_name(extract_city_name(df_p), country_name="US")
Out[143]:
  city_and_code city_name city_and_country
0   Chicago, IL   Chicago        ChicagoUS
```

等同于：

```python
In [144]: df_p.pipe(extract_city_name).pipe(add_country_name, country_name="US")
Out[144]:
  city_and_code city_name city_and_country
0   Chicago, IL   Chicago        ChicagoUS
```

pandas 鼓励使用第二种风格，这种风格被称为方法链。pipe 使得在方法链中调用你自己的或另一个库的函数变得容易。

在上面的例子中，函数 extract_city_name 和 add_country_name 每个都期望一个 DataFrame 作为第一个位置参数。如果你希望应用的函数接受数据作为第二个参数，那么向 pipe 提供一个元组 (callable, data_keyword)。.pipe 将把 DataFrame 路由到指定的参数。

例如，我们可以使用 statsmodels 进行回归拟合。他们的 API 期望首先是一个公式，然后是一个 DataFrame 作为第二个参数 data。我们向 pipe 传递函数和关键字对 (sm.ols, 'data')：

```python
In [147]: import statsmodels.formula.api as sm

In [148]: bb = pd.read_csv("data/baseball.csv", index_col="id")

In [149]:(
        bb.query("h > 0")
        .assign(ln_h=lambda df: np.log(df.h))
        .pipe((sm.ols, "data"), "hr ~ ln_h + year + g + C(lg)")
        .fit()
        .summary()
    )
Out[149]:
<class 'statsmodels.iolib.summary.Summary'>
"""
                           OLS Regression Results
==============================================================================
Dep. Variable:                     hr   R-squared:                       0.685
Model:                            OLS   Adj. R-squared:                  0.665
Method:                 Least Squares   F-statistic:                     34.28
Date:                Tue, 22 Nov 2022   Prob (F-statistic):           3.48e-15
Time:                        05:34:17   Log-Likelihood:                -205.92
No. Observations:                  68   AIC:                             421.8
Df Residuals:                      63   BIC:                             432.9
Df Model:                           4
Covariance Type:            nonrobust
===============================================================================
                  coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------
Intercept   -8484.7720   4664.146     -1.819      0.074   -1.78e+04     835.780
C(lg)[T.NL]    -2.2736      1.325     -1.716      0.091      -4.922       0.375
ln_h           -1.3542      0.875     -1.547      0.127      -3.103       0.395
year            4.2277      2.324      1.819      0.074      -0.417       8.872
g               0.1841      0.029      6.258      0.000       0.125       0.243
==============================================================================
Omnibus:                       10.875   Durbin-Watson:                   1.999
Prob(Omnibus):                  0.004   Jarque-Bera (JB):               17.298
Skew:                           0.537   Prob(JB):                     0.000175
Kurtosis:                       5.225   Cond. No.                     1.49e+07
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.49e+07. This might indicate that there are
strong multicollinearity or other numerical problems.
"""
```

pipe 方法的灵感来自 Unix 管道，以及最近的 [dplyr](https://github.com/tidyverse/dplyr "dplyr") 和 [magrittr](https://github.com/tidyverse/magrittr "magrittr")，它们为 R 引入了流行的 (%>%)（读管道）运算符。这里 pipe 的实现非常干净，感觉在 Python 中很自然。我们鼓励你查看 pipe() 的源代码。

### 行(列)级函数应用

可以使用 apply() 方法将任意函数应用于 DataFrame 的轴上，就像描述性统计方法一样，它接受一个可选的 axis 参数：

```python
In [145]: df.apply(lambda x: np.mean(x))
Out[145]:
one      0.811094
two      1.360588
three    0.187958
dtype: float64

In [146]: df.apply(lambda x: np.mean(x), axis=1)
Out[146]:
a    1.583749
b    0.734929
c    1.133683
d   -0.166914
dtype: float64

In [147]: df.apply(lambda x: x.max() - x.min())
Out[147]:
one      1.051928
two      1.632779
three    1.840607
dtype: float64

In [148]: df.apply(np.cumsum)
Out[148]:
        one       two     three
a  1.394981  1.772517       NaN
b  1.738035  3.684640 -0.050390
c  2.433281  5.163008  1.177045
d       NaN  5.442353  0.563873

In [149]: df.apply(np.exp)
Out[149]:
        one       two     three
a  4.034899  5.885648       NaN
b  1.409244  6.767440  0.950858
c  2.004201  4.385785  3.412466
d       NaN  1.322262  0.541630
```

apply() 方法也接受字符串方法名。

```python
In [150]: df.apply("mean")
Out[150]:
one      0.811094
two      1.360588
three    0.187958
dtype: float64

In [151]: df.apply("mean", axis=1)
Out[151]:
a    1.583749
b    0.734929
c    1.133683
d   -0.166914
dtype: float64
```

apply() 方法的返回类型取决于传递给 apply() 的函数：

- 如果应用的函数返回一个 Series，则最终输出是一个 DataFrame。列与应用函数返回的 Series 的索引相匹配。
- 如果应用的函数返回任何其他类型，则最终输出是一个 Series。

这个默认行为可以通过 result_type 参数覆盖，它接受三个选项：reduce, broadcast, 和 expand。这些将决定列表类返回值如何扩展（或不扩展）为 DataFrame。

apply() 结合一些巧妙的方法可以用来回答许多关于数据集的问题。例如，假设我们想提取每列最大值出现的日期：

```python
In [152]: tsdf = pd.DataFrame(
    np.random.randn(1000, 3),
    columns=["A", "B", "C"],
    index=pd.date_range("1/1/2000", periods=1000),
)


In [153]: tsdf.apply(lambda x: x.idxmax())
Out[153]:
A   2000-08-06
B   2001-01-18
C   2001-07-18
dtype: datetime64[ns]
```

你也可以向 apply() 方法传递额外的参数和关键字参数。

```python
In [154]: def subtract_and_divide(x, sub, divide=1):
    return (x - sub) / divide


In [155]: df_udf = pd.DataFrame(np.ones((2, 2)))

In [156]: df_udf.apply(subtract_and_divide, args=(5,), divide=3)
Out[156]:
          0         1
0 -1.333333 -1.333333
1 -1.333333 -1.333333
```

另一个有用的特性是能够传递 Series 方法来对每一行或列执行一些 Series 操作：

```python
In [157]: tsdf = pd.DataFrame(
    np.random.randn(10, 3),
    columns=["A", "B", "C"],
    index=pd.date_range("1/1/2000", periods=10),
)


In [158]: tsdf.iloc[3:7] = np.nan

In [159]: tsdf
Out[159]:
                   A         B         C
2000-01-01 -0.158131 -0.232466  0.321604
2000-01-02 -1.810340 -3.105758  0.433834
2000-01-03 -1.209847 -1.156793 -0.136794
2000-01-04       NaN       NaN       NaN
2000-01-05       NaN       NaN       NaN
2000-01-06       NaN       NaN       NaN
2000-01-07       NaN       NaN       NaN
2000-01-08 -0.653602  0.178875  1.008298
2000-01-09  1.007996  0.462824  0.254472
2000-01-10  0.307473  0.600337  1.643950

In [160]: tsdf.apply(pd.Series.interpolate)
Out[160]:
                   A         B         C
2000-01-01 -0.158131 -0.232466  0.321604
2000-01-02 -1.810340 -3.105758  0.433834
2000-01-03 -1.209847 -1.156793 -0.136794
2000-01-04 -1.098598 -0.889659  0.092225
2000-01-05 -0.987349 -0.622526  0.321243
2000-01-06 -0.876100 -0.355392  0.550262
2000-01-07 -0.764851 -0.088259  0.779280
2000-01-08 -0.653602  0.178875  1.008298
2000-01-09  1.007996  0.462824  0.254472
2000-01-10  0.307473  0.600337  1.643950
```

最后，`apply()` 接受一个 `raw` 参数，默认为 False，它将每一行或列转换为 Series 然后应用函数。当设置为 True 时，传递给函数的将是一个 ndarray 对象，如果你不需要索引功能，这将有助于提升性能。

### 聚合 API

聚合 API 允许你以一种简洁的方式表达可能的多个聚合操作。这个 API 在 pandas 对象中非常相似，见 [groupby API](https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#groupby-aggregate "groupby API"), [window API](https://pandas.pydata.org/pandas-docs/stable/user_guide/window.html#window-overview "window API"), 和 [resample API](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-aggregate "resample API")。聚合的入口点是 `DataFrame.aggregate()`，或别名 `DataFrame.agg()`。

我们使用上面部分中使用过的类似数据：

```python
In [161]: tsdf = pd.DataFrame(
    np.random.randn(10, 3),
    columns=["A", "B", "C"],
    index=pd.date_range("1/1/2000", periods=10),
)


In [162]: tsdf.iloc[3:7] = np.nan

In [163]: tsdf
Out[163]:
                   A         B         C
2000-01-01  1.257606  1.004194  0.167574
2000-01-02 -0.749892  0.288112 -0.757304
2000-01-03 -0.207550 -0.298599  0.116018
2000-01-04       NaN       NaN       NaN
2000-01-05       NaN       NaN       NaN
2000-01-06       NaN       NaN       NaN
2000-01-07       NaN       NaN       NaN
2000-01-08  0.814347 -0.257623  0.869226
2000-01-09 -0.250663 -1.206601  0.896839
2000-01-10  2.169758 -1.333363  0.283157
```

使用单个函数等同于 apply()。你也可以传递字符串的方法名。这些将返回聚合输出的 Series：

```python
In [164]: tsdf.agg(lambda x: np.sum(x))
Out[164]:
A    3.033606
B   -1.803879
C    1.575510
dtype: float64

In [165]: tsdf.agg("sum")
Out[165]:
A    3.033606
B   -1.803879
C    1.575510
dtype: float64

# these are equivalent to a ``.sum()`` because we are aggregating
# on a single function
In [166]: tsdf.sum()
Out[166]:
A    3.033606
B   -1.803879
C    1.575510
dtype: float64
```

在 Series 上，单个聚合将返回一个标量值：

```python
In [167]: tsdf["A"].agg("sum")
Out[167]: 3.033606102414146
```

#### 使用多个函数进行聚合

你可以用列表传递多个聚合函数。传递的每个函数的结果将是结果 DataFrame 中的一行。这些行自然地以聚合函数的名称命名：

```python
In [168]: tsdf.agg(["sum"])
Out[168]:
            A         B        C
sum  3.033606 -1.803879  1.57551
```

多个函数产生多行：

```python
In [169]: tsdf.agg(["sum", "mean"])
Out[169]:
             A         B         C
sum   3.033606 -1.803879  1.575510
mean  0.505601 -0.300647  0.262585
```

在 Series 上，多个函数返回一个 Series，以函数名称为索引：

```python
In [170]: tsdf["A"].agg(["sum", "mean"])
Out[170]:
sum     3.033606
mean    0.505601
Name: A, dtype: float64
```

传递一个 lambda 函数将产生一个 \<lambda\> 命名的行：

```python
In [171]: tsdf["A"].agg(["sum", lambda x: x.mean()])
Out[171]:
sum         3.033606
<lambda>    0.505601
Name: A, dtype: float64
```

传递一个命名函数将产生该名称的行：

```python
In [172]: def mymean(x):
    return x.mean()


In [173]: tsdf["A"].agg(["sum", mymean])
Out[173]:
sum       3.033606
mymean    0.505601
Name: A, dtype: float64
```

#### 使用字典进行聚合

传递一个字典到 DataFrame.agg 允许你定制哪些函数应用于哪些列。注意，结果没有特定的顺序，你可以使用 OrderedDict 来保证顺序。

```python
In [174]: tsdf.agg({"A": "mean", "B": "sum"})
Out[174]:
A    0.505601
B   -1.803879
dtype: float64
```

传递一个值为列表的字典将生成一个 DataFrame 输出。你将得到一个矩阵式的输出，包含所有聚合器。输出将包含所有唯一的函数。那些没有为特定列注明的将是 NaN：

```python
In [175]: tsdf.agg({"A": ["mean", "min"], "B": "sum"})
Out[175]:
             A         B
mean  0.505601       NaN
min  -0.749892       NaN
sum        NaN -1.803879
```

#### 自定义描述

使用 .agg()，你可以很容易地创建一个自定义描述函数，类似于内置的描述函数。

```python
In [176]: from functools import partial

In [177]: q_25 = partial(pd.Series.quantile, q=0.25)

In [178]: q_25.__name__ = "25%"

In [179]: q_75 = partial(pd.Series.quantile, q=0.75)

In [180]: q_75.__name__ = "75%"

In [181]: tsdf.agg(["count", "mean", "std", "min", q_25, "median", q_75, "max"])
Out[181]:
               A         B         C
count   6.000000  6.000000  6.000000
mean    0.505601 -0.300647  0.262585
std     1.103362  0.887508  0.606860
min    -0.749892 -1.333363 -0.757304
25%    -0.239885 -0.979600  0.128907
median  0.303398 -0.278111  0.225365
75%     1.146791  0.151678  0.722709
max     2.169758  1.004194  0.896839
```

### 变换API

transform() 方法返回一个与原始索引相同的对象。这个 API 允许你同时提供多个操作，而不是一个接一个地提供。它的 API 与 .agg API 非常相似。

我们创建一个与上面部分中使用的相似的数据

```python
In [182]: tsdf = pd.DataFrame(
    np.random.randn(10, 3),
    columns=["A", "B", "C"],
    index=pd.date_range("1/1/2000", periods=10),
)


In [183]: tsdf.iloc[3:7] = np.nan

In [184]: tsdf
Out[184]:
                   A         B         C
2000-01-01 -0.428759 -0.864890 -0.675341
2000-01-02 -0.168731  1.338144 -1.279321
2000-01-03 -1.621034  0.438107  0.903794
2000-01-04       NaN       NaN       NaN
2000-01-05       NaN       NaN       NaN
2000-01-06       NaN       NaN       NaN
2000-01-07       NaN       NaN       NaN
2000-01-08  0.254374 -1.240447 -0.201052
2000-01-09 -0.157795  0.791197 -1.144209
2000-01-10 -0.030876  0.371900  0.061932
```

对整个数据进行变换。`.transform()` 允许输入函数为：一个 NumPy 函数，一个字符串函数名或一个用户定义的函数。

```python
In [185]: tsdf.transform(np.abs)
Out[185]:
                   A         B         C
2000-01-01  0.428759  0.864890  0.675341
2000-01-02  0.168731  1.338144  1.279321
2000-01-03  1.621034  0.438107  0.903794
2000-01-04       NaN       NaN       NaN
2000-01-05       NaN       NaN       NaN
2000-01-06       NaN       NaN       NaN
2000-01-07       NaN       NaN       NaN
2000-01-08  0.254374  1.240447  0.201052
2000-01-09  0.157795  0.791197  1.144209
2000-01-10  0.030876  0.371900  0.061932

In [186]: tsdf.transform("abs")
Out[186]:
                   A         B         C
2000-01-01  0.428759  0.864890  0.675341
2000-01-02  0.168731  1.338144  1.279321
2000-01-03  1.621034  0.438107  0.903794
2000-01-04       NaN       NaN       NaN
2000-01-05       NaN       NaN       NaN
2000-01-06       NaN       NaN       NaN
2000-01-07       NaN       NaN       NaN
2000-01-08  0.254374  1.240447  0.201052
2000-01-09  0.157795  0.791197  1.144209
2000-01-10  0.030876  0.371900  0.061932

In [187]: tsdf.transform(lambda x: x.abs())
Out[187]:
                   A         B         C
2000-01-01  0.428759  0.864890  0.675341
2000-01-02  0.168731  1.338144  1.279321
2000-01-03  1.621034  0.438107  0.903794
2000-01-04       NaN       NaN       NaN
2000-01-05       NaN       NaN       NaN
2000-01-06       NaN       NaN       NaN
2000-01-07       NaN       NaN       NaN
2000-01-08  0.254374  1.240447  0.201052
2000-01-09  0.157795  0.791197  1.144209
2000-01-10  0.030876  0.371900  0.061932
```

这里 transform() 接收到一个单一函数；这等同于一个 ufunc 应用。

```python
In [188]: np.abs(tsdf)
Out[188]:
                   A         B         C
2000-01-01  0.428759  0.864890  0.675341
2000-01-02  0.168731  1.338144  1.279321
2000-01-03  1.621034  0.438107  0.903794
2000-01-04       NaN       NaN       NaN
2000-01-05       NaN       NaN       NaN
2000-01-06       NaN       NaN       NaN
2000-01-07       NaN       NaN       NaN
2000-01-08  0.254374  1.240447  0.201052
2000-01-09  0.157795  0.791197  1.144209
2000-01-10  0.030876  0.371900  0.061932
```

传递单个函数给Series的.transform()将产生一个 Series 作为返回。

```python
In [189]: tsdf["A"].transform(np.abs)
Out[189]:
2000-01-01    0.428759
2000-01-02    0.168731
2000-01-03    1.621034
2000-01-04         NaN
2000-01-05         NaN
2000-01-06         NaN
2000-01-07         NaN
2000-01-08    0.254374
2000-01-09    0.157795
2000-01-10    0.030876
Freq: D, Name: A, dtype: float64
```

#### 使用多个函数进行变换

传递多个函数将产生一个列为多索引的 DataFrame。索引的第一层将是原始数据列名；第二层将是变换函数的名称。

```python
In [190]: tsdf.transform([np.abs, lambda x: x + 1])
Out[190]:
                   A                   B                   C
            absolute  <lambda>  absolute  <lambda>  absolute  <lambda>
2000-01-01  0.428759  0.571241  0.864890  0.135110  0.675341  0.324659
2000-01-02  0.168731  0.831269  1.338144  2.338144  1.279321 -0.279321
2000-01-03  1.621034 -0.621034  0.438107  1.438107  0.903794  1.903794
2000-01-04       NaN       NaN       NaN       NaN       NaN       NaN
2000-01-05       NaN       NaN       NaN       NaN       NaN       NaN
2000-01-06       NaN       NaN       NaN       NaN       NaN       NaN
2000-01-07       NaN       NaN       NaN       NaN       NaN       NaN
2000-01-08  0.254374  1.254374  1.240447 -0.240447  0.201052  0.798948
2000-01-09  0.157795  0.842205  0.791197  1.791197  1.144209 -0.144209
2000-01-10  0.030876  0.969124  0.371900  1.371900  0.061932  1.061932
```

传递多个函数给一个 Series 将产生一个 DataFrame。结果列名将是变换函数。

```python
In [191]: tsdf["A"].transform([np.abs, lambda x: x + 1])
Out[191]:
            absolute  <lambda>
2000-01-01  0.428759  0.571241
2000-01-02  0.168731  0.831269
2000-01-03  1.621034 -0.621034
2000-01-04       NaN       NaN
2000-01-05       NaN       NaN
2000-01-06       NaN       NaN
2000-01-07       NaN       NaN
2000-01-08  0.254374  1.254374
2000-01-09  0.157795  0.842205
2000-01-10  0.030876  0.969124
```

#### 使用字典进行变换

传递一个函数字典，将允许你定制每列的变换。

```python
In [192]: tsdf.transform({"A": np.abs, "B": lambda x: x + 1})
Out[192]:
                   A         B
2000-01-01  0.428759  0.135110
2000-01-02  0.168731  2.338144
2000-01-03  1.621034  1.438107
2000-01-04       NaN       NaN
2000-01-05       NaN       NaN
2000-01-06       NaN       NaN
2000-01-07       NaN       NaN
2000-01-08  0.254374 -0.240447
2000-01-09  0.157795  1.791197
2000-01-10  0.030876  1.371900
```

传递一个字典的函数列表将生成一个选择性变换的多索引 DataFrame。

```python
In [193]: tsdf.transform({"A": np.abs, "B": [lambda x: x + 1, "sqrt"]})
Out[193]:
                   A         B
            absolute  <lambda>      sqrt
2000-01-01  0.428759  0.135110       NaN
2000-01-02  0.168731  2.338144  1.156782
2000-01-03  1.621034  1.438107  0.661897
2000-01-04       NaN       NaN       NaN
2000-01-05       NaN       NaN       NaN
2000-01-06       NaN       NaN       NaN
2000-01-07       NaN       NaN       NaN
2000-01-08  0.254374 -0.240447       NaN
2000-01-09  0.157795  1.791197  0.889493
2000-01-10  0.030876  1.371900  0.609836
```

### 逐元素函数应用

由于并非所有函数都可以向量化（接受 NumPy 数组并返回另一个数组或值），map() 方法在 DataFrame （类似地map() 方法在 Series 上）接受任何 Python 函数，它接受单个值并返回单个值。例如：

```python
In [194]: df4 = df.copy()

In [195]: df4
Out[195]:
        one       two     three
a  1.394981  1.772517       NaN
b  0.343054  1.912123 -0.050390
c  0.695246  1.478369  1.227435
d       NaN  0.279344 -0.613172

In [196]: def f(x):
    return len(str(x))


In [197]: df4["one"].map(f)
Out[197]:
a    18
b    19
c    18
d     3
Name: one, dtype: int64

In [198]: df4.map(f)
Out[198]:
   one  two  three
a   18   17      3
b   19   18     20
c   18   18     16
d    3   19     19
```

Series.map() 还有一个额外的特性；它可以很容易地“链接”或“映射”另一个Series定义的值。这与合并/连接功能密切相关：

```python
In [199]: s = pd.Series(
    ["six", "seven", "six", "seven", "six"], index=["a", "b", "c", "d", "e"]
)


In [200]: t = pd.Series({"six": 6.0, "seven": 7.0})

In [201]: s
Out[201]:
a      six
b    seven
c      six
d    seven
e      six
dtype: object

In [202]: s.map(t)
Out[202]:
a    6.0
b    7.0
c    6.0
d    7.0
e    6.0
dtype: float64
```

## 重新索引和更改标签

reindex() 是 pandas 中的基本数据对齐方法。它用于实现几乎所有依赖于标签对齐功能的特性。要进行 重新索引 意味着要使数据符合给定的一组标签。这完成几件事情：

- 重新排序现有数据以匹配新的标签集
- 在没有数据的标签位置插入缺失值（NA）标记
- 如果指定，使用逻辑填充缺失标签的数据（对于处理时间序列数据非常重要）

这里是一个简单的示例：

```python
In [203]: s = pd.Series(np.random.randn(5), index=["a", "b", "c", "d", "e"])

In [204]: s
Out[204]:
a    1.695148
b    1.328614
c    1.234686
d   -0.385845
e   -1.326508
dtype: float64

In [205]: s.reindex(["e", "b", "f", "d"])
Out[205]:
e   -1.326508
b    1.328614
f         NaN
d   -0.385845
dtype: float64
```

在这里，f 标签不包含在 Series 中，因此在结果中出现为 NaN。

对于 DataFrame，你可以同时重新索引索引和列：

```python
In [206]: df
Out[206]:
        one       two     three
a  1.394981  1.772517       NaN
b  0.343054  1.912123 -0.050390
c  0.695246  1.478369  1.227435
d       NaN  0.279344 -0.613172

In [207]: df.reindex(index=["c", "f", "b"], columns=["three", "two", "one"])
Out[207]:
      three       two       one
c  1.227435  1.478369  0.695246
f       NaN       NaN       NaN
b -0.050390  1.912123  0.343054
```

注意 Index 对象包含实际的轴标签可以在对象间共享。所以如果我们有一个 Series 和一个 DataFrame，可以这样做：

```python
In [208]: rs = s.reindex(df.index)

In [209]: rs
Out[209]:
a    1.695148
b    1.328614
c    1.234686
d   -0.385845
dtype: float64

In [210]: rs.index is df.index
Out[210]: True
```

这意味着重新索引的 Series 的索引是与 DataFrame 的索引相同的 Python 对象。

DataFrame.reindex() 还支持一个“轴风格”的调用方式，其中你指定一个 labels 参数和它应用的 axis。

```python
In [211]: df.reindex(["c", "f", "b"], axis="index")
Out[211]:
        one       two     three
c  0.695246  1.478369  1.227435
f       NaN       NaN       NaN
b  0.343054  1.912123 -0.050390

In [212]: df.reindex(["three", "two", "one"], axis="columns")
Out[212]:
      three       two       one
a       NaN  1.772517  1.394981
b -0.050390  1.912123  0.343054
c  1.227435  1.478369  0.695246
d -0.613172  0.279344       NaN
```

> **参见**
>
> [MultiIndex / Advanced Indexing](https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#advanced "MultiIndex / Advanced Indexing")是一种更简洁的重新索引方式

> **备注**
>
> 当编写性能要求高的代码时，很有必要花一些时间考虑一下重新索引：许多操作在预对齐的数据上更快。添加两个未对齐的 DataFrame 在内部会触发重新索引步骤。对于探索性分析，你几乎不会注意到差异（因为 reindex 已经过高度优化），但当 CPU 周期很重要时，在到处显式的 reindex 调用可能会有影响。

### 重新索引以与另一个对象对齐

你可能希望重新索引对象使其标签与另一个对象相同。这种语法直接但略显冗长，这是一个常见的操作，所以提供了 reindex_like() 方法来简化这个过程：

```python
In [213]: df2 = df.reindex(["a", "b", "c"], columns=["one", "two"])

In [214]: df3 = df2 - df2.mean()

In [215]: df2
Out[215]:
        one       two
a  1.394981  1.772517
b  0.343054  1.912123
c  0.695246  1.478369

In [216]: df3
Out[216]:
        one       two
a  0.583888  0.051514
b -0.468040  0.191120
c -0.115848 -0.242634

In [217]: df.reindex_like(df2)
Out[217]:
        one       two
a  1.394981  1.772517
b  0.343054  1.912123
c  0.695246  1.478369
```

### 使用 `align` 将对象互相对齐

`align()` 方法是同时对齐两个对象的最快方式。它支持一个 `join` 参数（与连接和合并有关）：

> - `join='outer'`：取索引的并集（默认）
> - `join='left'`：使用调用对象的索引
> - `join='right'`：使用传递对象的索引
> - `join='inner'`：相交的索引

它返回一个元组，包含两个重新索引的对象：

```python
In [218]: s = pd.Series(np.random.randn(5), index=["a", "b", "c", "d", "e"])

In [219]: s1 = s[:4]

In [220]: s2 = s[1:]

In [221]: s1.align(s2)
Out[221]:
(a   -0.186646
 b   -1.692424
 c   -0.303893
 d   -1.425662
 e         NaN
 dtype: float64,
 a         NaN
 b   -1.692424
 c   -0.303893
 d   -1.425662
 e    1.114285
 dtype: float64)

In [222]: s1.align(s2, join="inner")
Out[222]:
(b   -1.692424
 c   -0.303893
 d   -1.425662
 dtype: float64,
 b   -1.692424
 c   -0.303893
 d   -1.425662
 dtype: float64)

In [223]: s1.align(s2, join="left")
Out[223]:
(a   -0.186646
 b   -1.692424
 c   -0.303893
 d   -1.425662
 dtype: float64,
 a         NaN
 b   -1.692424
 c   -0.303893
 d   -1.425662
 dtype: float64)
```

对于 DataFrame，join 方法将默认应用于index和columns：

```python
In [224]: df.align(df2, join="inner")
Out[224]:
(        one       two
 a  1.394981  1.772517
 b  0.343054  1.912123
 c  0.695246  1.478369,
         one       two
 a  1.394981  1.772517
 b  0.343054  1.912123
 c  0.695246  1.478369)
```

你也可以传递一个 axis 选项，只对指定的轴进行对齐：

```python
In [225]: df.align(df2, join="inner", axis=0)
Out[225]:
(        one       two     three
 a  1.394981  1.772517       NaN
 b  0.343054  1.912123 -0.050390
 c  0.695246  1.478369  1.227435,
         one       two
 a  1.394981  1.772517
 b  0.343054  1.912123
 c  0.695246  1.478369)
```

如果你向 DataFrame.align() 传递一个 Series，你可以使用 axis 参数选择在 DataFrame 的索引或列上进行对齐：

```python
In [226]: df.align(df2.iloc[0], axis=1)
Out[226]:
(        one     three       two
 a  1.394981       NaN  1.772517
 b  0.343054 -0.050390  1.912123
 c  0.695246  1.227435  1.478369
 d       NaN -0.613172  0.279344,
 one      1.394981
 three         NaN
 two      1.772517
 Name: a, dtype: float64)
```

### 重新索引时的填充

reindex() 接受一个可选参数 method，用于指定填充方法：

| method | 动作 |
| --- | --- |
| pad / ffill | 用前面的值填充 |
| bfill / backfill | 用后面的值填充 |
| nearest | 用最近的值填充 |

我们用一个简单的 Series 来说明这些填充方法：

```python
In [227]: rng = pd.date_range("1/3/2000", periods=8)

In [228]: ts = pd.Series(np.random.randn(8), index=rng)

In [229]: ts2 = ts.iloc[[0, 3, 6]]

In [230]: ts
Out[230]:
2000-01-03    0.183051
2000-01-04    0.400528
2000-01-05   -0.015083
2000-01-06    2.395489
2000-01-07    1.414806
2000-01-08    0.118428
2000-01-09    0.733639
2000-01-10   -0.936077
Freq: D, dtype: float64

In [231]: ts2
Out[231]:
2000-01-03    0.183051
2000-01-06    2.395489
2000-01-09    0.733639
Freq: 3D, dtype: float64

In [232]: ts2.reindex(ts.index)
Out[232]:
2000-01-03    0.183051
2000-01-04         NaN
2000-01-05         NaN
2000-01-06    2.395489
2000-01-07         NaN
2000-01-08         NaN
2000-01-09    0.733639
2000-01-10         NaN
Freq: D, dtype: float64

In [233]: ts2.reindex(ts.index, method="ffill")
Out[233]:
2000-01-03    0.183051
2000-01-04    0.183051
2000-01-05    0.183051
2000-01-06    2.395489
2000-01-07    2.395489
2000-01-08    2.395489
2000-01-09    0.733639
2000-01-10    0.733639
Freq: D, dtype: float64

In [234]: ts2.reindex(ts.index, method="bfill")
Out[234]:
2000-01-03    0.183051
2000-01-04    2.395489
2000-01-05    2.395489
2000-01-06    2.395489
2000-01-07    0.733639
2000-01-08    0.733639
2000-01-09    0.733639
2000-01-10         NaN
Freq: D, dtype: float64

In [235]: ts2.reindex(ts.index, method="nearest")
Out[235]:
2000-01-03    0.183051
2000-01-04    0.183051
2000-01-05    2.395489
2000-01-06    2.395489
2000-01-07    2.395489
2000-01-08    0.733639
2000-01-09    0.733639
2000-01-10    0.733639
Freq: D, dtype: float64
```

这些方法要求索引是有序的，递增或递减。

注意，相同的结果也可以通过 ffill (除了 method='nearest') 或者 [interpolate](https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html#missing-data-interpolate "interpolate") 获得：

```python
In [236]: ts2.reindex(ts.index).ffill()
Out[236]:
2000-01-03    0.183051
2000-01-04    0.183051
2000-01-05    0.183051
2000-01-06    2.395489
2000-01-07    2.395489
2000-01-08    2.395489
2000-01-09    0.733639
2000-01-10    0.733639
Freq: D, dtype: float64
```

reindex() 会在索引不是单调递增或递减时引发 ValueError。fillna() 和 interpolate() 不会对索引的顺序进行检查。

### 重新索引时限制填充

reindex() 接受 limit 和 tolerance 参数，为填充提供额外的控制。Limit 指定连续匹配的最大计数：

```python
In [237]: ts2.reindex(ts.index, method="ffill", limit=1)
Out[237]:
2000-01-03    0.183051
2000-01-04    0.183051
2000-01-05         NaN
2000-01-06    2.395489
2000-01-07    2.395489
2000-01-08         NaN
2000-01-09    0.733639
2000-01-10    0.733639
Freq: D, dtype: float64
```

相比之下，tolerance 指定了索引和索引器值之间的最大距离：

```python
In [238]: ts2.reindex(ts.index, method="ffill", tolerance="1 day")
Out[238]:
2000-01-03    0.183051
2000-01-04    0.183051
2000-01-05         NaN
2000-01-06    2.395489
2000-01-07    2.395489
2000-01-08         NaN
2000-01-09    0.733639
2000-01-10    0.733639
Freq: D, dtype: float64
```

注意，当用于 DatetimeIndex, TimedeltaIndex 或 PeriodIndex 时，tolerance 将被强制转换为 Timedelta（如果可能）。这使你能够使用适当的字符串指定容忍度。

### 从轴上删除标签

与 reindex 密切相关的 drop() 函数。它从轴上移除一组标签：

```python
In [239]: df
Out[239]:
        one       two     three
a  1.394981  1.772517       NaN
b  0.343054  1.912123 -0.050390
c  0.695246  1.478369  1.227435
d       NaN  0.279344 -0.613172

In [240]: df.drop(["a", "d"], axis=0)
Out[240]:
        one       two     three
b  0.343054  1.912123 -0.050390
c  0.695246  1.478369  1.227435

In [241]: df.drop(["one"], axis=1)
Out[241]:
        two     three
a  1.772517       NaN
b  1.912123 -0.050390
c  1.478369  1.227435
d  0.279344 -0.613172
```

注意，以下方法也有效，但不那么明显/清晰：

```python
In [242]: df.reindex(df.index.difference(["a", "d"]))
Out[242]:
        one       two     three
b  0.343054  1.912123 -0.050390
c  0.695246  1.478369  1.227435
```

### 重命名 / 映射标签

rename() 方法允许你根据一些映射（一个字典或 Series）或一个任意函数重新标记轴。

```python
In [243]: s
Out[243]:
a   -0.186646
b   -1.692424
c   -0.303893
d   -1.425662
e    1.114285
dtype: float64

In [244]: s.rename(str.upper)
Out[244]:
A   -0.186646
B   -1.692424
C   -0.303893
D   -1.425662
E    1.114285
dtype: float64
```

如果你传递一个函数，它必须在使用任何标签调用时返回一个值（并且必须产生一组唯一的值）。
也可以使用字典或 Series ：

```python
In [245]: df.rename(
    columns={"one": "foo", "two": "bar"},
    index={"a": "apple", "b": "banana", "d": "durian"},
)
Out[245]:
             foo       bar     three
apple   1.394981  1.772517       NaN
banana  0.343054  1.912123 -0.050390
c       0.695246  1.478369  1.227435
durian       NaN  0.279344 -0.613172
```

如果映射中不包含列/索引标签，则不会被重命名。注意，映射中的额外标签不会引起错误。

DataFrame.rename() 还支持一个“轴风格”式的调用，只需指定一个 mapper 和要应用该映射的 axis。

```python
In [246]: df.rename({"one": "foo", "two": "bar"}, axis="columns")
Out[246]:
        foo       bar     three
a  1.394981  1.772517       NaN
b  0.343054  1.912123 -0.050390
c  0.695246  1.478369  1.227435
d       NaN  0.279344 -0.613172

In [247]: df.rename({"a": "apple", "b": "banana", "d": "durian"}, axis="index")
Out[247]:
             one       two     three
apple   1.394981  1.772517       NaN
banana  0.343054  1.912123 -0.050390
c       0.695246  1.478369  1.227435
durian       NaN  0.279344 -0.613172
```

最后，rename() 方法也接受一个标量，用于更改 Series.name 属性。

```python
In [248]: s.rename("scalar-name")
Out[248]:
a   -0.186646
b   -1.692424
c   -0.303893
d   -1.425662
e    1.114285
Name: scalar-name, dtype: float64
```

DataFrame.rename_axis() 和 Series.rename_axis() 方法允许更改 MultiIndex 的特定名称（与更改标签相对）。

```python
In [249]: df = pd.DataFrame(
    {"x": [1, 2, 3, 4, 5, 6], "y": [10, 20, 30, 40, 50, 60]},
    index=pd.MultiIndex.from_product(
        [["a", "b", "c"], [1, 2]], names=["let", "num"]
    ),
)


In [250]: df
Out[250]:
         x   y
let num
a   1    1  10
    2    2  20
b   1    3  30
    2    4  40
c   1    5  50
    2    6  60

In [251]: df.rename_axis(index={"let": "abc"})
Out[251]:
         x   y
abc num
a   1    1  10
    2    2  20
b   1    3  30
    2    4  40
c   1    5  50
    2    6  60

In [252]: df.rename_axis(index=str.upper)
Out[252]:
         x   y
LET NUM
a   1    1  10
    2    2  20
b   1    3  30
    2    4  40
c   1    5  50
    2    6  60
```

## 迭代

pandas 对象的基本迭代行为取决于类型。当迭代 Series 时，它被视为类似数组，基本迭代产生值。DataFrame 遵循字典风格的约定，迭代产生对象的“键”。

简而言之，基本迭代（for i in object）产生：

- Series：值
- DataFrame：列标签

因此，例如，迭代 DataFrame 会给你列名：

```python
In [253]: df = pd.DataFrame(
    {"col1": np.random.randn(3), "col2": np.random.randn(3)}, index=["a", "b", "c"]
)


In [254]: for col in df:
    print(col)

col1
col2
```

pandas 对象还有 items() 方法，用于迭代键值对。

要迭代 DataFrame 的行，可以使用以下方法：

- iterrows()：以（索引，Series）对的形式迭代 DataFrame 的行。 这将行转换为 Series 对象，可能会改变数据类型并有一些性能影响。
- itertuples()：以 namedtuples 的形式迭代 DataFrame 的行。 这比 iterrows() 快得多，并且在大多数情况下更可取，用于迭代 DataFrame 的值。

> **警告**
>
> 迭代 pandas 对象通常是慢的。在许多情况下，不需要迭代行，可以通过以下方法避免：
> - 寻找 向量化 解决方案：许多操作可以使用内置方法或 NumPy 函数，（布尔）索引，……
> - 当你有一个不能直接在整个 DataFrame/Series 上工作的函数时，最好使用 apply() 而不是迭代。参见[函数应用](https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html#basics-apply "函数应用")。
> - 如果你需要对值进行迭代操作但性能很重要，请考虑使用 cython 或 numba 编写内部循环。参见[提高性能](https://pandas.pydata.org/pandas-docs/stable/user_guide/enhancingperf.html#enhancingperf "提高性能")部分，了解这种方法的一些示例。

> **警告**
>
> **绝不修改**你正在迭代的东西。这在所有情况下都不保证有效。根据数据类型，迭代器返回的是副本而不是视图，并且写入它将没有任何效果！
> 例如，在以下情况下设置值没有任何效果：
>
> ```
> In [255]: df = pd.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
>
> In [256]: for index, row in df.iterrows():
>     row["a"] = 10
>
>
> In [257]: df
> Out[257]:
>    a  b
> 0  1  a
> 1  2  b
> 2  3  c
> ```

### items

与字典风格接口一致，items() 迭代键值对：

- Series：（索引，标量值）对
- DataFrame：（列，Series）对

例如：

```python
In [258]: for label, ser in df.items():
    print(label)
    print(ser)

a
0    1
1    2
2    3
Name: a, dtype: int64
b
0    a
1    b
2    c
Name: b, dtype: object
```

### iterrows

iterrows() 允许你迭代 DataFrame 的行，作为 Series 对象。它返回一个迭代器，产生每个索引值以及包含每行数据的 Series：

```python
In [259]: for row_index, row in df.iterrows():
    print(row_index, row, sep="\n")

0
a    1
b    a
Name: 0, dtype: object
1
a    2
b    b
Name: 1, dtype: object
2
a    3
b    c
Name: 2, dtype: object
```

> **注意**
>
> 因为 iterrows() 返回每个行的 Series，它不会保留行的 dtypes（DataFrame 会保留列的 dtypes）。例如，
>
> ```
> In [260]: df_orig = pd.DataFrame([[1, 1.5]], columns=["int", "float"])
>
> In [261]: df_orig.dtypes
> Out[261]:
> int        int64
> float    float64
> dtype: object
>
> In [262]: row = next(df_orig.iterrows())[1]
>
> In [263]: row
> Out[263]:
> int      1.0
> float    1.5
> Name: 0, dtype: float64
> ```
>
> 所有 row 中的值，作为 Series 返回，现在都被转换为浮点数，也包括列 x 中的原始整数值：
>
> ```
> In [264]: row["int"].dtype
> Out[264]: dtype('float64')
>
> In [265]: df_orig["int"].dtype
> Out[265]: dtype('int64')
> ```
>
> 为了在迭代行时保留 dtypes，最好使用 itertuples()，它返回值的 namedtuples，并且通常比 iterrows() 快得多。

例如，转置 DataFrame 的一种牵强的方法是：

```python
In [266]: df2 = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

In [267]: print(df2)
   x  y
0  1  4
1  2  5
2  3  6

In [268]: print(df2.T)
   0  1  2
x  1  2  3
y  4  5  6

In [269]: df2_t = pd.DataFrame({idx: values for idx, values in df2.iterrows()})

In [270]: print(df2_t)
   0  1  2
x  1  2  3
y  4  5  6
```

### itertuples

itertuples() 方法将返回一个迭代器，为 DataFrame 中的每一行产生一个 namedtuple。第一元素将是行的相应索引值，而其余值是行值。

例如：

```python
In [271]: for row in df.itertuples():
    print(row)

Pandas(Index=0, a=1, b='a')
Pandas(Index=1, a=2, b='b')
Pandas(Index=2, a=3, b='c')
```

这个方法不会将行转换为 Series 对象；它只是返回一个包含值的 namedtuple。因此，itertuples() 保留了数据类型的值，并且通常比 iterrows() 快。

> **备注**
>
> 如果列名是无效的 Python 标识符、重复的，或以下划线开头，列名将被重命名为位置名称。有大量列（>255）时，将返回普通元组。

## .dt 访问器

日期时间/周期类型的Series 有一个访问器，用于简洁地返回类似日期时间的属性。这将返回一个 Series，索引与现有 Series 相同。

```python
# datetime
In [272]: s = pd.Series(pd.date_range("20130101 09:10:12", periods=4))

In [273]: s
Out[273]:
0   2013-01-01 09:10:12
1   2013-01-02 09:10:12
2   2013-01-03 09:10:12
3   2013-01-04 09:10:12
dtype: datetime64[ns]

In [274]: s.dt.hour
Out[274]:
0    9
1    9
2    9
3    9
dtype: int32

In [275]: s.dt.second
Out[275]:
0    12
1    12
2    12
3    12
dtype: int32

In [276]: s.dt.day
Out[276]:
0    1
1    2
2    3
3    4
dtype: int32
```

这使得像这样的表达式变得容易：

```python
In [277]: s[s.dt.day == 2]
Out[277]:
1   2013-01-02 09:10:12
dtype: datetime64[ns]
```

你可以很容易地进行时区的转换：

```python
In [278]: stz = s.dt.tz_localize("US/Eastern")

In [279]: stz
Out[279]:
0   2013-01-01 09:10:12-05:00
1   2013-01-02 09:10:12-05:00
2   2013-01-03 09:10:12-05:00
3   2013-01-04 09:10:12-05:00
dtype: datetime64[ns, US/Eastern]

In [280]: stz.dt.tz
Out[280]: <DstTzInfo 'US/Eastern' LMT-1 day, 19:04:00 STD>
```

你也可以链式执行这些类型的操作：

```python
In [281]: s.dt.tz_localize("UTC").dt.tz_convert("US/Eastern")
Out[281]:
0   2013-01-01 04:10:12-05:00
1   2013-01-02 04:10:12-05:00
2   2013-01-03 04:10:12-05:00
3   2013-01-04 04:10:12-05:00
dtype: datetime64[ns, US/Eastern]
```

你也可以使用 Series.dt.strftime() 将日期时间值格式化为字符串，它支持与标准 strftime() 相同的格式。

```python
# DatetimeIndex
In [282]: s = pd.Series(pd.date_range("20130101", periods=4))

In [283]: s
Out[283]:
0   2013-01-01
1   2013-01-02
2   2013-01-03
3   2013-01-04
dtype: datetime64[ns]

In [284]: s.dt.strftime("%Y/%m/%d")
Out[284]:
0    2013/01/01
1    2013/01/02
2    2013/01/03
3    2013/01/04
dtype: object
```

```python
# PeriodIndex
In [285]: s = pd.Series(pd.period_range("20130101", periods=4))

In [286]: s
Out[286]:
0    2013-01-01
1    2013-01-02
2    2013-01-03
3    2013-01-04
dtype: period[D]

In [287]: s.dt.strftime("%Y/%m/%d")
Out[287]:
0    2013/01/01
1    2013/01/02
2    2013/01/03
3    2013/01/04
dtype: object
```

.dt 访问器适用于周期和时间差类型。

```python
# period
In [288]: s = pd.Series(pd.period_range("20130101", periods=4, freq="D"))

In [289]: s
Out[289]:
0    2013-01-01
1    2013-01-02
2    2013-01-03
3    2013-01-04
dtype: period[D]

In [290]: s.dt.year
Out[290]:
0    2013
1    2013
2    2013
3    2013
dtype: int64

In [291]: s.dt.day
Out[291]:
0    1
1    2
2    3
3    4
dtype: int64
```

```python
# timedelta
In [292]: s = pd.Series(pd.timedelta_range("1 day 00:00:05", periods=4, freq="s"))

In [293]: s
Out[293]:
0   1 days 00:00:05
1   1 days 00:00:06
2   1 days 00:00:07
3   1 days 00:00:08
dtype: timedelta64[ns]

In [294]: s.dt.days
Out[294]:
0    1
1    1
2    1
3    1
dtype: int64

In [295]: s.dt.seconds
Out[295]:
0    5
1    6
2    7
3    8
dtype: int32

In [296]: s.dt.components
Out[296]:
   days  hours  minutes  seconds  milliseconds  microseconds  nanoseconds
0     1      0        0        5             0             0            0
1     1      0        0        6             0             0            0
2     1      0        0        7             0             0            0
3     1      0        0        8             0             0            0
```

> **备注**
>
> Series.dt 会在你访问非日期时间类型的值时引发 TypeError。

## 向量化字符串方法

Series 配备了一组字符串处理方法，可以轻松操作数组中的每个元素。最重要的是，这些方法自动排除缺失/NA值。这些方法是通过 Series 的 str 属性访问的，通常名称与内置字符串方法相匹配。例如：

```python
In [297]: s = pd.Series(
    ["A", "B", "C", "Aaba", "Baca", np.nan, "CABA", "dog", "cat"], dtype="string"
)


In [298]: s.str.lower()
Out[298]:
0       a
1       b
2       c
3    aaba
4    baca
5    <NA>
6    caba
7     dog
8     cat
dtype: string
```

还提供了强大的模式匹配方法，但请注意，默认情况下（在某些情况下总是）模式匹配使用[正则表达式](https://docs.python.org/3/library/re.html "正则表达式")。

参阅[向量化字符串方法](https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html#text-string-methods "向量化字符串方法")

## 排序

pandas 支持三种类型的排序：按索引标签排序、按列名排序以及按两者的组合排序。

### 按索引排序

Series.sort_index() 和 DataFrame.sort_index() 方法用于按 pandas 对象的索引排序。

```python
In [299]: df = pd.DataFrame(
    {
        "one": pd.Series(np.random.randn(3), index=["a", "b", "c"]),
        "two": pd.Series(np.random.randn(4), index=["a", "b", "c", "d"]),
        "three": pd.Series(np.random.randn(3), index=["b", "c", "d"]),
    }
)


In [300]: unsorted_df = df.reindex(
    index=["a", "d", "c", "b"], columns=["three", "two", "one"]
)


In [301]: unsorted_df
Out[301]:
      three       two       one
a       NaN -1.152244  0.562973
d -0.252916 -0.109597       NaN
c  1.273388 -0.167123  0.640382
b -0.098217  0.009797 -1.299504

# DataFrame
In [302]: unsorted_df.sort_index()
Out[302]:
      three       two       one
a       NaN -1.152244  0.562973
b -0.098217  0.009797 -1.299504
c  1.273388 -0.167123  0.640382
d -0.252916 -0.109597       NaN

In [303]: unsorted_df.sort_index(ascending=False)
Out[303]:
      three       two       one
d -0.252916 -0.109597       NaN
c  1.273388 -0.167123  0.640382
b -0.098217  0.009797 -1.299504
a       NaN -1.152244  0.562973

In [304]: unsorted_df.sort_index(axis=1)
Out[304]:
        one     three       two
a  0.562973       NaN -1.152244
d       NaN -0.252916 -0.109597
c  0.640382  1.273388 -0.167123
b -1.299504 -0.098217  0.009797

# Series
In [305]: unsorted_df["three"].sort_index()
Out[305]:
a         NaN
b   -0.098217
c    1.273388
d   -0.252916
Name: three, dtype: float64
```

排序索引还支持一个 key 参数，该参数接受一个可应用于正在排序的索引的可调用函数。对于 MultiIndex 对象，key 被应用于指定的级别。

```python
In [306]: s1 = pd.DataFrame({"a": ["B", "a", "C"], "b": [1, 2, 3], "c": [2, 3, 4]}).set_index(
    list("ab")
)


In [307]: s1
Out[307]:
     c
a b
B 1  2
a 2  3
C 3  4
```

```python
In [308]: s1.sort_index(level="a")
Out[308]:
     c
a b
B 1  2
C 3  4
a 2  3

In [309]: s1.sort_index(level="a", key=lambda idx: idx.str.lower())
Out[309]:
     c
a b
a 2  3
B 1  2
C 3  4
```

对于按值排序，参见[按值排序](https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html#basics-sort-value-key "按值排序")。

### 按值排序

Series.sort_values() 方法用于按 Series 的值进行排序。DataFrame.sort_values() 方法用于按 DataFrame 的列或行值进行排序。DataFrame.sort_values() 的可选参数 by 用于指定一个或多个列来确定排序顺序。

```python
In [310]: df1 = pd.DataFrame(
    {"one": [2, 1, 1, 1], "two": [1, 3, 2, 4], "three": [5, 4, 3, 2]}
)


In [311]: df1.sort_values(by="two")
Out[311]:
   one  two  three
0    2    1      5
2    1    2      3
1    1    3      4
3    1    4      2
```

by 参数可以接收列名列表，例如：

```python
In [312]: df1[["one", "two", "three"]].sort_values(by=["one", "two"])
Out[312]:
   one  two  three
2    1    2      3
1    1    3      4
3    1    4      2
0    2    1      5
```

这些方法通过 na_position 参数对 Nan 值进行特殊处理，：

```python
In [313]: s[2] = np.nan

In [314]: s.sort_values()
Out[314]:
0       A
3    Aaba
1       B
4    Baca
6    CABA
8     cat
7     dog
2    <NA>
5    <NA>
dtype: string

In [315]: s.sort_values(na_position="first")
Out[315]:
2    <NA>
5    <NA>
0       A
3    Aaba
1       B
4    Baca
6    CABA
8     cat
7     dog
dtype: string
```

排序也支持 key 参数，该参数接受一个可应用于正在排序的值的可调用函数。

```python
In [316]: s1 = pd.Series(["B", "a", "C"])

In [317]: s1.sort_values()
Out[317]:
0    B
2    C
1    a
dtype: object

In [318]: s1.sort_values(key=lambda x: x.str.lower())
Out[318]:
1    a
0    B
2    C
dtype: object
```

key 将接收 Series 的值，并应返回具有相同形状的 Series 或数组，其中包含转换后的值。对于 DataFrame 对象，key 应用于每列，因此 key 仍应期望一个 Series 并返回一个 Series，例如：

```python
In [319]: df = pd.DataFrame({"a": ["B", "a", "C"], "b": [1, 2, 3]})

In [320]: df.sort_values(by="a")
Out[320]:
   a  b
0  B  1
2  C  3
1  a  2

In [321]: df.sort_values(by="a", key=lambda col: col.str.lower())
Out[321]:
   a  b
1  a  2
0  B  1
2  C  3
```

### 按索引和值排序

DataFrame.sort_values() 的 by 参数可以是索引和列名的组合

```python
# Build MultiIndex
In [322]: idx = pd.MultiIndex.from_tuples(
    [("a", 1), ("a", 2), ("a", 2), ("b", 2), ("b", 1), ("b", 1)]
)


In [323]: idx.names = ["first", "second"]

# Build DataFrame
In [32]: df_multi = pd.DataFrame({"A": np.arange(6, 0, -1)}, index=idx)

In [325]: df_multi
Out[325]:
              A
first second
a     1       6
      2       5
      2       4
b     2       3
      1       2
      1       1
```

按 ‘second’ (index) 和 ‘A’ (column) 排序

```python
In [326]: df_multi.sort_values(by=["second", "A"])
Out[326]:
              A
first second
b     1       1
      1       2
a     1       6
b     2       3
a     2       4
      2       5
```

> **注意**
>
> 如果字符串同时匹配列名和索引名称，则会发出警告，并且列优先。这将在未来的版本中导致歧义错误。

### searchsorted

Series 有一个 searchsorted() 方法，其工作方式类似于 numpy.ndarray.searchsorted()。

```python
In [327]: ser = pd.Series([1, 2, 3])

In [328]: ser.searchsorted([0, 3])
Out[328]: array([0, 2])

In [329]: ser.searchsorted([0, 4])
Out[329]: array([0, 3])

In [330]: ser.searchsorted([1, 3], side="right")
Out[330]: array([1, 3])

In [331]: ser.searchsorted([1, 3], side="left")
Out[331]: array([0, 2])

In [332]: ser = pd.Series([3, 1, 2])

In [333]: ser.searchsorted([0, 3], sorter=np.argsort(ser))
Out[333]: array([0, 2])
```

### 最小/最大值

Series 有 nsmallest() 和 nlargest() 方法，它们返回最小或最大的 n 个值。对于一个大型 Series，这可以比对整个 Series 进行排序然后调用 head(n) 的结果要快得多。

```python
In [334]: s = pd.Series(np.random.permutation(10))

In [335]: s
Out[335]:
0    2
1    0
2    3
3    7
4    1
5    5
6    9
7    6
8    8
9    4
dtype: int64

In [336]: s.sort_values()
Out[336]:
1    0
4    1
0    2
2    3
9    4
5    5
7    6
3    7
8    8
6    9
dtype: int64

In [337]: s.nsmallest(3)
Out[337]:
1    0
4    1
0    2
dtype: int64

In [338]: s.nlargest(3)
Out[338]:
6    9
8    8
3    7
dtype: int64
```

DataFrame 也有 nlargest 和 nsmallest 方法。

```python
In [339]: df = pd.DataFrame(
    {
        "a": [-2, -1, 1, 10, 8, 11, -1],
        "b": list("abdceff"),
        "c": [1.0, 2.0, 4.0, 3.2, np.nan, 3.0, 4.0],
    }
)


In [340]: df.nlargest(3, "a")
Out[340]:
    a  b    c
5  11  f  3.0
3  10  c  3.2
4   8  e  NaN

In [341]: df.nlargest(5, ["a", "c"])
Out[341]:
    a  b    c
5  11  f  3.0
3  10  c  3.2
4   8  e  NaN
2   1  d  4.0
6  -1  f  4.0

In [342]: df.nsmallest(3, "a")
Out[342]:
   a  b    c
0 -2  a  1.0
1 -1  b  2.0
6 -1  f  4.0

In [343]: df.nsmallest(5, ["a", "c"])
Out[343]:
   a  b    c
0 -2  a  1.0
1 -1  b  2.0
6 -1  f  4.0
2  1  d  4.0
4  8  e  NaN
```

### 按 MultiIndex 列排序

当列是 MultiIndex 时，必须明确说明排序，并指定 by 参数中的所有级别。

```python
In [344]: df1.columns = pd.MultiIndex.from_tuples(
    [("a", "one"), ("a", "two"), ("b", "three")]
)


In [345]: df1.sort_values(by=("a", "two"))
Out[345]:
    a         b
  one two three
0   2   1     5
2   1   2     3
1   1   3     4
3   1   4     2
```

## 复制

pandas 对象的 copy() 方法复制底层数据（不复制轴索引，因为它们是不可变的）并返回一个新对象。注意，**很少需要复制对象**。例如，只有少数几种方法可以直接修改 DataFrame：

- 插入、删除或修改列。
- 设置 index 或 columns 属性。
- 对于同质数据，通过直接修改 values 属性或高级索引。

要明确，没有 pandas 方法会有修改你的数据的副作用；几乎每个方法都返回一个新对象，留下原始对象不受影响。如果数据被修改了，那是因为你明确地这样做了。

## dtypes

大多数情况下，pandas 使用 NumPy 数组和类型（dtype）作为 Series 或 DataFrame 的单个列。NumPy 提供了对 float、int、bool、timedelta64[ns] 和 datetime64[ns] 的支持（注意 NumPy 不支持带时区的 datetimes）。

pandas 和第三方库在一些地方 扩展 了 NumPy 的类型系统。本节描述了 pandas 内部所做的扩展类型。有关如何编写与 pandas 兼容的你自己的扩展的信息，请参阅[扩展类型](https://pandas.pydata.org/pandas-docs/stable/development/extending.html#extending-extension-types "扩展类型")。有关实现了扩展的第三方库列表，请参阅[生态系统](https://pandas.pydata.org/community/ecosystem.html "生态系统")页面。

下表列出了 pandas 扩展的所有数据类型。对于需要 dtype 参数的方法，可以使用字符串，如下所示。有关每种类型的更多信息，请参阅各自的文档部分。


 | Kind of Data | Data Type | Scalar | Array | String Aliases |
 | --- | --- | --- | --- | --- |
 | tz-aware datetime | DatetimeTZDtype | Timestamp | arrays.DatetimeArray | 'datetime64[ns, <tz>]'
 | Categorical | CategoricalDtype | (none) | Categorical | 'category'
 | period (time spans) | PeriodDtype | Period | arrays.PeriodArray 'Period[<freq>]' | 'period[<freq>]',
 | sparse | SparseDtype | (none) | arrays.SparseArray | 'Sparse', 'Sparse[int]', 'Sparse[float]'
 | intervals | IntervalDtype | Interval | arrays.IntervalArray | 'interval', 'Interval', 'Interval[<numpy_dtype>]', 'Interval[datetime64[ns, <tz>]]', 'Interval[timedelta64[<freq>]]'
 | nullable integer | Int64Dtype, … | (none) | arrays.IntegerArray | 'Int8', 'Int16', 'Int32', 'Int64', 'UInt8', 'UInt16', 'UInt32', 'UInt64'
 | nullable float | Float64Dtype, … | (none) | arrays.FloatingArray | 'Float32', 'Float64'
 | Strings | StringDtype | str | arrays.StringArray | 'string'
 | Boolean (with NA) | BooleanDtype | bool | arrays.BooleanArray | 'boolean'

pandas 有两种存储字符串的方式：

1. object 类型，可以容纳任何 Python 对象，包括字符串。
1. StringDtype，专门用于字符串。

通常，我们建议使用 StringDtype。有关更多信息，请参见[文本数据类型](https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html#text-types "文本数据类型")。

最后，可以使用 object 类型存储任意对象，但应尽可能避免（为了性能和与其他库及方法的互操作性）。有关更多信息，请参见[对象转换](https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html#basics-object-conversion "对象转换")。

一个方便的 dtypes 属性，用于 DataFrame，返回每列的数据类型 Series。

```python
In [346]: dft = pd.DataFrame(
    {
        "A": np.random.rand(3),
        "B": 1,
        "C": "foo",
        "D": pd.Timestamp("20010102"),
        "E": pd.Series([1.0] * 3).astype("float32"),
        "F": False,
        "G": pd.Series([1] * 3, dtype="int8"),
    }
)


In [347]: dft
Out[347]:
          A  B    C          D    E      F  G
0  0.035962  1  foo 2001-01-02  1.0  False  1
1  0.701379  1  foo 2001-01-02  1.0  False  1
2  0.281885  1  foo 2001-01-02  1.0  False  1

In [348]: dft.dtypes
Out[348]:
A          float64
B            int64
C           object
D    datetime64[s]
E          float32
F             bool
G             int8
dtype: object
```

在 Series 对象上，使用 dtype 属性。

```python
In [349]: dft["A"].dtype
Out[349]: dtype('float64')
```

如果 pandas 对象单个列中包含具有多种数据类型的数据，则该列的 dtype 将被选择为可以容纳所有数据类型的类型（object 是最通用的）。

```python
# these ints are coerced to floats
In [350]: pd.Series([1, 2, 3, 4, 5, 6.0])
Out[350]:
0    1.0
1    2.0
2    3.0
3    4.0
4    5.0
5    6.0
dtype: float64

# string data forces an ``object`` dtype
In [351]: pd.Series([1, 2, 3, 6.0, "foo"])
Out[351]:
0      1
1      2
2      3
3    6.0
4    foo
dtype: object
```

可以调用 DataFrame.dtypes.value_counts() 查看 各个类型的列数量

```python
In [352]: dft.dtypes.value_counts()
Out[352]:
float64          1
int64            1
object           1
datetime64[s]    1
float32          1
bool             1
int8             1
Name: count, dtype: int64
```

数值数据类型会传播并可以共存于 DataFrame 中。
如果传递了dtype（无论是直接通过 dtype 关键字、传递的 ndarray，还是传递的 Series），则它将在 DataFrame 操作中保持。此外，不同的数值数据类型**不会**合并。

```python
In [353]: df1 = pd.DataFrame(np.random.randn(8, 1), columns=["A"], dtype="float32")

In [354]: df1
Out[354]:
          A
0  0.224364
1  1.890546
2  0.182879
3  0.787847
4 -0.188449
5  0.667715
6 -0.011736
7 -0.399073

In [355]: df1.dtypes
Out[355]:
A    float32
dtype: object

In [356]: df2 = pd.DataFrame(
    {
        "A": pd.Series(np.random.randn(8), dtype="float16"),
        "B": pd.Series(np.random.randn(8)),
        "C": pd.Series(np.random.randint(0, 255, size=8), dtype="uint8"),  # [0,255] (range of uint8)
    }
)


In [357]: df2
Out[357]:
          A         B    C
0  0.823242  0.256090   26
1  1.607422  1.426469   86
2 -0.333740 -0.416203   46
3 -0.063477  1.139976  212
4 -1.014648 -1.193477   26
5  0.678711  0.096706    7
6 -0.040863 -1.956850  184
7 -0.357422 -0.714337  206

In [358]: df2.dtypes
Out[358]:
A    float16
B    float64
C      uint8
dtype: object
```

### 默认值

默认情况下，整数类型是 int64，浮点类型是 float64，与平台（32位或64位）无关。

```python
In [359]: pd.DataFrame([1, 2], columns=["a"]).dtypes
Out[359]:
a    int64
dtype: object

In [360]: pd.DataFrame({"a": [1, 2]}).dtypes
Out[360]:
a    int64
dtype: object

In [361]: pd.DataFrame({"a": 1}, index=list(range(2))).dtypes
Out[361]:
a    int64
dtype: object
```

请注意，Numpy 在创建数组时会选择平台依赖的类型。以下将在 32 位平台上产生 int32：

```python
In [362]: frame = pd.DataFrame(np.array([1, 2]))
```

### 向上转型

在组合其他类型时，类型可能会被悄悄地向上转型，这意味着它们会从当前类型被动提升（例如 int 转换为 float）。

```python
In [363]: df3 = df1.reindex_like(df2).fillna(value=0.0) + df2

In [364]: df3
Out[364]:
          A         B      C
0  1.047606  0.256090   26.0
1  3.497968  1.426469   86.0
2 -0.150862 -0.416203   46.0
3  0.724370  1.139976  212.0
4 -1.203098 -1.193477   26.0
5  1.346426  0.096706    7.0
6 -0.052599 -1.956850  184.0
7 -0.756495 -0.714337  206.0

In [365]: df3.dtypes
Out[365]:
A    float32
B    float64
C    float64
dtype: object
```

`DataFrame.to_numpy()` 将返回数据类型**低通用分母**，这意味着可以容纳**所有**数据类型的结果的 NumPy 数组的数据类型。这可能会导致一些**向上转型**。

```python
In [366]: df3.to_numpy().dtype
Out[366]: dtype('float64')
```

### astype

你可以使用 astype() 方法显式地将数据类型从一种转换为另一种。这些默认情况下会返回一个副本，即使数据类型未改变（传递 copy=False 可以更改此行为）。此外，如果 astype 操作无效，它们会引发异常。

向上转型始终根据 NumPy 规则进行。如果两种不同的数据类型参与运算，则会使用更通用的数据类型作为运算结果。

```python
In [367]: df3
Out[367]:
          A         B      C
0  1.047606  0.256090   26.0
1  3.497968  1.426469   86.0
2 -0.150862 -0.416203   46.0
3  0.724370  1.139976  212.0
4 -1.203098 -1.193477   26.0
5  1.346426  0.096706    7.0
6 -0.052599 -1.956850  184.0
7 -0.756495 -0.714337  206.0

In [368]: df3.dtypes
Out[368]:
A    float32
B    float64
C    float64
dtype: object

# conversion of dtypes
In [369]: df3.astype("float32").dtypes
Out[369]:
A    float32
B    float32
C    float32
dtype: object
```

使用 astype() 转换特定列到指定类型。

```python
In [370]: dft = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

In [371]: dft[["a", "b"]] = dft[["a", "b"]].astype(np.uint8)

In [372]: dft
Out[372]:
   a  b  c
0  1  4  7
1  2  5  8
2  3  6  9

In [373]: dft.dtypes
Out[373]:
a    uint8
b    uint8
c    int64
dtype: object
```

通过传递字典到 astype() 将某些列转换为特定数据类型。

```python
In [374]: dft1 = pd.DataFrame({"a": [1, 0, 1], "b": [4, 5, 6], "c": [7, 8, 9]})

In [375]: dft1 = dft1.astype({"a": np.bool_, "c": np.float64})

In [376]: dft1
Out[376]:
       a  b    c
0   True  4  7.0
1  False  5  8.0
2   True  6  9.0

In [377]: dft1.dtypes
Out[377]:
a       bool
b      int64
c    float64
dtype: object
```

> **注意**
>
> 当使用 astype() 和 loc() 将部分列转换为指定类型时，会发生向上转型。
>
> loc() 尝试将我们分配赋值的东西适配到当前数据类型，尽管 [] 想覆盖它们，采用右侧的数据类型。因此，以下代码产生了意想不到的结果。
>
> ```
> In [378]: dft = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
>
> In [379]: dft.loc[:, ["a", "b"]].astype(np.uint8).dtypes
> Out[379]:
> a    uint8
> b    uint8
> dtype: object
>
> In [380]: dft.loc[:, ["a", "b"]] = dft.loc[:, ["a", "b"]].astype(np.uint8)
>
> In [381]: dft.dtypes
> Out[381]:
> a    int64
> b    int64
> c    int64
> dtype: object
> ```

### 对象转换
pandas 提供了多种函数，用于强制将对象类型转换为其他类型。如果数据的类型已经正确，但存储在对象数组中，则可以使用 DataFrame.infer_objects() 和 Series.infer_objects() 方法将数据软转换为正确的类型。

```python
In [382]: import datetime

In [383]: df = pd.DataFrame(
    [
        [1, 2],
        ["a", "b"],
        [datetime.datetime(2016, 3, 2), datetime.datetime(2016, 3, 2)],
    ]
)


In [384]: df = df.T

In [385]: df
Out[385]:
   0  1                    2
0  1  a  2016-03-02 00:00:00
1  2  b  2016-03-02 00:00:00

In [386]: df.dtypes
Out[386]:
0    object
1    object
2    object
dtype: object
```

由于数据被转置，原始推理将所有列都存储为对象，infer_objects 将纠正这一点。

```python
In [387]: df.infer_objects().dtypes
Out[387]:
0             int64
1            object
2    datetime64[ns]
dtype: object
```

以下函数可用于一维对象数组或标量，将对象硬转换为指定类型：

- to_numeric() (转化为数字类型)

```python
In [388]: m = ["1.1", 2, 3]

In [389]: pd.to_numeric(m)
Out[389]: array([1.1, 2. , 3. ])
```

- to_datetime() (转化为日期时间类型)

```python
In [390]: import datetime

In [391]: m = ["2016-07-09", datetime.datetime(2016, 3, 2)]

In [392]: pd.to_datetime(m)
Out[392]: DatetimeIndex(['2016-07-09', '2016-03-02'], dtype='datetime64[ns]', freq=None)
```

- to_timedelta() (转化为timedelta)

```python
In [393]: m = ["5us", pd.Timedelta("1day")]

In [394]: pd.to_timedelta(m)
Out[394]: TimedeltaIndex(['0 days 00:00:00.000005', '1 days 00:00:00'], dtype='timedelta64[ns]', freq=None)
```
要强制转换，我们可以传入 errors 参数，指定 pandas 应如何处理无法转换为所需 dtype或对象的元素。默认情况下，error='raise'，这意味着在转换过程中遇到的任何错误都会被抛出。但是，如果 errors='coerce'，这些错误将被忽略，pandas 会将有问题的元素转换为 pd.NaT（用于日期时间和 timedelta）或 np.nan（用于数值）。如果读入的数据大多是所需的 dtype（如数字、日期时间），但偶尔会夹杂一些不符合要求的元素，而您希望将其表示为缺失，那么这可能会很有用：

```python
In [395]: import datetime

In [396]: m = ["apple", datetime.datetime(2016, 3, 2)]

In [397]: pd.to_datetime(m, errors="coerce")
Out[397]: DatetimeIndex(['NaT', '2016-03-02'], dtype='datetime64[ns]', freq=None)

In [398]: m = ["apple", 2, 3]

In [399]: pd.to_numeric(m, errors="coerce")
Out[399]: array([nan,  2.,  3.])

In [400]: m = ["apple", pd.Timedelta("1day")]

In [401]: pd.to_timedelta(m, errors="coerce")
Out[401]: TimedeltaIndex([NaT, '1 days'], dtype='timedelta64[ns]', freq=None)
```

除了对象转换外，to_numeric() 还提供了另一个参数 downcast，可以将新的（或已经存在的）数值数据向下转换为较小的 dtype，从而节省内存：

```python
In [402]: m = ["1", 2, 3]

In [403]: pd.to_numeric(m, downcast="integer")  # smallest signed int dtype
Out[403]: array([1, 2, 3], dtype=int8)

In [404]: pd.to_numeric(m, downcast="signed")  # same as 'integer'
Out[404]: array([1, 2, 3], dtype=int8)

In [405]: pd.to_numeric(m, downcast="unsigned")  # smallest unsigned int dtype
Out[405]: array([1, 2, 3], dtype=uint8)

In [406]: pd.to_numeric(m, downcast="float")  # smallest float dtype
Out[406]: array([1., 2., 3.], dtype=float32)
```

由于这些方法仅适用于一维数组、列表或标量，因此不能直接用于多维对象（如 DataFrames）。不过，通过 apply()，我们可以在每一列上有效地 “应用 ”函数：

```python
In [407]: import datetime

In [408]: df = pd.DataFrame([["2016-07-09", datetime.datetime(2016, 3, 2)]] * 2, dtype="O")

In [409]: df
Out[409]:
            0                    1
0  2016-07-09  2016-03-02 00:00:00
1  2016-07-09  2016-03-02 00:00:00

In [410]: df.apply(pd.to_datetime)
Out[410]:
           0          1
0 2016-07-09 2016-03-02
1 2016-07-09 2016-03-02

In [411]: df = pd.DataFrame([["1.1", 2, 3]] * 2, dtype="O")

In [412]: df
Out[412]:
     0  1  2
0  1.1  2  3
1  1.1  2  3

In [413]: df.apply(pd.to_numeric)
Out[413]:
     0  1  2
0  1.1  2  3
1  1.1  2  3

In [414]: df = pd.DataFrame([["5us", pd.Timedelta("1day")]] * 2, dtype="O")

In [415]: df
Out[415]:
     0                1
0  5us  1 days 00:00:00
1  5us  1 days 00:00:00

In [416]: df.apply(pd.to_timedelta)
Out[416]:
                       0      1
0 0 days 00:00:00.000005 1 days
1 0 days 00:00:00.000005 1 days
```

### 陷阱

在整数类型数据上执行选择操作时，很容易地将数据向上转型到浮动类型。在不引入 nans 的情况下，输入数据的 dtype 被保留。另请参阅[整数 NA 的支持](https://pandas.pydata.org/pandas-docs/stable/user_guide/gotchas.html#gotchas-intna "整数 NA 的支持")。

```python
In [417]: dfi = df3.astype("int32")

In [418]: dfi["E"] = 1

In [419]: dfi
Out[419]:
   A  B    C  E
0  1  0   26  1
1  3  1   86  1
2  0  0   46  1
3  0  1  212  1
4 -1 -1   26  1
5  1  0    7  1
6  0 -1  184  1
7  0  0  206  1

In [420]: dfi.dtypes
Out[420]:
A    int32
B    int32
C    int32
E    int64
dtype: object

In [421]: casted = dfi[dfi > 0]

In [422]: casted
Out[422]:
     A    B    C  E
0  1.0  NaN   26  1
1  3.0  1.0   86  1
2  NaN  NaN   46  1
3  NaN  1.0  212  1
4  NaN  NaN   26  1
5  1.0  NaN    7  1
6  NaN  NaN  184  1
7  NaN  NaN  206  1

In [423]: casted.dtypes
Out[423]:
A    float64
B    float64
C      int32
E      int64
dtype: object
```

float类型保持不变.

```python
In [424]: dfa = df3.copy()

In [425]: dfa["A"] = dfa["A"].astype("float32")

In [426]: dfa.dtypes
Out[426]:
A    float32
B    float64
C    float64
dtype: object

In [427]: casted = dfa[df2 > 0]

In [428]: casted
Out[428]:
          A         B      C
0  1.047606  0.256090   26.0
1  3.497968  1.426469   86.0
2       NaN       NaN   46.0
3       NaN  1.139976  212.0
4       NaN       NaN   26.0
5  1.346426  0.096706    7.0
6       NaN       NaN  184.0
7       NaN       NaN  206.0

In [429]: casted.dtypes
Out[429]:
A    float32
B    float64
C    float64
dtype: object
```

## 基于dtype选择列

select_dtypes() 方法实现基于dtype选择列

首先我们创建一个只有少数dtype列的 DataFrame:

```python
In [430]: df = pd.DataFrame(
    {
        "string": list("abc"),
        "int64": list(range(1, 4)),
        "uint8": np.arange(3, 6).astype("u1"),
        "float64": np.arange(4.0, 7.0),
        "bool1": [True, False, True],
        "bool2": [False, True, False],
        "dates": pd.date_range("now", periods=3),
        "category": pd.Series(list("ABC")).astype("category"),
    }
)


In [431]: df["tdeltas"] = df.dates.diff()

In [432]: df["uint64"] = np.arange(3, 6).astype("u8")

In [433]: df["other_dates"] = pd.date_range("20130101", periods=3)

In [434]: df["tz_aware_dates"] = pd.date_range("20130101", periods=3, tz="US/Eastern")

In [435]: df
Out[435]:
  string  int64  uint8  ...  uint64  other_dates            tz_aware_dates
0      a      1      3  ...       3   2013-01-01 2013-01-01 00:00:00-05:00
1      b      2      4  ...       4   2013-01-02 2013-01-02 00:00:00-05:00
2      c      3      5  ...       5   2013-01-03 2013-01-03 00:00:00-05:00

[3 rows x 12 columns]
```

dtypes如下:

```python
In [436]: df.dtypes
Out[436]:
string                                object
int64                                  int64
uint8                                  uint8
float64                              float64
bool1                                   bool
bool2                                   bool
dates                         datetime64[ns]
category                            category
tdeltas                      timedelta64[ns]
uint64                                uint64
other_dates                   datetime64[ns]
tz_aware_dates    datetime64[ns, US/Eastern]
dtype: object
```

select_dtypes() 有两个参数 include 和 exclude，可以指定包含哪些 dtypes 的列（include）和/或 不包含哪些 dtypes 的列（exclude）。

例如，选择布尔列：

```python
In [437]: df.select_dtypes(include=[bool])
Out[437]:
   bool1  bool2
0   True  False
1  False   True
2   True  False
```

也可以传递 [NumPy dtype 层次结构](https://numpy.org/doc/stable/reference/arrays.scalars.html "NumPy dtype 层次结构")中的 dtype 名称：

```python
In [438]: df.select_dtypes(include=["bool"])
Out[438]:
   bool1  bool2
0   True  False
1  False   True
2   True  False
```

select_dtypes() 也适用于一般 dtypes。

例如，选择所有数字和布尔列，但不包括无符号整数：

```python
In [439]: df.select_dtypes(include=["number", "bool"], exclude=["unsignedinteger"])
Out[439]:
   int64  float64  bool1  bool2 tdeltas
0      1      4.0   True  False     NaT
1      2      5.0  False   True  1 days
2      3      6.0   True  False  1 days
```

选择字符串列必须指定object类型:

```python
In [440]: df.select_dtypes(include=["object"])
Out[440]:
  string
0      a
1      b
2      c
```

要查看诸如 numpy.number 这样的泛型 dtypes 的所有子 dtypes，可以定义一个返回子 dtypes 树的函数：

```python
In [441]: def subdtypes(dtype):
    subs = dtype.__subclasses__()
    if not subs:
        return dtype
    return [dtype, [subdtypes(dt) for dt in subs]]
```

所有NumPy dtypes都是numpy.generic的子类:

```python
In [442]: subdtypes(np.generic)
Out[442]:
[numpy.generic,
 [[numpy.number,
   [[numpy.integer,
     [[numpy.signedinteger,
       [numpy.int8,
        numpy.int16,
        numpy.int32,
        numpy.int64,
        numpy.longlong,
        numpy.timedelta64]],
      [numpy.unsignedinteger,
       [numpy.uint8,
        numpy.uint16,
        numpy.uint32,
        numpy.uint64,
        numpy.ulonglong]]]],
    [numpy.inexact,
     [[numpy.floating,
       [numpy.float16, numpy.float32, numpy.float64, numpy.longdouble]],
      [numpy.complexfloating,
       [numpy.complex64, numpy.complex128, numpy.clongdouble]]]]]],
  [numpy.flexible,
   [[numpy.character, [numpy.bytes_, numpy.str_]],
    [numpy.void, [numpy.record]]]],
  numpy.bool_,
  numpy.datetime64,
  numpy.object_]]
```

> **注意**
>
> pandas 还定义了 category 和 datetime64[ns, tz]类型，这些类型没有集成到正常的 NumPy 层次结构中，因此不会在上述函数中显示。

pandas 还定义了 category 和 datetime64[ns, tz]类型，这些类型没有集成到正常的 NumPy 层次结构中，因此不会在上述函数中显示。

## 总结

这是文章的最后一部分了——非常感谢你能阅读到最后。

希望这篇文章对你有所帮助。

如果你对AI+数据分析感兴趣，可以试试[易从](https://www.openai36.com)这个网站。这个网站，无需懂数学，无需会Python，用聊天的方式就能完成数据分析。把excel（或csv）上传给大模型，大模型根据聊天指令自动完成数据分析。

我在[易从](https://www.openai36.com)上，用AI做了一个[使用随机森林预测泰塔尼克号幸存者](https://www.openai36.com/share/f4b5d7d1-6c38-4898-b1b6-71d8fd8e57f8)，可作为辅助参考。

感谢你阅读本文！[原文地址](https://blog.openai36.com/2024/09/09/pandas%e5%9f%ba%e7%a1%80%e5%8a%9f%e8%83%bd/)
