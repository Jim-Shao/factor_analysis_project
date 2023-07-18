#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@Author  :   shaoshijie
@File    :   factor_calc.py
@Description    :   实现多进程因子的计算（给定表达式，对dataframe进行多进程计算得到因子值的列）
'''

import os
import re
import math
import pandas as pd
import numpy as np

from typing import List, Union, Dict, Tuple
from abc import ABC, abstractmethod
from multiprocessing import Manager, Process, Queue
from multiprocessing.managers import Namespace
from tqdm import tqdm

from factor_analysis.utils import assert_std_index

# ===========================================================
# region 1. Base
# ===========================================================


class Operator(ABC):
    """运算符基类，后续的子类需要实现_apply方法"""
    def __init__(self):
        pass


class UnaryOperator(Operator):
    """一元运算符基类"""
    @abstractmethod
    def _apply(self, x: pd.Series) -> pd.Series:
        pass


class BinaryOperator(Operator):
    """二元运算符基类"""
    @abstractmethod
    def _apply(self, x: pd.Series, y: pd.Series) -> pd.Series:
        pass


class CrossSectionalOperator(Operator):
    """横截面运算符基类"""
    @abstractmethod
    def _apply(self, x: pd.Series) -> pd.Series:
        pass


class TsOperator(Operator):
    """一元时间序列运算符基类"""
    @abstractmethod
    def _apply(self, x: pd.Series, window: int) -> pd.Series:
        pass


class BinaryTsOperator(Operator):
    """二元时间序列运算符基类（两个时间序列的运算）"""
    @abstractmethod
    def _apply(self, x: pd.Series, y: pd.Series, window: int) -> pd.Series:
        pass


# endregion==================================================

# ===========================================================
# region 2. UnaryOperator
# ===========================================================


class Neg(UnaryOperator):
    """取负运算符"""
    def _apply(self, x: pd.Series) -> pd.Series:
        return -x


class Abs(UnaryOperator):
    """取绝对值运算符"""
    def _apply(self, x: pd.Series) -> pd.Series:
        return x.abs()


class Sign(UnaryOperator):
    """取符号运算符"""
    def _apply(self, x: pd.Series) -> pd.Series:
        return x.apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)


class Log(UnaryOperator):
    """取对数运算符"""
    def _apply(self, x: pd.Series) -> pd.Series:
        return x.apply(lambda x: math.log(x) if x > 0 else 0)


# endregion==================================================

# ===========================================================
# region 3. BinaryOperator
# ===========================================================


class Sum(BinaryOperator):
    """求和运算符"""
    def _apply(self, x: pd.Series, y: Union[int, float,
                                            pd.Series]) -> pd.Series:
        return x + y


class Diff(BinaryOperator):
    """求差运算符"""
    def _apply(self, x: pd.Series, y: Union[int, float,
                                            pd.Series]) -> pd.Series:
        return x - y


class Mul(BinaryOperator):
    """求积运算符"""
    def _apply(self, x: pd.Series, y: Union[int, float,
                                            pd.Series]) -> pd.Series:
        return x * y


class Div(BinaryOperator):
    """求商运算符"""
    def _apply(self, x: pd.Series, y: Union[int, float,
                                            pd.Series]) -> pd.Series:
        return x / y


class Power(BinaryOperator):
    """求幂运算符"""
    def _apply(self, x: pd.Series, y: Union[int, float,
                                            pd.Series]) -> pd.Series:
        return x**y


class Cov(BinaryOperator):
    """求协方差运算符"""
    def _apply(self, x: pd.Series, y: pd.Series) -> pd.Series:
        return x.cov(y)


class Corr(BinaryOperator):
    """求相关系数运算符"""
    def _apply(self, x: pd.Series, y: pd.Series) -> pd.Series:
        return x.corr(y)


class Equal(BinaryOperator):
    """判断一个序列与一个数值或者另一个序列是否相等"""
    def _apply(self, x: pd.Series, y: Union[int, float,
                                            pd.Series]) -> pd.Series:
        return x == y


class Greater(BinaryOperator):
    """判断一个序列是否大于一个数值或者另一个序列"""
    def _apply(self, x: pd.Series, y: Union[int, float,
                                            pd.Series]) -> pd.Series:
        return x > y


class Less(BinaryOperator):
    """判断一个序列是否小于一个数值或者另一个序列"""
    def _apply(self, x: pd.Series, y: Union[int, float,
                                            pd.Series]) -> pd.Series:
        return x < y


# endregion==================================================

# ===========================================================
# region 4. CrossSectionalOperator
# ===========================================================


class Rank(CrossSectionalOperator):
    """横截面排序运算符"""
    def _apply(self, x: pd.Series) -> pd.Series:
        return x.groupby(level='datetime').rank()


# endregion==================================================

# ===========================================================
# region 5. TsOperator
# ===========================================================


class TsSum(TsOperator):
    """时序求窗口内和运算符"""
    def _apply(self, x: pd.Series, window: int) -> pd.Series:
        return x.groupby(
            level='order_book_id').rolling(window).sum().droplevel(0)


class TsMean(TsOperator):
    """时序求窗口内均值运算符"""
    def _apply(self, x: pd.Series, window: int) -> pd.Series:
        return x.groupby(
            level='order_book_id').rolling(window).mean().droplevel(0)


class TsStd(TsOperator):
    """时序求窗口内标准差运算符"""
    def _apply(self, x: pd.Series, window: int) -> pd.Series:
        return x.groupby(
            level='order_book_id').rolling(window).std().droplevel(0)


class TsSkew(TsOperator):
    """时序求窗口内偏度运算符"""
    def _apply(self, x: pd.Series, window: int) -> pd.Series:
        return x.groupby(
            level='order_book_id').rolling(window).skew().droplevel(0)


class TsKurt(TsOperator):
    """时序求窗口内峰度运算符"""
    def _apply(self, x: pd.Series, window: int) -> pd.Series:
        return x.groupby(
            level='order_book_id').rolling(window).kurt().droplevel(0)


class TsMedian(TsOperator):
    """时序求窗口内中位数运算符"""
    def _apply(self, x: pd.Series, window: int) -> pd.Series:
        return x.groupby(
            level='order_book_id').rolling(window).median().droplevel(0)


class TsMax(TsOperator):
    """时序求窗口内最大值运算符"""
    def _apply(self, x: pd.Series, window: int) -> pd.Series:
        return x.groupby(
            level='order_book_id').rolling(window).max().droplevel(0)


class TsMin(TsOperator):
    """时序求窗口内最小值运算符"""
    def _apply(self, x: pd.Series, window: int) -> pd.Series:
        return x.groupby(
            level='order_book_id').rolling(window).min().droplevel(0)


class TsArgmax(TsOperator):
    """时序求窗口内最大值索引运算符"""
    def _apply(self, x: pd.Series, window: int) -> pd.Series:
        return x.groupby(level='order_book_id').rolling(window).apply(
            lambda x: x.argmax()).droplevel(0)


class TsArgmin(TsOperator):
    """时序求窗口内最小值索引运算符"""
    def _apply(self, x: pd.Series, window: int) -> pd.Series:
        return x.groupby(level='order_book_id').rolling(window).apply(
            lambda x: x.argmin()).droplevel(0)


class TsRank(TsOperator):
    """时序求排序运算符"""
    def _apply(self, x: pd.Series, window: int) -> pd.Series:
        return x.groupby(level='order_book_id').rolling(window).apply(
            lambda x: x.rank()[-1]).droplevel(0)


class TsDelay(TsOperator):
    """时序求延迟运算符，得到若干天前的数据"""
    def _apply(self, x: pd.Series, window: int) -> pd.Series:
        return x.groupby(level='order_book_id').shift(window)


class TsDelta(TsOperator):
    """时序求差分运算符，减去若干天前的数据"""
    def _apply(self, x: pd.Series, window: int) -> pd.Series:
        return x.groupby(level='order_book_id').diff(window)


# endregion==================================================

# ===========================================================
# region 6. BinaryTsOperator
# ===========================================================


class BinaryTsCorr(BinaryTsOperator):
    """两个时序求窗口内相关系数运算符"""
    def _apply(self, x: pd.Series, y: pd.Series, window: int) -> pd.Series:
        return x.groupby(level='order_book_id').rolling(
            window=window).apply(lambda x: x.corr(y)).droplevel(0)


# endregion==================================================

# ===========================================================
# region 7. Expression
# ===========================================================


class TreeNode:
    def __init__(self, value: str, children: List, parent=None):
        self.value: str = value  # 节点的值
        self.children: List[TreeNode] = children  # 子节点
        self.parent: TreeNode = parent  # 父节点


class Expression:
    def __init__(self, expression: str) -> None:
        self.expression = re.sub(r'\s+', '', expression)  # 去除空格
        self.expression_tree = self.build_expression_tree()  # 构建表达式树
        self._operators = {
            'neg': Neg(),
            'abs': Abs(),
            'sign': Sign(),
            'log': Log(),
            'sum': Sum(),
            'diff': Diff(),
            'mul': Mul(),
            'div': Div(),
            'power': Power(),
            'cov': Cov(),
            'corr': Corr(),
            'equal': Equal(),
            'greater': Greater(),
            'less': Less(),
            'rank': Rank(),
            'ts_sum': TsSum(),
            'ts_mean': TsMean(),
            'ts_std': TsStd(),
            'ts_skew': TsSkew(),
            'ts_kurt': TsKurt(),
            'ts_median': TsMedian(),
            'ts_max': TsMax(),
            'ts_min': TsMin(),
            'ts_argmax': TsArgmax(),
            'ts_argmin': TsArgmin(),
            'ts_rank': TsRank(),
            'ts_delay': TsDelay(),
            'ts_delta': TsDelta(),
            'ts_corr': BinaryTsCorr(),
        }

    def build_expression_tree(self):
        """根据括号的嵌套关系构建表达式树"""
        # 创建一个空的根节点
        curr_node = TreeNode('', [])
        root_node = curr_node
        for char in self.expression:
            # 如果字符是 '('，表示一个子表达式的开始。创建一个新的节点 curr_node，
            # 将其作为当前节点的子节点，并将当前节点更新为新创建的节点。
            if char == '(':
                old_node = curr_node
                curr_node = TreeNode('', [], curr_node)
                old_node.children.append(curr_node)
            # 如果字符是 ')'，表示一个子表达式的结束。将当前节点更新为其父节点，回退到上一层的节点。
            elif char == ')':
                curr_node = curr_node.parent
            # 如果字符是 ','，表示一个子表达式的分隔符。创建一个新的节点 curr_node，
            # 将其作为当前节点的父节点的子节点，并将当前节点更新为新创建的节点。
            elif char == ',':
                parent_node = curr_node.parent
                curr_node = TreeNode('', [], parent_node)
                parent_node.children.append(curr_node)
            # 如果字符是其他值，表示一个子表达式的值。将其添加到当前节点的值中。
            else:
                curr_node.value += char
        # 去除根节点，得到表达式树
        self.expression_tree = root_node.children[0]
        return self.expression_tree

    def traverse_tree(self):
        """递归遍历表达式树"""
        def _traverse_tree(node: TreeNode, indent: str = ''):
            if node:
                print(indent + node.value)
                for child in node.children:
                    _traverse_tree(child, indent + '  ')

        _traverse_tree(self.expression_tree)

    def apply_to_df(self, df: pd.DataFrame, node: TreeNode = None):
        """递归应用表达式树到 DataFrame"""
        assert_std_index(df, 'df', 'df')

        if node is None:
            node: TreeNode = self.expression_tree

        if node.value in self._operators:
            operator: Operator = self._operators[node.value]
        elif node.value in df.columns:
            return df[node.value]
        else:
            raise ValueError(
                f'Unknown value <{node.value}>, not in operators or columns')

        # 递归应用表达式树
        if isinstance(operator, CrossSectionalOperator):
            assert len(node.children) == 1
            return operator._apply(self.apply_to_df(df, node.children[0]))
        elif isinstance(operator, UnaryOperator):
            assert len(node.children) == 1
            return operator._apply(self.apply_to_df(df, node.children[0]))
        elif isinstance(operator, BinaryOperator):
            assert len(node.children) == 2
            return operator._apply(self.apply_to_df(df, node.children[0]),
                                   self.apply_to_df(df, node.children[1]))
        elif isinstance(operator, TsOperator):
            assert len(node.children) == 2
            return operator._apply(self.apply_to_df(df, node.children[0]),
                                   int(node.children[1].value))
        elif isinstance(operator, BinaryTsOperator):
            assert len(node.children) == 3
            return operator._apply(self.apply_to_df(df, node.children[0]),
                                   self.apply_to_df(df, node.children[1]),
                                   int(node.children[2].value))
        else:
            raise ValueError(f'Unknown operator type <{type(operator)}>')


# endregion==================================================

# ===========================================================
# region 8. Multiprocess
# ===========================================================


class CalcPool:
    def __init__(
        self,
        df: pd.DataFrame,
        expressions: Union[List[str], Dict[str, str]],
        n_jobs: int = 1,
    ):
        """
        计算因子的进程池

        Parameters
        ----------
        df : pd.DataFrame
            因子计算所需的数据，multi-index[datetime, order_book_id]
        expressions : Union[List[str], Dict[str, str]]
            因子计算表达式。
            如果为列表，每个元素为因子表达式，因子名称按照顺序为factor_1, factor_2, ...自动生成；
            如果为字典，key为因子名称，value为因子表达式；
        n_jobs : int, optional
            并行计算的进程数, by default 1
        """
        # 将df设置为共享内存储存在name space中
        self.manager = Manager()
        self.name_space = self.manager.Namespace(df=df)

        # 检查
        assert isinstance(expressions,
                          (list, dict)), 'expressions must be list or dict'
        if isinstance(expressions, list):
            expressions = {
                f'factor_{i}': expr
                for i, expr in enumerate(expressions, 1)
            }
        self.expressions = expressions
        self.factor_names = list(expressions.keys())

        self.n_jobs = n_jobs

        # 将factors设置为共享内存储存在name space中
        self.index: List[Tuple] = self.manager.list()
        self.index.extend(df.index.to_list())  # 储存索引
        self.index_len = len(df.index)
        # 创建字典，key为因子名称，value为共享内存中的因子数据
        self.factors: Dict[str, np.ndarray] = self.manager.dict()
        for factor_name in self.factor_names:
            self.factors[factor_name] = None

    def _merge_factors(self):
        # 将共享内存中的index从ListProxy转换为list
        index = list(self.index)
        # 将共享内存中的factors从DictProxy转换为dict
        factors = dict(self.factors)

        # 将共享内存中的index和factors取出，重新组合成DataFrame
        index = pd.MultiIndex.from_tuples(index,
                                          names=['datetime', 'order_book_id'])
        factors = pd.DataFrame(factors, index=index)
        return factors

    # 传递name space才能确保共享的对象能够序列化，通过name_space.df传递df
    def _calc_factor(
        self,
        name_space: Namespace,
        factor_name: str,
        expression: str,
        queue: Queue = None,
    ) -> None:
        """计算单个因子

        Parameters
        ----------
        name_space : Namespace
            命名空间，包含df
        factor_name : str
            因子名称
        expression : str
            因子表达式
        queue : Queue, optional
            进程队列，用于记录进程id, by default None
        """
        expr = Expression(expression)
        factor = expr.apply_to_df(name_space.df).values
        if not len(factor) == self.index_len:
            raise ValueError(
                f'factor length {len(factor)} != index length {self.index_len}'
            )
        self.factors[factor_name] = factor
        if queue is not None:
            queue.put(os.getpid())

    def calc_factors(self) -> pd.DataFrame:
        """计算因子（单进程或多进程）"""
        if self.n_jobs == 1:
            for factor_name, expression in tqdm(
                    self.expressions.items(), desc='>>> Calculating factors'):
                self._calc_factor(self.name_space, factor_name, expression)
        else:
            processes: List[Process] = []  # 进程列表，用于存储正在运行的进程
            queue = Queue()  # 进程队列，用于存储已完成的进程

            tqdm_obj = tqdm(total=len(self.expressions),
                            desc='>>> Calculating factors')

            # 遍历 expressions创建进程
            for factor_name, expression in self.expressions.items():
                p = Process(target=self._calc_factor,
                            args=(self.name_space, factor_name, expression,
                                  queue))

                # 当进程数达到最大值时，等待其中一个进程结束并移除
                if len(processes) == self.n_jobs:
                    completed_process_pid = queue.get()  # 等待一个进程基本结束
                    for process in processes:
                        if process.pid == completed_process_pid:
                            process.join()  # 等待进程彻底结束
                            processes.remove(process)  # 移除已完成的进程
                            tqdm_obj.update()

                # 在进程数未达到最大值时，直接加入新进程
                processes.append(p)
                p.start()

            # 等待剩余的进程结束
            for process in processes:
                process.join()
                tqdm_obj.update()

            tqdm_obj.close()
        return self._merge_factors()


# endregion==================================================

if __name__ == '__main__':
    from factor_analysis.utils import generate_stock_df
    from time import time

    df = generate_stock_df()
    expression1 = '(open)'
    expression2 = '(high)'
    expression3 = '(low)'
    expression4 = '(ts_corr(rank(ts_delta(log(volume), 2)), rank(div(diff(close, open), open)), 6))'
    expression5 = '(rank(ts_corr(rank(ts_delta(log(volume), 2)), rank(div(diff(close, open), open)), 6)))'
    expression6 = '(neg(rank(ts_corr(rank(ts_delta(log(volume), 2)), rank(div(diff(close, open), open)), 6))))'
    expressions = [
        expression1, expression2, expression3, expression4, expression5,
        expression6
    ] * 2

    # start = time()
    # n_jobs = 1
    # pool = CalcPool(df, expressions, n_jobs=n_jobs)
    # factors = pool.calc_factors()
    # end = time()
    # print(factors)
    # print(f'不使用多进程逐个计算耗时：{end - start}s')

    start = time()
    n_jobs = 6
    pool = CalcPool(df, expressions, n_jobs=n_jobs)
    factors = pool.calc_factors()
    end = time()
    print(factors)
    print(f'使用多进程计算耗时：{end - start}s')
