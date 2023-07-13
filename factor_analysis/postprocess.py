#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@Author  :   shaoshijie
@File    :   postprocess.py
@Description    :   去极值、标准化、归一化、中性化等
参考https://zhuanlan.zhihu.com/p/37190373
参考https://zhuanlan.zhihu.com/p/624094435
'''

import pandas as pd
import numpy as np

from numpy.linalg import inv
from typing import Union, Callable, List
from functools import partial

from factor_analysis.utils import assert_std_index, assert_index_equal


def winsorize_series_mad(
    series: pd.Series,
    n: float = 3,
) -> pd.Series:
    """去极值，使用中位数绝对偏差（MAD）。

    Parameters
    ----------
    series : pd.Series (multi-indexed by ['datetime', 'order_book_id'])
        需要去极值的序列。
    n : float, optional
        去极值的倍数, by default 3.

    Returns
    -------
    pd.Series
        去极值后的序列。
    """
    median = series.median()
    mad = (series - median).abs().median()
    top = median + n * mad
    bottom = median - n * mad
    series[series > top] = top
    series[series < bottom] = bottom
    return series


def winsorize_series_std(
    series: pd.Series,
    n: float = 3,
) -> pd.Series:
    """去极值，使用标准差。

    Parameters
    ----------
    series : pd.Series (multi-indexed by ['datetime', 'order_book_id'])
        需要去极值的序列。
    n : float, optional
        去极值的倍数, by default 3.

    Returns
    -------
    pd.Series
        去极值后的序列。
    """
    mean = series.mean()
    std = series.std()
    top = mean + n * std
    bottom = mean - n * std
    series[series > top] = top
    series[series < bottom] = bottom
    return series


def winsorize_series_pct(
    series: pd.Series,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.Series:
    """去极值，使用百分位数。

    Parameters
    ----------
    series : pd.Series (multi-indexed by ['datetime', 'order_book_id'])
        需要去极值的序列。
    lower : float, optional
        下界, by default 0.01.
    upper : float, optional
        上界, by default 0.99.

    Returns
    -------
    pd.Series
        去极值后的序列。
    """
    series = series.rank(pct=True)
    series[series < lower] = lower
    series[series > upper] = upper
    return series


def winsorize_series(
    series: pd.Series,
    method: str = 'mad',
    n: float = 3,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.Series:
    """去极值。

    Parameters
    ----------
    series : pd.Series (multi-indexed by ['datetime', 'order_book_id'])
        需要去极值的序列。
    method : str, optional
        使用的去极值方法, 可选值为'mad'、'std'、'pct', by default 'mad'.
    n : float, optional
        去极值的倍数, by default 3.
    lower : float, optional
        下界, by default 0.01.
    upper : float, optional
        上界, by default 0.99.

    Returns
    -------
    pd.Series
        去极值后的序列。
    """
    assert_std_index(series, 'series', 'series')
    if method == 'mad':
        series = winsorize_series_mad(series, n)
    elif method == 'std':
        series = winsorize_series_std(series, n)
    elif method == 'pct':
        series = winsorize_series_pct(series, lower, upper)
    else:
        raise ValueError(f'Unknown method: {method}')
    return series


def winsorize_df(
    df: pd.DataFrame,
    method: str = 'mad',
    n: float = 3,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    """去极值。

    Parameters
    ----------
    df : pd.DataFrame (multi-indexed by ['datetime', 'order_book_id'])
        需要去极值的表格。
    method : str, optional
        使用的去极值方法, 可选值为'mad'、'std'、'pct', by default 'mad'.
    n : float, optional
        去极值的倍数, by default 3.
    lower : float, optional
        下界, by default 0.01.
    upper : float, optional
        上界, by default 0.99.

    Returns
    -------
    pd.DataFrame
        去极值后的表格。
    """
    assert_std_index(df, 'df', 'df')
    for col in df.columns:
        df[col] = winsorize_series(df[col], method, n, lower, upper)
    return df


def standardize_series(series: pd.Series) -> pd.Series:
    """标准化一个序列。

    Parameters
    ----------
    series : pd.Series (multi-indexed by ['datetime', 'order_book_id'])
        需要标准化的序列。

    Returns
    -------
    pd.Series
        标准化后的序列。
    """
    assert_std_index(series, 'series', 'series')
    return (series - series.groupby(level='datetime').mean()) / series.groupby(
        level='datetime').std()


def standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    """标准化一个表格。

    Parameters
    ----------
    df : pd.DataFrame (multi-indexed by ['datetime', 'order_book_id'])
        需要标准化的表格。

    Returns
    -------
    pd.DataFrame
        标准化后的表格。
    """
    assert_std_index(df, 'df', 'df')
    for col in df.columns:
        df[col] = standardize_series(df[col])
    return df


def neutralize(
    y: Union[pd.Series, pd.DataFrame],
    x: Union[pd.Series, pd.DataFrame],
) -> Union[pd.Series, pd.DataFrame]:
    """Neutralize

    Parameters
    ----------
    y : Union[pd.Series, pd.DataFrame]
        需要被中性化的序列或表格。
    x : Union[pd.Series, pd.DataFrame]
        中性化的依据因子。

    Returns
    -------
    Union[pd.Series, pd.DataFrame]
        中性化后的残差
    """
    assert_index_equal(y, x, 'y', 'x')
    y = y.values.reshape(-1, 1) if y.ndim == 1 else y.values  # 转化为二维数组
    x = x.values.reshape(-1, 1) if x.ndim == 1 else x.values  # 转化为二维数组
    for col in range(y.shape[1]):
        beta = inv(x.T @ x) @ x.T @ y[:, col]
        y[:, col] = y[:, col] - x @ beta
    return y


class Postprocess:
    @staticmethod
    def standardize(
        factor: Union[pd.Series, pd.DataFrame],
    ) -> Union[pd.Series, pd.DataFrame]:
        """标准化。（减去截面均值，除以截面标准差）

        Parameters
        ----------
        factor : Union[pd.Series, pd.DataFrame]
            需要标准化的序列或表格。multi-indexed by ['datetime', 'order_book_id']

        Returns
        -------
        Union[pd.Series, pd.DataFrame]
            标准化后的序列或表格。multi-indexed by ['datetime', 'order_book_id']
        """
        if isinstance(factor, pd.Series):
            return standardize_series(factor)
        elif isinstance(factor, pd.DataFrame):
            return standardize_df(factor)
        else:
            raise ValueError('factor must be a series or dataframe.')

    @staticmethod
    def normalize(
        factor: Union[pd.Series, pd.DataFrame],
    ) -> Union[pd.Series, pd.DataFrame]:
        """归一化。（减去截面最小值，除以截面最大值减去最小值）

        Parameters
        ----------
        factor : Union[pd.Series, pd.DataFrame]
            需要归一化的序列或表格。multi-indexed by ['datetime', 'order_book_id']

        Returns
        -------
        Union[pd.Series, pd.DataFrame]
            归一化后的序列或表格。multi-indexed by ['datetime', 'order_book_id']
        """
        if isinstance(factor, pd.Series):
            assert_std_index(factor, 'source', 'series')
        elif isinstance(factor, pd.DataFrame):
            assert_std_index(factor, 'source', 'df')
        else:
            raise ValueError('source must be a series or dataframe.')
        daily_min = factor.groupby(level='datetime').min()
        daily_max = factor.groupby(level='datetime').max()
        return (factor - daily_min) / (daily_max - daily_min)

    @staticmethod
    def winsorize(
        factor: Union[pd.Series, pd.DataFrame],
        method: str = 'mad',
        n: float = 3,
        lower: float = 0.01,
        upper: float = 0.99,
    ) -> Union[pd.Series, pd.DataFrame]:
        """去极值。

        Parameters
        ----------
        factor : Union[pd.Series, pd.DataFrame]
            需要去极值的序列或表格。multi-indexed by ['datetime', 'order_book_id']
        method : str, optional
            去极值的方法, by default 'mad'
            - 'mad': 使用中位数绝对偏差去极值
            - 'std': 使用标准差去极值
            - 'pct': 使用百分位数去极值
        n : float, optional, used when method is 'mad' or 'std'
            中位数绝对偏差或标准差的倍数，当method为'mad'或'std'时使用, by default 3.
        lower : float, optional, used when method is 'pct'
            最低百分位数，当method为'pct'时使用, by default 0.01
        upper : float, optional, used when method is 'pct'
            最高百分位数，当method为'pct'时使用, by default 0.99

        Returns
        -------
        Union[pd.Series, pd.DataFrame]
            去极值后的序列或表格。multi-indexed by ['datetime', 'order_book_id']
        """
        if isinstance(factor, pd.Series):
            return winsorize_series(factor, method, n, lower, upper)
        elif isinstance(factor, pd.DataFrame):
            return winsorize_df(factor, method, n, lower, upper)
        else:
            raise ValueError('factor must be a series or dataframe.')

    @staticmethod
    def ind_neutralize(
        factor: Union[pd.Series, pd.DataFrame],
        industry: pd.DataFrame,
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        市值中性化。

        Parameters
        ----------
        factor : Union[pd.Series, pd.DataFrame]
            需要中性化的序列或表格。multi-indexed by ['datetime', 'order_book_id']
        industry : pd.DataFrame
            行业分类。multi-indexed by ['datetime', 'order_book_id']

        Returns
        -------
        Union[pd.Series, pd.DataFrame]
            中性化后的序列或表格。multi-indexed by ['datetime', 'order_book_id']
        """
        assert_index_equal(factor, industry, 'factor', 'industry')
        if isinstance(factor, pd.Series):
            factor_is_series = True
            factor = pd.DataFrame(factor)
        else:
            factor_is_series = False
        factor_cols = factor.columns
        industry_cols = industry.columns

        # 市值中性化
        factor = pd.concat([factor, industry], axis=1)
        residual = factor.groupby(level='datetime').apply(
            lambda x: neutralize(x[factor_cols], x[industry_cols])).to_dict()

        # 使用多重索引
        residual_df_list = [
            pd.DataFrame(residual[date],
                         index=pd.MultiIndex.from_product(
                             [[date], factor.loc[date].index],
                             names=['datetime', 'order_book_id']))
            for date in residual.keys()
        ]
        factor = pd.concat(residual_df_list)
        factor.columns = factor_cols

        # 如果是series，转化为pd.Series
        if factor_is_series:
            factor = factor.iloc[:, 0]
        return factor

    @staticmethod
    def cap_neutralize(
        factor: Union[pd.Series, pd.DataFrame],
        cap: pd.Series,
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Cap neutralize a series.

        Parameters
        ----------
        factor : Union[pd.Series, pd.DataFrame]
            需要中性化的序列或表格，multi-indexed by ['datetime', 'order_book_id']。
            pd.Series对应单因子，pd.DataFrame对应多因子同时中性化。
        cap : pd.Series
            市值序列，multi-indexed by ['datetime', 'order_book_id']。

        Returns
        -------
        Union[pd.Series, pd.DataFrame]
            中性化后的序列或表格，multi-indexed by ['datetime', 'order_book_id']。
        """
        assert_index_equal(factor, cap, 'factor', 'cap')
        if isinstance(factor, pd.Series):
            factor_is_series = True
            factor = pd.DataFrame(factor)
        else:
            factor_is_series = False
        factor_cols = factor.columns

        factor['_cap'] = np.log(cap)  # 将市值取对数再中性化
        residual = factor.groupby(level='datetime').apply(
            lambda x: neutralize(x[factor_cols], x['_cap'])).to_dict()

        # 使用多重索引
        residual_df_list = [
            pd.DataFrame(residual[date],
                         index=pd.MultiIndex.from_product(
                             [[date], factor.loc[date].index],
                             names=['datetime', 'order_book_id']))
            for date in residual.keys()
        ]
        factor = pd.concat(residual_df_list)
        factor.columns = factor_cols

        # 如果是series，转化为pd.Series
        if factor_is_series:
            factor = factor.iloc[:, 0]
        return factor

    @staticmethod
    def ind_cap_neutralize(
        factor: Union[pd.Series, pd.DataFrame],
        industry: pd.DataFrame,
        cap: pd.Series,
    ) -> pd.Series:
        """
        行业市值中性化。

        Parameters
        ----------
        factor : Union[pd.Series, pd.DataFrame]
            需要中性化的序列或表格，multi-indexed by ['datetime', 'order_book_id']。
            pd.Series对应单因子，pd.DataFrame对应多因子同时中性化。
        industry : pd.DataFrame
            行业分类，multi-indexed by ['datetime', 'order_book_id']。
        cap : pd.Series
            市值序列，multi-indexed by ['datetime', 'order_book_id']。

        Returns
        -------
        Union[pd.Series, pd.DataFrame]
            中性化后的序列或表格，multi-indexed by ['datetime', 'order_book_id']。
        """
        assert_index_equal(factor, industry, 'factor', 'industry')
        assert_index_equal(factor, cap, 'factor', 'cap')
        if isinstance(factor, pd.Series):
            factor_is_series = True
            factor = pd.DataFrame(factor)
        else:
            factor_is_series = False
        factor_cols = factor.columns

        cap = np.log(cap)  # 将市值取对数再中性化
        cap_ind = pd.concat([cap, industry], axis=1)  # 合并市值和行业分类
        cap_ind_cols = cap_ind.columns

        # 将市值和行业分类合并到因子表格中，再进行中性化
        factor = pd.concat([factor, cap_ind], axis=1)
        residual = factor.groupby(level='datetime').apply(
            lambda x: neutralize(x[factor_cols], x[cap_ind_cols])).to_dict()

        # 使用多重索引
        residual_df_list = [
            pd.DataFrame(residual[date],
                         index=pd.MultiIndex.from_product(
                             [[date], factor.loc[date].index],
                             names=['datetime', 'order_book_id']))
            for date in residual.keys()
        ]
        factor = pd.concat(residual_df_list)
        factor.columns = factor_cols

        # 如果是series，转化为pd.Series
        if factor_is_series:
            factor = factor.iloc[:, 0]
        return factor


class PostprocessQueue:
    def __init__(self):
        self.queue: List[partial] = []

    def add_step(self, postprocess_func: Callable, **kwargs) -> None:
        """
        添加后处理步骤。

        Parameters
        ----------
        postprocess_func : Callable
            后处理函数。
        kwargs : dict
            后处理函数的参数。
        """
        self.queue.append(partial(postprocess_func, **kwargs))

    def __call__(
        self,
        factor: Union[pd.Series, pd.DataFrame],
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        对因子进行一系列后处理操作。

        Parameters
        ----------
        factor : Union[pd.Series, pd.DataFrame]
            需要后处理的因子，multi-indexed by ['datetime', 'order_book_id']。
            pd.Series对应单因子，pd.DataFrame对应多因子同时后处理。

        Returns
        -------
        Union[pd.Series, pd.DataFrame]
            后处理后的因子，multi-indexed by ['datetime', 'order_book_id']。
        """
        for func in self.queue:
            factor = func(factor)
        return factor


if __name__ == '__main__':
    from factor_analysis.utils import (generate_stock_df, generate_factor_df,
                                       generate_industry_df, generate_cap_df)

    stock_df = generate_stock_df()
    factor_df = generate_factor_df()
    industry_df = generate_industry_df()
    cap_df = generate_cap_df()

    factor = Postprocess.standardize(factor_df)
    factor = Postprocess.normalize(factor_df)
    factor = Postprocess.winsorize(factor)
    factor = Postprocess.ind_neutralize(factor, industry_df)
    factor = Postprocess.cap_neutralize(factor, cap_df)
    factor = Postprocess.ind_cap_neutralize(factor, industry_df, cap_df)
    print()

    postprocess_queue = PostprocessQueue()
    postprocess_queue.add_step(Postprocess.winsorize)
    postprocess_queue.add_step(Postprocess.ind_cap_neutralize,
                               industry=industry_df,
                               cap=cap_df)
    postprocess_queue.add_step(Postprocess.standardize)
    factor = postprocess_queue(factor_df)
    print(factor)