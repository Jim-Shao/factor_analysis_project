#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@Author  :   shaoshijie
@File    :   utils.py
@Description    :   工具函数
'''

import pandas as pd
import numpy as np

from typing import Tuple, Union, List

# ===========================================================
# region 1. 随机生成数据的函数
# ===========================================================


def generate_stock_df(
    n_stocks: int = 10,
    n_days: int = 252,
    start_date: str = '2010-01-04',
    seed: int = 0,
) -> pd.DataFrame:
    """
    随机生成股票数据，multi-index[datetime, order_book_id]，
    列为开盘价、最高价、最低价、收盘价、成交量

    Parameters
    ----------
    n_stocks : int, optional
        股票数量, by default 10
    n_days : int, optional
        交易日数量, by default 252
    start_date : str, optional
        起始日期, by default '2010-01-04'
    seed : int, optional
        随机种子, by default 0

    Returns
    -------
    pd.DataFrame
        股票数据
    """
    np.random.seed(seed)
    dates = pd.date_range(start_date, periods=n_days, freq='B')

    returns = np.random.uniform(low=-0.1, high=0.1, size=(n_days, n_stocks))
    closes = pd.DataFrame((1 + returns).cumprod(axis=0) * 100)

    highs = closes + np.random.uniform(low=0, high=10, size=(n_days, n_stocks))
    lows = closes - np.random.uniform(low=0, high=10, size=(n_days, n_stocks))
    opens = lows + np.random.uniform(low=-3, high=3, size=(n_days, n_stocks))
    volumes = pd.DataFrame(
        np.random.uniform(low=1e5, high=5e5, size=(n_days, n_stocks)))

    order_book_ids = [([f'{i:06d}.XSHE'] * n_days) for i in range(n_stocks)]
    order_book_ids = np.concatenate(order_book_ids)

    data = pd.DataFrame({
        'datetime': np.concatenate([dates] * n_stocks),
        'order_book_id': order_book_ids,
        'open': pd.melt(opens).value,
        'high': pd.melt(highs).value,
        'low': pd.melt(lows).value,
        'close': pd.melt(closes).value,
        'volume': pd.melt(volumes).value,
    })
    data = data.set_index(['datetime', 'order_book_id'])
    data = data.sort_index()
    return data


def generate_factor_df(
    n_stocks: int = 10,
    n_factors: int = 3,
    n_days: int = 252,
    start_date: str = '2010-01-04',
    seed: int = 0,
) -> pd.DataFrame:
    """
    随机生成因子数据，multi-index[datetime, order_book_id]，列为不同因子的因子值

    Parameters
    ----------
    n_stocks : int, optional
        股票数量, by default 10
    n_factors : int, optional
        因子数量, by default 3
    n_days : int, optional
        交易日数量, by default 252
    start_date : str, optional
        起始日期, by default '2010-01-04'
    seed : int, optional
        随机种子, by default 0

    Returns
    -------
    pd.DataFrame
        因子数据
    """
    np.random.seed(seed)
    dates = pd.date_range(start_date, periods=n_days, freq='B')

    factor_values = np.random.uniform(low=-5,
                                      high=5,
                                      size=(n_days, n_stocks, n_factors))
    factor_values = factor_values / np.sqrt(n_factors)
    order_book_ids = [([f'{i:06d}.XSHE'] * n_days) for i in range(n_stocks)]
    order_book_ids = np.concatenate(order_book_ids)
    data = pd.DataFrame({
        'datetime': np.concatenate([dates] * n_stocks),
        'order_book_id': order_book_ids,
    })
    for i in range(n_factors):
        data[f'factor_{i}'] = factor_values[:, :, i].reshape(-1)
    data = data.set_index(['datetime', 'order_book_id'])
    data = data.sort_index()
    return data


def generate_industry_df(
    n_stocks: int = 10,
    n_days: int = 252,
    n_industries: int = 3,
    start_date: str = '2010-01-04',
    seed: int = 0,
) -> pd.DataFrame:
    """
    随机生成行业数据，multi-index[datetime, order_book_id]，列为不同行业的哑变量

    Parameters
    ----------
    n_stocks : int, optional
        股票数量, by default 10
    n_days : int, optional
        交易日数量, by default 252
    n_industries : int, optional
        行业数量, by default 3
    start_date : str, optional
        起始日期, by default '2010-01-04'
    seed : int, optional
        随机种子, by default 0

    Returns
    -------
    pd.DataFrame
        行业数据
    """
    np.random.seed(seed)
    dates = pd.date_range(start_date, periods=n_days, freq='B')
    order_book_ids = [f'{i:06d}.XSHE' for i in range(n_stocks)]
    industry_list = [f'industry_{i}' for i in range(n_industries)]
    data = pd.DataFrame({
        'order_book_id':
        order_book_ids,
        'industry':
        np.random.choice(industry_list, size=n_stocks),
    })
    data['is_in'] = 1
    data = data.pivot(index='order_book_id',
                      columns='industry',
                      values='is_in')
    data = data.fillna(0)
    data = data.astype(int)
    data = pd.concat([data] * n_days, keys=dates)
    data.index.names = ['datetime', 'order_book_id']
    return data


def generate_cap_df(
    n_stocks: int = 10,
    n_days: int = 252,
    start_date: str = '2010-01-04',
    seed: int = 0,
) -> pd.DataFrame:
    """
    随机生成股票市值数据，multi-index[datetime, order_book_id]，列为股票市值

    Parameters
    ----------
    n_stocks : int, optional
        股票数量, by default 10
    n_days : int, optional
        交易日数量, by default 252
    start_date : str, optional
        起始日期, by default '2010-01-04'
    seed : int, optional
        随机种子, by default 0

    Returns
    -------
    pd.DataFrame
        股票市值数据
    """
    np.random.seed(seed)
    dates = pd.date_range(start_date, periods=n_days, freq='B')
    order_book_ids = [([f'{i:06d}.XSHE'] * n_days) for i in range(n_stocks)]
    order_book_ids = np.concatenate(order_book_ids)
    data = pd.DataFrame({
        'datetime':
        np.concatenate([dates] * n_stocks),
        'order_book_id':
        order_book_ids,
        'cap':
        np.random.uniform(low=1e8, high=1e9, size=n_stocks * n_days),
    })
    data = data.set_index(['datetime', 'order_book_id'])
    data = data.sort_index()
    return data


# endregion

# ===========================================================
# region 2. 断言检查的函数
# ===========================================================


def assert_index_equal(
    source: Union[pd.Series, pd.DataFrame],
    target: Union[pd.Series, pd.DataFrame],
    source_name: str,
    target_name: str,
) -> None:
    """
    断言两个序列的长度和索引完全相同

    Parameters
    ----------
    source : Union[pd.Series, pd.DataFrame]
        源数据
    target : Union[pd.Series, pd.DataFrame]
        目标数据
    source_name : str
        源数据名称
    target_name : str
        目标数据名称
    """
    if len(source) != len(target):
        raise ValueError(
            f"{source_name} and {target_name} must have the same length")
    if not source.index.equals(target.index):
        raise ValueError(
            f"{source_name} and {target_name} must have the totally same index"
        )


def assert_std_index(
    source: Union[pd.Series, pd.DataFrame],
    name: str,
    want_type: str = 'df',
) -> None:
    """
    断言一个Series或DataFrame的索引为标准多重索引:
    ['datetime'(level=0, pd.DatetimeIndex), 'order_book_id'(level=1)]

    Parameters
    ----------
    source : Union[pd.Series, pd.DataFrame]
        源数据
    name : str
        源数据名称
    want_type : str, optional
        源数据类型, 可选值为'df'或'series', by default 'df'
    """
    if want_type == 'df':
        assert isinstance(source, pd.DataFrame), f"{name} is not pd.DataFrame"
    elif want_type == 'series':
        assert isinstance(source, pd.Series), f"{name} is not pd.Series"
    else:
        raise ValueError(f"want_type={want_type} must be 'df' or 'series'")

    std_index = ['datetime', 'order_book_id']
    if not source.index.names == std_index:
        raise ValueError(f"{name} must be multi-indexed with {std_index}")

    datetime_index = source.index.get_level_values('datetime')
    if not isinstance(datetime_index, pd.DatetimeIndex):
        raise ValueError(f"{name} datetime index must be pd.DatetimeIndex")


# endregion

# ===========================================================
# region 3. 计算函数
# ===========================================================


def calc_return(
    bar_df: pd.DataFrame,
    forward_periods: Union[int, List[int]] = 0,
) -> pd.DataFrame:
    """ 计算未来多期的收益率

    Parameters
    ----------
    bar_df : pd.DataFrame
        后复权的K线数据，multi-index[datetime, order_book_id]，columns=['open', 'high', 'low', 'close', 'volume', ...]
    forward_periods : Union[int, List[int]], optional
        未来的期数，by default 0

    Returns
    -------
    pd.DataFrame
        未来多期的收益率，multi-index[datetime, order_book_id]，columns=[f'{forward_period}D', ...]
    """
    # 检查输入数据
    assert_std_index(bar_df, 'bar_df', 'df')
    target_cols = ['open', 'high', 'low', 'close', 'volume']
    assert set(target_cols).issubset(
        bar_df.columns
    ), 'bar_df must contain open, high, low, close, volume columns'

    # 计算未来多期的收益率
    close_df = bar_df['close'].unstack()
    if isinstance(forward_periods, int):
        forward_periods = [forward_periods]
    return_df = pd.concat(
        [
            close_df.pct_change(periods=periods, fill_method=None).shift(
                periods=-periods).stack(dropna=False)
            for periods in forward_periods
        ],
        axis=1,
        keys=[f'{forward_period}D' for forward_period in forward_periods],
    )
    return return_df


def cross_sectional_group_cut(
    x: pd.Series,
    n_groups: int,
) -> pd.Series:
    """
    生成横截面分组的分组标签（1代表最小值，n_groups代表最大值）

    Parameters
    ----------
    x : pd.Series
        待分组的数据
    n_groups : int
        分组数

    Returns
    -------
    group_cuts : pd.Series
        分组标签
    """
    assert_std_index(x, 'x', 'series')
    assert n_groups > 0, "n_groups must be positive"

    # 先进行排序，以避免bin无法正常分割的问题：bin edges must be unique
    group_cuts = x.groupby(level='datetime').apply(lambda x: pd.qcut(
        x.rank(method='first'), q=n_groups, labels=range(1, n_groups + 1)))
    group_cuts.name = 'group'
    return group_cuts


def calc_ic(
    factor: pd.Series,
    forward_return: pd.Series,
) -> Tuple[pd.Series, pd.Series]:
    """
    计算因子的IC和RankIC

    Parameters
    ----------
    factor : pd.Series
        因子数据
    forward_return : pd.Series
        未来某期的收益率

    Returns
    -------
    ic : pd.Series
        Information Coefficient
    rank_ic : pd.Series
        Rank Information Coefficient
    """
    assert_std_index(factor, 'factor', 'series')
    assert_std_index(forward_return, 'forward_return', 'series')
    assert_index_equal(factor, forward_return, 'factor', 'forward_return')

    merge_df = pd.concat([factor, forward_return], axis=1)
    groupby = merge_df.groupby(level='datetime')
    ic = groupby.apply(lambda x: x.corr().iloc[0, 1])
    rank_ic = groupby.apply(lambda x: x.corr(method='spearman').iloc[0, 1])
    return ic, rank_ic


def calc_group_returns(
    forward_returns: pd.Series,
    bench_forward_returns: pd.Series,
    group_cuts: pd.Series,
) -> pd.DataFrame:
    """
    计算分组收益率

    Parameters
    ----------
    forward_returns : pd.Series
        未来某期的收益率
    bench_forward_returns : pd.Series
        基准未来某期的收益率
    group_cuts : pd.Series
        分组标签

    Returns
    -------
    group_return_df : pd.DataFrame, shape = (n_dates, n_groups+4)
        不同分组的收益率
        (index: datetime, columns: group1, group2, ..., groupn, benchmark, long-excess, short-excess, long-short)
    """
    assert_std_index(forward_returns, 'forward_returns', 'series')
    assert_std_index(bench_forward_returns, 'bench_forward_returns', 'series')
    assert_std_index(group_cuts, 'group_cuts', 'series')

    assert_index_equal(forward_returns, group_cuts, 'forward_returns',
                       'group_cuts')
    assert bench_forward_returns.index.get_level_values('datetime').equals(
        group_cuts.index.get_level_values('datetime').unique()
    ), "bench_forward_returns and group_cuts should have the same datetime index"

    # 计算不同组的等权平均收益率
    group_return_df = forward_returns.groupby(['datetime', group_cuts]).mean()
    group_return_df = group_return_df.unstack()
    group_return_df.columns = [f'group{i}' for i in group_return_df.columns]
    groups = group_return_df.columns

    # 计算benchmark、long-excess、short-excess、long-short
    long_group = groups[-1]
    short_group = groups[0]
    group_return_df['benchmark'] = bench_forward_returns.groupby(
        'datetime').mean()
    group_return_df['long_excess'] = group_return_df[
        long_group] - group_return_df['benchmark']
    group_return_df['short_excess'] = group_return_df[
        'benchmark'] - group_return_df[short_group]
    group_return_df['long_short'] = group_return_df[
        long_group] - group_return_df[short_group]
    return group_return_df


# endregion

if __name__ == '__main__':
    stock_df = generate_stock_df()
    assert_std_index(stock_df, 'stock_df')
    print(stock_df)

    factor_df = generate_factor_df()
    assert_std_index(factor_df, 'factor_df')
    print(factor_df)

    industry_df = generate_industry_df()
    print(industry_df)

    cap_df = generate_cap_df()
    print(cap_df)

    return_df = calc_return(stock_df, forward_periods=[1, 5, 10])
    print(return_df)