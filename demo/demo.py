#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@Author  :   shaoshijie
@File    :   demo.py
@Description    :   用于展示如何使用factor_analysis
'''

from factor_analysis.backtest import FactorBacktest
from factor_analysis.postprocess import Postprocess, PostprocessQueue

if __name__ == '__main__':
    import pandas as pd
    from factor_analysis.utils import generate_stock_df, generate_industry_df, generate_cap_df

    # 随机生成测试数据
    stock_bar_df = generate_stock_df(n_stocks=10, n_days=800)  # 行情数据
    benchmark_df = generate_stock_df(n_stocks=1, n_days=800)  # 基准数据，比如沪深300
    industry_df = generate_industry_df(n_stocks=10, n_days=800)  # 行业数据
    cap_df = generate_cap_df(n_stocks=10, n_days=800)  # 市值数据

    # 提取每月最后一个交易日作为调仓日期
    trade_dates = stock_bar_df.index.get_level_values('datetime').unique()
    trade_dates = pd.DataFrame(trade_dates, columns=['datetime'])
    trade_dates['year'] = trade_dates['datetime'].dt.year
    trade_dates['month'] = trade_dates['datetime'].dt.month
    position_dates = trade_dates.groupby(['year', 'month']).last()
    position_dates = position_dates['datetime'].tolist()

    # 需要计算的因子（所有的算子需要以括号开头，以括号结尾，例：(open)）
    expression1 = '(ts_corr(rank(ts_delta(log(volume), 2)), rank(div(diff(close, open), open)), 6))'
    expression2 = '(ts_corr(rank(ts_delta(log(volume), 4)), rank(div(diff(close, open), open)), 10))'
    expression3 = '(rank(ts_corr(rank(ts_delta(log(volume), 2)), rank(div(diff(close, open), open)), 6)))'
    expression4 = '(neg(rank(ts_corr(rank(ts_delta(log(volume), 4)), rank(div(diff(close, open), open)), 10))))'
    expressions = [expression1, expression2, expression3, expression4]

    # 因子计算完成之后的处理（一般的后处理步骤为去极值、行业市值中性化、标准化）
    postprocess_queue = PostprocessQueue()
    postprocess_queue.add_step(Postprocess.winsorize)  # 可以加入去极值的参数
    postprocess_queue.add_step(Postprocess.ind_cap_neutralize,
                               industry=industry_df,
                               cap=cap_df)
    postprocess_queue.add_step(Postprocess.standardize)  # 可以加入标准化的参数

    # 因子回测
    factor_backtest = FactorBacktest(
        bar_df=stock_bar_df,  # <后复权>行情数据
        factor_df=stock_bar_df[['open']] + 1,  # 计算好的因子数据，可直接传入
        factor_expressions=expressions,  # 需要计算的因子表达式
        benchmark_df=None,  # 基准数据，比如沪深300，如果不提供代表每天等权持仓且日度换仓
        forward_periods=[1, 5, 10, 20],  # 预测未来收益的时间跨度
        position_adjust_datetimes=position_dates,  # 调仓日期
        postprocess_queue=postprocess_queue,  # 因子后处理
        choice=(0.8, 1.0),  # 单因子策略根据因子值的选股范围
        output_dir=None,  # 回测结果输出路径
        n_groups=5,  # 分组数
        n_jobs=1,  # 并行计算的进程数
    )
    factor_backtest.run(try_optimize=True)  # 运行回测，并且进行组合优化
