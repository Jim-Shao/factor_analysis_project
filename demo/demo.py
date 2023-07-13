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

    stock_bar_df = generate_stock_df(n_stocks=10, n_days=800)  # 行情数据
    benchmark_df = generate_stock_df(n_stocks=1, n_days=800)  # 基准数据，比如沪深300
    industry_df = generate_industry_df(n_stocks=10, n_days=800)  # 行业数据
    cap_df = generate_cap_df(n_stocks=10, n_days=800)  # 市值数据

    # 需要计算的因子
    expression1 = '(ts_corr(rank(ts_delta(log(volume), 2)), rank(div(diff(close, open), open)), 6))'
    expression2 = '(neg(ts_corr(rank(ts_delta(log(volume), 2)), rank(div(diff(close, open), open)), 6)))'
    expression3 = '(rank(ts_corr(rank(ts_delta(log(volume), 2)), rank(div(diff(close, open), open)), 6)))'
    expression4 = '(neg(rank(ts_corr(rank(ts_delta(log(volume), 2)), rank(div(diff(close, open), open)), 6))))'
    expression5 = '(neg(neg(ts_corr(rank(ts_delta(log(volume), 2)), rank(div(diff(close, open), open)), 6))))'
    expression6 = '(neg(neg(neg(ts_corr(rank(ts_delta(log(volume), 2)), rank(div(diff(close, open), open)), 6)))))'
    expressions = [
        expression1, expression2, expression3, expression4, expression5,
        expression6
    ]

    # 因子后处理（一般的后处理步骤为去极值、行业市值中性化、标准化）
    postprocess_queue = PostprocessQueue()
    postprocess_queue.add_step(Postprocess.winsorize)
    postprocess_queue.add_step(Postprocess.ind_cap_neutralize,
                              industry=industry_df,
                              cap=cap_df)
    postprocess_queue.add_step(Postprocess.standardize)

    # 因子回测
    factor_backtest = FactorBacktest(
        bar_df=stock_bar_df,
        # factor_df=stock_bar_df[['open']] + 1,
        factor_expressions=expressions,
        benchmark_df=benchmark_df,
        forward_periods=[1, 5, 10],
        position_adjust_datetimes=[
            pd.Timestamp('2010-04-15'),
            pd.Timestamp('2010-05-18'),
            pd.Timestamp('2010-06-15')
        ],
        postprocess_queue=postprocess_queue,
        output_dir=None,
        n_groups=5,
        n_jobs=6,
    )
    factor_backtest.run()
