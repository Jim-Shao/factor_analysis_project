#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@Author  :   shaoshijie
@File    :   demo.py
@Description    :   用于展示如何使用factor_analysis
'''

from factor_analysis.backtest import FactorBacktest
from factor_analysis.postprocess import Postprocess, PostprocessQueue
from factor_analysis.summary import Summary

if __name__ == '__main__':
    import pandas as pd
    from factor_analysis.utils import generate_stock_df, generate_industry_df, generate_cap_df

    # 随机生成测试数据
    stock_bar_df1 = generate_stock_df(n_stocks=10, n_days=200, seed=0)  # 行情数据
    benchmark_df1 = generate_stock_df(n_stocks=1, n_days=200,
                                      seed=42)  # 基准数据，比如沪深300
    industry_df1 = generate_industry_df(n_stocks=10, n_days=200,
                                        seed=42)  # 行业数据
    cap_df1 = generate_cap_df(n_stocks=10, n_days=200, seed=42)  # 市值数据

    stock_bar_df2 = generate_stock_df(n_stocks=10, n_days=200, seed=43)  # 行情数据
    benchmark_df2 = generate_stock_df(n_stocks=1, n_days=200,
                                      seed=43)  # 基准数据，比如沪深300
    industry_df2 = generate_industry_df(n_stocks=10, n_days=200,
                                        seed=43)  # 行业数据
    cap_df2 = generate_cap_df(n_stocks=10, n_days=200, seed=43)  # 市值数据

    # 提取每月最后一个交易日作为调仓日期
    trade_dates = stock_bar_df1.index.get_level_values('datetime').unique()
    trade_dates = pd.DataFrame(trade_dates, columns=['datetime'])
    trade_dates['year'] = trade_dates['datetime'].dt.year
    trade_dates['month'] = trade_dates['datetime'].dt.month
    position_dates = trade_dates.groupby(['year', 'month']).last()
    position_dates = position_dates['datetime'].tolist()

    # 需要计算的因子（所有的算子需要以括号开头，以括号结尾，例：(open)）
    expression1 = '(rank(div(diff(close, open), open)))'
    expression2 = '(div(diff(close, open), open))'
    expression3 = '(ts_rank(volume, 10))'
    expression4 = '(rank(ts_rank(volume, 10)))'
    expressions1 = [expression1, expression2]
    expressions2 = [expression3, expression4]

    # 因子计算完成之后的处理（一般的后处理步骤为去极值、行业市值中性化、标准化）
    postprocess_queue1 = PostprocessQueue()
    postprocess_queue1.add_step(Postprocess.winsorize)  # 可以加入去极值的参数
    postprocess_queue1.add_step(Postprocess.ind_cap_neutralize,
                                industry=industry_df1,
                                cap=cap_df1)
    postprocess_queue1.add_step(Postprocess.standardize)  # 可以加入标准化的参数

    postprocess_queue2 = PostprocessQueue()
    postprocess_queue2.add_step(Postprocess.winsorize)  # 可以加入去极值的参数
    postprocess_queue2.add_step(Postprocess.ind_cap_neutralize,
                                industry=industry_df2,
                                cap=cap_df2)
    postprocess_queue2.add_step(Postprocess.standardize)  # 可以加入标准化的参数

    # 因子回测
    factor_backtest1 = FactorBacktest(
        universe='RandomGenerated1',
        bar_df=stock_bar_df1,  # <后复权>行情数据
        factor_df=stock_bar_df1[['open']] + 1,  # 计算好的因子数据，可直接传入
        factor_expressions=expressions1,  # 需要计算的因子表达式
        benchmark_df=benchmark_df1,  # 基准数据，比如沪深300，如果不提供代表每天等权持仓且日度换仓
        forward_periods=[1, 5, 10, 20],  # 预测未来收益的时间跨度
        position_adjust_datetimes=position_dates,  # 调仓日期
        postprocess_queue=postprocess_queue1,  # 因子后处理
        choice=(0.8, 1.0),  # 单因子策略根据因子值的选股范围
        analyze_fields=['IC', 'return', 'turnover'],  # 需要分析的内容
        output_dir=None,  # 回测结果输出路径
        n_groups=5,  # 分组数
        n_jobs=1,  # 并行计算的进程数
    )
    factor_backtest1.run(try_optimize=True)  # 运行回测，并且进行组合优化

    factor_backtest2 = FactorBacktest(
        universe='RandomGenerated2',
        bar_df=stock_bar_df2,  # <后复权>行情数据
        factor_df=stock_bar_df2[['open']] + 1,  # 计算好的因子数据，可直接传入
        factor_expressions=expressions2,  # 需要计算的因子表达式
        benchmark_df=None,  # 基准数据，比如沪深300，如果不提供代表每天等权持仓且日度换仓
        forward_periods=[1, 5, 10, 20],  # 预测未来收益的时间跨度
        position_adjust_datetimes=position_dates,  # 调仓日期
        postprocess_queue=postprocess_queue2,  # 因子后处理
        choice=(0.8, 1.0),  # 单因子策略根据因子值的选股范围
        analyze_fields=['IC', 'return', 'turnover'],  # 需要分析的内容
        output_dir=None,  # 回测结果输出路径
        n_groups=5,  # 分组数
        n_jobs=1,  # 并行计算的进程数
    )
    factor_backtest2.run(try_optimize=True)  # 运行回测，并且进行组合优化

    # 因子回测结果汇总（注意两个选股池的回测需要统一输出路径，这样才能比较单因子在不同选股池下的表现）
    output_dir1 = factor_backtest1.output_dir
    output_dir2 = factor_backtest2.output_dir
    assert output_dir1 == output_dir2, '两个选股池的回测需要统一输出路径'
    common_output_dir = output_dir1

    summary = Summary(common_output_dir)
    summary.compare_across_factors(period=20)
    summary.compare_across_universes(period=20)
    summary.to_pdf(concat_pdfs=False)
