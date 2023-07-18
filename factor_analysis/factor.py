#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@Author  :   shaoshijie
@File    :   factor.py
@Description    :   单因子分析类
'''

import os
import pandas as pd
import numpy as np

from typing import Dict, Tuple, List, Union

from factor_analysis.performance import ReturnPerformance
from factor_analysis.markdown_writer import MarkdownWriter
from factor_analysis.utils import assert_std_index, calc_group_returns, calc_ic, cross_sectional_group_cut
from factor_analysis.plot import plot_net_value, plot_ic_series, plot_turnover


class Factor:
    def __init__(
        self,
        name: str,
        factor_series: pd.Series,
        forward_return_df: pd.DataFrame,
        bench_forward_return_df: pd.DataFrame,
        position_adjust_datetimes: List[pd.Timestamp] = None,
        n_group: int = 5,
        quantile: Union[Tuple[float, float], int] = (0.8, 1.0),
        output_dir: str = None,
    ) -> None:
        """因子分析类

        Parameters
        ----------
        name : str
            因子名称
        factor_series : pd.Series
            因子值序列，multi-index[datetime, order_book_id]
        forward_return_df : pd.DataFrame
            前瞻收益序列，multi-index[datetime, order_book_id]，columns如['1D', '5D', '10D',...]
        bench_forward_return_df : pd.DataFrame
            基准前瞻收益序列，multi-index[datetime, order_book_id]，columns如['1D', '5D', '10D',...]
        position_adjust_datetimes : List[pd.Timestamp], optional
            调仓日期列表, by default None
        n_group : int, optional
            分组数目, by default 5
        quantile : Union[Tuple[float, float], int], optional
            单因子策略的多头选择；
            若为tuple，则为分位数选择，如(0.8, 1.0)表示选择分位数80%~100%的股票；
            若为int，则为排序选择，如5表示选择排序在前五名，-5表示选择排序在后五名；
        output_dir : str, optional
            输出路径, by default None
        """
        self.name = name
        self.n_group = n_group
        self.quantile = quantile

        # 输出路径
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), 'factor_output', self.name)
        else:
            output_dir = os.path.join(output_dir, self.name)
        self.output_dir = output_dir

        # 因子分组收益，key为前瞻收益的时间长度，value为分组收益dataframe
        self.group_return_dict: Dict[str, pd.DataFrame] = {}

        # 判断输入数据是否符合multi-index[datetime, order_book_id]
        assert_std_index(factor_series, 'factor_series', 'series')
        assert_std_index(forward_return_df, 'forward_return_df', 'df')
        assert_std_index(bench_forward_return_df, 'bench_forward_return_df',
                         'df')
        self.factor_series = factor_series
        self.forward_return_df = forward_return_df
        self.bench_forward_return_df = bench_forward_return_df

        # 交易日期和调仓日期
        self.trading_dates = factor_series.index.get_level_values(
            'datetime').unique().sort_values()
        if position_adjust_datetimes is None:
            self.position_adjust_datetimes = self.trading_dates
        else:
            self.position_adjust_datetimes = position_adjust_datetimes

    def analyze(self) -> None:
        """因子全面分析"""
        self.md_writer = MarkdownWriter(
            md_path=f'{self.output_dir}/{self.name}_report.md',
            title=f'{self.name} 因子报告')

        # 分组
        self.group_cut()

        # 计算IC、净值、换手率
        self.calc_ic()
        self.calc_periodic_net_values()
        self.calc_position_weights()
        self.calc_positions()
        self.calc_return()
        self.calc_turnover()

        # 制作IC表现表格、IC时序图，输出到markdown
        self.make_IC_tables()
        self.plot_ic()
        self.report_ic()

        # 制作收益表现表格、收益时序图，输出到markdown
        self.make_return_performance()
        self.plot_net_value()
        self.report_return_performance()

        # 制作换手率表格、换手率时序图，输出到markdown
        self.make_turnover_table()
        self.plot_turnover()
        self.report_turnover()

    def analyze_quantile(self) -> None:
        """因子分位数超额收益分析"""
        self.calc_periodic_net_values()
        self.calc_quantile_position_weights()
        self.calc_quantile_positions(if_save=False)
        self.calc_quantile_return()
        self.make_quantile_return_performance()

    def group_cut(self) -> None:
        """因子分组"""
        count_non_nan = self.factor_series.groupby(
            level='datetime').apply(lambda x: x.count())
        filtered_dates = count_non_nan[count_non_nan > 0].index
        groups = cross_sectional_group_cut(
            self.factor_series.loc[filtered_dates], self.n_group)
        self.groups = groups

    def calc_ic(self) -> None:
        """计算IC、RankIC"""
        IC_df = pd.DataFrame()  # columns为如'1D', '5D',...，index为datetime
        Rank_IC_df = pd.DataFrame()  # columns如'1D', '5D',...，index为datetime
        for period, forward_return in self.forward_return_df.iteritems():
            ic, rank_ic = calc_ic(self.factor_series, forward_return)
            IC_df[period] = ic
            Rank_IC_df[period] = rank_ic
        self.IC_df, self.Rank_IC_df = IC_df, Rank_IC_df

    def calc_factor_weighted_position_weights(self) -> None:
        """计算因子加权持仓"""
        factor_series = self.factor_series[
            self.factor_series.index.get_level_values('datetime').isin(
                self.position_adjust_datetimes)]

        factor_weighted_positions = factor_series.groupby(
            'datetime').transform(lambda x: x / x.abs().sum())
        self._factor_weighted_positions = factor_weighted_positions

    def calc_quantile_position_weights(self) -> None:
        """计算所需分位数持仓"""
        factor_series = self.factor_series[
            self.factor_series.index.get_level_values('datetime').isin(
                self.position_adjust_datetimes)]

        def quantile_pos(x: pd.Series, start: float, end: float) -> pd.Series:
            """计算分位数持仓"""
            ranked_x = x.rank(pct=True)
            index = ranked_x[(ranked_x >= start) & (ranked_x <= end)].index
            n = len(index)
            if n > 0:
                x.loc[index] = 1 / n
                x.loc[x.index.difference(index)] = 0
                return x
            elif n == 0:
                x.loc[x.index] = 0
                return x

        # 如果是tuple分位数，选择对应的分位数持仓
        if isinstance(self.quantile, tuple):
            quantile_positions = factor_series.groupby('datetime').transform(
                lambda x: quantile_pos(x, *self.quantile))
        # 如果是int：如果为正，选择排序在前self.quantile名的股票；如果为负，选择排序在后-self.quantile名的股票
        elif isinstance(self.quantile, int):
            if self.quantile > 0:
                quantile_positions = factor_series.groupby(
                    'datetime').transform(lambda x: x.nlargest(self.quantile))
            elif self.quantile < 0:
                quantile_positions = factor_series.groupby(
                    'datetime').transform(
                        lambda x: x.nsmallest(-self.quantile))

        self._quantile_positions = quantile_positions

    def calc_position_weights(self) -> None:
        """计算持仓"""
        self.calc_factor_weighted_position_weights()
        self.calc_quantile_position_weights()

    def calc_positions(self) -> None:
        self.calc_factor_weighted_positions()
        self.calc_quantile_positions()

    def calc_factor_weighted_positions(self) -> None:
        # 得到每次调仓的目标权重
        positions = self._factor_weighted_positions
        positions = positions.unstack()
        positions['adjust_datetime'] = positions.index
        positions = positions.reindex(self.trading_dates)
        positions['adjust_datetime'] = positions['adjust_datetime'].ffill(
        ).fillna(self.trading_dates[0])
        positions = positions.ffill().fillna(0)

        # 得到每个调仓后持仓阶段的阶段性净值变化
        periodic_net_values = self.periodic_net_values
        periodic_weighted_net_values = periodic_net_values * positions.iloc[:, :
                                                                            -1]
        periodic_weighted_net_values_sum = periodic_weighted_net_values.abs(
        ).sum(axis=1)
        periodic_weighted_net_values_sum.name = 'weighted_net_value'
        periodic_weighted_net_values_sum = periodic_weighted_net_values_sum.to_frame(
        )
        periodic_weighted_net_values_sum['adjust_datetime'] = positions[
            'adjust_datetime']
        periodic_net_value_base = periodic_weighted_net_values_sum.groupby(
            'adjust_datetime').last()
        periodic_net_value_base.iloc[0] = 1

        # 得到每个阶段一开始的总净值基数
        periodic_net_value_base = periodic_net_value_base.cumprod()
        periodic_net_value_base = periodic_net_value_base.reindex(
            self.trading_dates).ffill()

        # 将总净值基数乘以该阶段的当天净值，得到实际当天净值
        total_net_values = periodic_weighted_net_values_sum[[
            'weighted_net_value'
        ]] * periodic_net_value_base
        total_net_values.iloc[-1] = np.nan

        # 计算每部分持仓的实际净值
        factor_weighted_positions = pd.DataFrame()
        for col in periodic_net_values.columns:
            factor_weighted_positions[col] = total_net_values[
                'weighted_net_value'] * periodic_weighted_net_values[col]

        self.total_factor_weighted_net_values = total_net_values
        self.factor_weighted_positions = factor_weighted_positions

    def calc_quantile_positions(self, if_save: bool = True) -> None:
        # 得到每次调仓的目标权重
        positions = self._quantile_positions
        positions = positions.unstack()
        positions['adjust_datetime'] = positions.index
        positions = positions.reindex(self.trading_dates)
        positions['adjust_datetime'] = positions['adjust_datetime'].ffill(
        ).fillna(self.trading_dates[0])
        positions = positions.ffill().fillna(0)

        # 得到每个调仓后持仓阶段的阶段性净值变化
        periodic_net_values = self.periodic_net_values
        periodic_weighted_net_values = periodic_net_values * positions.iloc[:, :
                                                                            -1]
        periodic_weighted_net_values_sum = periodic_weighted_net_values.abs(
        ).sum(axis=1)
        periodic_weighted_net_values_sum.name = 'weighted_net_value'
        periodic_weighted_net_values_sum = periodic_weighted_net_values_sum.to_frame(
        )
        periodic_weighted_net_values_sum['adjust_datetime'] = positions[
            'adjust_datetime']
        periodic_net_value_base = periodic_weighted_net_values_sum.groupby(
            'adjust_datetime').last()
        periodic_net_value_base.iloc[0] = 1

        # 得到每个阶段一开始的总净值基数
        periodic_net_value_base = periodic_net_value_base.cumprod()
        periodic_net_value_base = periodic_net_value_base.reindex(
            self.trading_dates).ffill()

        # 将总净值基数乘以该阶段的当天净值，得到实际当天净值
        total_net_values = periodic_weighted_net_values_sum[[
            'weighted_net_value'
        ]] * periodic_net_value_base
        total_net_values.iloc[-1] = np.nan

        # 计算每部分持仓的实际净值
        quantile_positions = pd.DataFrame()
        for col in periodic_net_values.columns:
            quantile_positions[col] = total_net_values[
                'weighted_net_value'] * periodic_weighted_net_values[col]

        self.total_quantile_net_values = total_net_values
        self.quantile_positions = quantile_positions

        if if_save == True:
            quantile_positions.to_csv(
                f'{self.output_dir}/quantile_positions.csv')

    def calc_periodic_net_values(self) -> None:
        """计算阶段净值"""
        returns = self.forward_return_df['1D'].unstack()
        returns.loc[self.position_adjust_datetimes,
                    'adjust_datetime'] = self.position_adjust_datetimes
        returns['adjust_datetime'] = returns['adjust_datetime'].ffill().fillna(
            returns.index[0])
        periodic_net_values = returns.groupby('adjust_datetime').transform(
            lambda x: (x + 1).cumprod())
        self.periodic_net_values = periodic_net_values

    def calc_group_turnover(self) -> None:
        """计算分组换手率"""
        old_groups = self.groups.groupby(level='order_book_id').shift(1)
        groups_df = pd.concat([self.groups, old_groups], axis=1)
        groups_df.columns = ['group', 'old_groups']
        groups_df[
            'group_change'] = groups_df['group'] != groups_df['old_groups']
        groups_turnover = groups_df.groupby([
            'datetime', 'old_groups'
        ])['group_change'].sum() / groups_df.groupby(
            ['datetime', 'old_groups'])['group_change'].count()
        groups_turnover = groups_turnover.unstack()
        groups_turnover.columns = [
            f'group_{i}' for i in range(1, self.n_group + 1)
        ]
        groups_turnover.columns.name = 'group'
        self.groups_turnover = groups_turnover

    def calc_factor_weighted_turnover(self) -> None:
        """计算因子加权持仓换手率"""
        forward_return_1D = (self.forward_return_df['1D'].unstack() +
                             1).cumprod()
        forward_return = forward_return_1D.loc[
            self.position_adjust_datetimes].pct_change() + 1

        chg_value = self._factor_weighted_positions.unstack().diff().abs()
        chg_value_adjusted = chg_value * forward_return
        total_value = self._factor_weighted_positions.unstack().abs()
        total_value_adjusted = total_value * forward_return

        factor_weighted_turnover = chg_value_adjusted.sum(
            axis=1) / total_value_adjusted.sum(axis=1)
        factor_weighted_turnover = factor_weighted_turnover.to_frame()
        factor_weighted_turnover.columns = ['turnover']
        factor_weighted_turnover /= 2
        factor_weighted_turnover.loc[self.position_adjust_datetimes[0],
                                     'turnover'] = 1
        self.factor_weighted_turnover = factor_weighted_turnover

    def calc_quantile_turnover(self) -> None:
        """计算分位数持仓换手率"""
        forward_return_1D = (self.forward_return_df['1D'].unstack() +
                             1).cumprod()
        forward_return = forward_return_1D.loc[
            self.position_adjust_datetimes].pct_change() + 1

        chg_value = self._quantile_positions.unstack().diff().abs()
        chg_value_adjusted = chg_value * forward_return
        total_value = self._quantile_positions.unstack().abs()
        total_value_adjusted = total_value * forward_return

        quantile_turnover = chg_value_adjusted.sum(
            axis=1) / total_value_adjusted.sum(axis=1)
        quantile_turnover = quantile_turnover.to_frame()
        quantile_turnover.columns = ['turnover']
        quantile_turnover /= 2
        quantile_turnover.loc[self.position_adjust_datetimes[0],
                              'turnover'] = 1
        self.quantile_turnover = quantile_turnover

    def calc_turnover(self) -> None:
        """计算换手率"""
        self.calc_group_turnover()
        self.calc_factor_weighted_turnover()
        self.calc_quantile_turnover()

    def calc_group_returns(self) -> None:
        """计算分组收益"""
        group_return_dict = {}
        for period in self.forward_return_df.columns:
            forward_return = self.forward_return_df[period]
            bench_forward_return = self.bench_forward_return_df[period]

            dates = self.groups.index.get_level_values('datetime').unique()
            forward_return = forward_return.loc[dates]
            bench_forward_return = bench_forward_return.loc[dates]

            group_return_dict[period] = calc_group_returns(
                forward_return, bench_forward_return, self.groups)
        self.group_return_dict = group_return_dict

    def calc_factor_weighted_return(self) -> None:
        """计算因子加权收益"""
        factor_weighted_return = self.total_factor_weighted_net_values.pct_change(
        )
        factor_weighted_return = factor_weighted_return.loc[
            self.position_adjust_datetimes[0]:].iloc[1:].dropna()
        factor_weighted_return.columns = ['factor_weighted']
        factor_weighted_return['benchmark'] = self.bench_forward_return_df[
            '1D'].unstack().iloc[:, 0]
        factor_weighted_return['excess'] = factor_weighted_return[
            'factor_weighted'] - factor_weighted_return['benchmark']
        self.factor_weighted_return = factor_weighted_return

    def calc_quantile_return(self) -> None:
        """计算分位数收益"""
        quantile_return = self.total_quantile_net_values.pct_change()
        quantile_return = quantile_return.loc[
            self.position_adjust_datetimes[0]:].iloc[1:].dropna()
        quantile_return.columns = ['quantile']
        quantile_return['benchmark'] = self.bench_forward_return_df[
            '1D'].unstack().iloc[:, 0]
        quantile_return['excess'] = quantile_return[
            'quantile'] - quantile_return['benchmark']
        self.quantile_return = quantile_return

    def calc_return(self) -> None:
        """计算收益"""
        self.calc_group_returns()
        self.calc_factor_weighted_return()
        self.calc_quantile_return()

    def make_IC_tables(self) -> None:
        """生成IC表格"""
        IC_df: pd.DataFrame = self.IC_df
        IC_summary_df_dict = {}

        def get_IC_summary(IC_series: pd.Series) -> pd.DataFrame:
            IC_mean = IC_series.mean()
            IC_std = IC_series.std()
            IC_IR = IC_mean / IC_std
            IC_summary = pd.DataFrame([{
                'IC_mean': IC_mean,
                'IC_std': IC_std,
                'IC_IR': IC_IR
            }])
            return IC_summary

        for period in IC_df.columns:
            IC_series: pd.Series = IC_df[period]

            year_IC = IC_series.groupby(
                IC_series.index.year).apply(get_IC_summary)
            year_IC = year_IC.reset_index(level=1, drop=True)
            total_IC = get_IC_summary(IC_series)
            total_IC.index = ['total']

            IC_summary_df = pd.concat([total_IC, year_IC], axis=0).T
            IC_summary_df_dict[period] = IC_summary_df

        self.IC_tables = IC_summary_df_dict

    def plot_ic(self):
        """绘制IC图"""
        output_dir = f'{self.output_dir}/IC'
        os.makedirs(output_dir, exist_ok=True)
        for period in self.IC_df.columns:
            plot_ic_series(self.IC_df[period], output_dir,
                           f'{self.name}_{period}')

    def report_ic(self):
        """生成IC报告至Markdown"""
        self.md_writer.add_title('IC分析', 2)
        for period, output_df in self.IC_tables.items():
            self.md_writer.add_title(period, 3)
            self.md_writer.add_image(
                'IC分布', f'{self.output_dir}/IC/{self.name}_{period}.png')
            self.md_writer.add_table(output_df, float_format='.4f')

    def make_group_return_performance(self):
        """生成分组收益表现"""
        group_return_dict = self.group_return_dict

        # calculate group return performance
        return_performance_df_dict = {}
        for period, group_returns in group_return_dict.items():
            period_int = int(period[:-1])

            groups = group_returns.columns.to_list()
            # [group_last, group1], [long_excess, short_excess, long_short]
            target_groups = [groups[-5], groups[0]] + groups[-4:]

            performance_dict_list = []
            for group in target_groups:
                # divide by period_int to get rolling position
                group_series = group_returns[group].dropna() / period_int
                performance_dict = ReturnPerformance(
                    group_series.values).performance()
                performance_dict_list.append(performance_dict)

            target_groups[0] = f'{target_groups[0]}(long)'
            target_groups[1] = f'{target_groups[1]}(short)'
            performance_df = pd.DataFrame(performance_dict_list,
                                          index=target_groups,
                                          columns=performance_dict.keys())
            return_performance_df_dict[period] = performance_df

        self.group_return_performance = return_performance_df_dict

    def make_factor_weighted_return_performance(self):
        """生成因子加权持仓收益表现"""
        factor_weighted_return = self.factor_weighted_return

        def get_performances(return_df: pd.DataFrame) -> pd.DataFrame:
            performance_dict_list = []
            for group in return_df.columns:
                group_series = return_df[group].dropna()
                performance_dict = ReturnPerformance(
                    group_series.values).performance()
                performance_dict_list.append(performance_dict)
            performance_df = pd.DataFrame(performance_dict_list,
                                          index=return_df.columns)
            return performance_df

        self.factor_weighted_return_performance = get_performances(
            factor_weighted_return)

    def make_quantile_return_performance(self):
        """生成分位数收益表现"""
        quantile_return = self.quantile_return

        def get_performances(return_df: pd.DataFrame) -> pd.DataFrame:
            performance_dict_list = []
            for group in return_df.columns:
                group_series = return_df[group].dropna()
                performance_dict = ReturnPerformance(
                    group_series.values).performance()
                performance_dict_list.append(performance_dict)
            performance_df = pd.DataFrame(performance_dict_list,
                                          index=return_df.columns)
            return performance_df

        self.quantile_return_performance = get_performances(quantile_return)

    def make_return_performance(self):
        """生成收益表现"""
        self.make_group_return_performance()
        self.make_factor_weighted_return_performance()
        self.make_quantile_return_performance()

    def plot_group_net_value(self):
        """绘制分组净值图"""
        output_dir = f'{self.output_dir}/net_value'
        os.makedirs(output_dir, exist_ok=True)
        for period, group_return in self.group_return_dict.items():
            period_int = int(period[:-1])
            plot_net_value(group_return, period_int, output_dir,
                           f'{self.name}_{period}')

    def plot_factor_weighted_net_value(self):
        """绘制因子加权持仓净值图"""
        output_dir = f'{self.output_dir}/net_value'
        os.makedirs(output_dir, exist_ok=True)
        plot_net_value(self.factor_weighted_return, 1, output_dir,
                       f'{self.name}_factor_weighted')

    def plot_quantile_net_value(self):
        """绘制分位数净值图"""
        output_dir = f'{self.output_dir}/net_value'
        os.makedirs(output_dir, exist_ok=True)
        plot_net_value(self.quantile_return, 1, output_dir,
                       f'{self.name}_quantile')

    def plot_net_value(self):
        """绘制净值图"""
        self.plot_group_net_value()
        self.plot_factor_weighted_net_value()
        self.plot_quantile_net_value()

    def report_group_return_performance(self):
        """生成分组收益表现报告至Markdown"""
        self.md_writer.add_title('分层收益', 3)
        for period, output_df in self.group_return_performance.items():
            self.md_writer.add_title(period, 3)
            self.md_writer.add_table(output_df, float_format='.4f')
            self.md_writer.add_image(
                '分层净值',
                f'{self.output_dir}/net_value/{self.name}_{period}.png')

    def report_factor_weighted_return_performance(self):
        """生成因子加权持仓收益表现报告至Markdown"""
        self.md_writer.add_title('因子加权日频收益', 3)
        self.md_writer.add_table(self.factor_weighted_return_performance,
                                 float_format='.4f')
        self.md_writer.add_image(
            '因子加权净值',
            f'{self.output_dir}/net_value/{self.name}_factor_weighted.png')

    def report_quantile_return_performance(self):
        """生成分位数收益表现报告至Markdown"""
        self.md_writer.add_title(f'分位数{self.quantile}收益', 3)
        self.md_writer.add_table(self.quantile_return_performance,
                                 float_format='.4f')
        self.md_writer.add_image(
            f'分位数净值{self.quantile}',
            f'{self.output_dir}/net_value/{self.name}_quantile.png')

    def report_return_performance(self):
        """生成收益表现报告至Markdown"""
        self.md_writer.add_title('收益分析', 2)
        self.report_group_return_performance()
        self.report_factor_weighted_return_performance()
        self.report_quantile_return_performance()

    def make_group_turnover_table(self):
        """生成分组换手率表"""
        groups_turnover = self.groups_turnover
        turnover_dict_list = []
        for group, turnover in groups_turnover.iteritems():
            turnover = turnover[turnover != 0]
            turnover_dict_list.append({
                'group': group,
                'turnover_count': turnover.count(),
                'turnover_mean': turnover.mean(),
                'turnover_std': turnover.std()
            })
        turnover_table = pd.DataFrame(turnover_dict_list).set_index('group')
        turnover_table.index.name = None
        self.turnover_table = turnover_table

    def make_factor_weighted_turnover_table(self):
        """生成因子加权换手率表"""
        factor_weighted_turnover = self.factor_weighted_turnover
        assert factor_weighted_turnover.shape[1] == 1, '因子加权换手率不止一列'
        factor_weighted_turnover = factor_weighted_turnover.iloc[:, 0]
        factor_weighted_turnover = factor_weighted_turnover[
            factor_weighted_turnover != 0]
        turnover_dict = {
            'turnover_count': factor_weighted_turnover.count(),
            'turnover_mean': factor_weighted_turnover.mean(),
            'turnover_std': factor_weighted_turnover.std()
        }
        factor_weighted_turnover_table = pd.DataFrame([turnover_dict])
        self.factor_weighted_turnover_table = factor_weighted_turnover_table

    def make_quantile_turnover_table(self):
        """生成分位数换手率表"""
        quantile_turnover = self.quantile_turnover
        assert quantile_turnover.shape[1] == 1, '分位数换手率不止一列'
        quantile_turnover = quantile_turnover.iloc[:, 0]
        quantile_turnover = quantile_turnover[quantile_turnover != 0]
        turnover_dict = {
            'turnover_count': quantile_turnover.count(),
            'turnover_mean': quantile_turnover.mean(),
            'turnover_std': quantile_turnover.std()
        }
        quantile_turnover_table = pd.DataFrame([turnover_dict])
        self.quantile_turnover_table = quantile_turnover_table

    def make_turnover_table(self):
        """生成换手率表"""
        self.make_group_turnover_table()
        self.make_factor_weighted_turnover_table()
        self.make_quantile_turnover_table()

    def plot_group_turnover(self):
        """绘制分组换手率图"""
        output_dir = f'{self.output_dir}/turnover'
        os.makedirs(output_dir, exist_ok=True)
        plot_turnover(self.groups_turnover, output_dir, f'{self.name}_group')

    def plot_factor_weighted_turnover(self):
        """绘制因子加权持仓换手率图"""
        output_dir = f'{self.output_dir}/turnover'
        os.makedirs(output_dir, exist_ok=True)
        daily_factor_weighted_turnover = pd.DataFrame(index=self.trading_dates,
                                                      columns=['turnover'])
        daily_factor_weighted_turnover.loc[
            self.position_adjust_datetimes,
            'turnover'] = self.factor_weighted_turnover['turnover']
        daily_factor_weighted_turnover = daily_factor_weighted_turnover.fillna(
            0)
        plot_turnover(daily_factor_weighted_turnover, output_dir,
                      f'{self.name}_factor_weighted')

    def plot_quantile_turnover(self):
        """绘制分位数持仓换手率图"""
        output_dir = f'{self.output_dir}/turnover'
        os.makedirs(output_dir, exist_ok=True)
        daily_quantile_turnover = pd.DataFrame(index=self.trading_dates,
                                               columns=['turnover'])
        daily_quantile_turnover.loc[
            self.position_adjust_datetimes,
            'turnover'] = self.quantile_turnover['turnover']
        daily_quantile_turnover = daily_quantile_turnover.fillna(0)
        plot_turnover(daily_quantile_turnover, output_dir,
                      f'{self.name}_quantile')

    def plot_turnover(self):
        """绘制换手率图"""
        self.plot_group_turnover()
        self.plot_factor_weighted_turnover()
        self.plot_quantile_turnover()

    def report_group_turnover(self):
        """生成分组换手率报告至Markdown"""
        self.md_writer.add_title('分层换手率', 3)
        self.md_writer.add_image(
            '分层换手率分布', f'{self.output_dir}/turnover/{self.name}_group.png')
        self.md_writer.add_table(self.turnover_table, float_format='.4f')

    def report_factor_weighted_turnover(self):
        """生成因子加权持仓换手率报告至Markdown"""
        self.md_writer.add_title('因子加权持仓换手率', 3)
        self.md_writer.add_image(
            '因子加权持仓换手率',
            f'{self.output_dir}/turnover/{self.name}_factor_weighted.png')
        self.md_writer.add_table(self.factor_weighted_turnover_table,
                                 float_format='.4f')

    def report_quantile_turnover(self):
        """生成分位数持仓换手率报告至Markdown"""
        self.md_writer.add_title(f'分位数{self.quantile}换手率', 3)
        self.md_writer.add_image(
            f'分位数换手率 {self.quantile}',
            f'{self.output_dir}/turnover/{self.name}_quantile.png')
        self.md_writer.add_table(self.quantile_turnover_table,
                                 float_format='.4f')

    def report_turnover(self):
        """生成换手率报告至Markdown"""
        self.md_writer.add_title('换手率分析', 2)
        self.report_group_turnover()
        self.report_factor_weighted_turnover()
        self.report_quantile_turnover()
