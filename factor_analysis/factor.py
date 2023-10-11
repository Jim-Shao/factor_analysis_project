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

from typing import Dict, Tuple, List, Union, Literal

from factor_analysis.risk import Risk
from factor_analysis.markdown_writer import MarkdownWriter
from factor_analysis.utils import assert_std_index, calc_ic, rank_with_random_duplicates, cross_sectional_group_cut, get_IC_summary, calc_positions
from factor_analysis.plot import plot_net_value, plot_ic_series, plot_turnover, plot_ann_return, plot_line, plot_dist, plot_log_return


def quantile_pos(x: pd.Series, start: float, end: float) -> pd.Series:
    """分位数等权持仓

    Parameters
    ----------
    x : pd.Series
        单个时间戳下的因子值序列
    start : float
        开始分位数
    end : float
        结束分位数

    Returns
    -------
    pd.Series
        分位数等权持仓序列
    """
    ranked_x = rank_with_random_duplicates(x, pct=True)
    if start != 0:
        index = ranked_x[(ranked_x > start) & (ranked_x <= end)].index
    else:
        index = ranked_x[ranked_x <= end].index
    n = len(index)
    if n > 0:
        x.loc[index] = 1 / n
        x.loc[x.index.difference(index)] = 0
        return x
    elif n == 0:
        x.loc[x.index] = 0
        return x


def n_pos(x: pd.Series, n: int) -> pd.Series:
    """排序等权持仓 Top N 或 Bottom N

    Parameters
    ----------
    x : pd.Series
        单个时间戳下的因子值序列
    n : int
        排序等权持仓 Top N 或 Bottom N，正数为Top N，负数为Bottom N

    Returns
    -------
    pd.Series
        排序等权持仓序列
    """
    if n > 0:
        index = x.nlargest(n).index
    elif n < 0:
        index = x.nsmallest(-n).index
    x = pd.Series(0, index=x.index)
    if len(index) > 0:
        x.loc[index] = 1 / len(index)
    return x


class Factor:
    def __init__(
        self,
        name: str,
        universe: str,
        fields: Literal['IC', 'return', 'turnover'],
        factor_series: pd.Series,
        forward_return_df: pd.DataFrame,
        bench_forward_return_df: pd.DataFrame,
        group_periodic_net_values: Dict[str, pd.DataFrame],
        periodic_net_values: pd.DataFrame,
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
        universe : str
            股票池名称
        fields : List[str]
            需要分析的部分，可包含['IC', 'return', 'turnover']
        factor_series : pd.Series
            因子值序列，multi-index[datetime, order_book_id]
        forward_return_df : pd.DataFrame
            前瞻收益序列，multi-index[datetime, order_book_id]，columns如['1D', '5D', '10D',...]
        bench_forward_return_df : pd.DataFrame
            基准前瞻收益序列，multi-index[datetime, order_book_id]，columns如['1D', '5D', '10D',...]
        group_periodic_net_values : Dict[str, pd.DataFrame]
            固定间隔调仓阶段净值，key为调仓周期，value为净值序列，index为datetime，columns为group
        periodic_net_values : pd.DataFrame
            自定义调仓阶段净值，index为datetime，columns为order_book_id
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
        # 判断输入数据是否符合multi-index[datetime, order_book_id]
        assert_std_index(factor_series, 'factor_series', 'series')
        assert_std_index(forward_return_df, 'forward_return_df', 'df')
        assert_std_index(bench_forward_return_df, 'bench_forward_return_df',
                         'df')
        # 判断factor_series中是否有-inf或inf，有则报错
        inf_df = factor_series.loc[np.isinf(factor_series)]
        if len(inf_df) > 0:
            raise ValueError(f'{name}因子中共有{len(inf_df)}个inf值，请对因子进行去极值处理')
        # 判断fields是否合法
        assert set(fields) <= set(['IC', 'return', 'turnover'
                                   ]), 'fields中包含非法字段'

        self.name = name
        self.universe = universe
        self.n_group = n_group
        self.quantile = quantile
        self.group_periodic_net_values = group_periodic_net_values
        self.periodic_net_values = periodic_net_values
        self.fields = fields
        self.factor_series = factor_series
        self.forward_return_df = forward_return_df
        self.bench_forward_return_df = bench_forward_return_df

        # 输出路径
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), 'factor_analysis_output',
                                      self.universe, self.name)
        else:
            output_dir = os.path.join(output_dir, self.universe, self.name)
        self.output_dir = output_dir

        # 交易日期和调仓日期
        self.trading_dates = factor_series.index.get_level_values(
            'datetime').unique().sort_values()
        if position_adjust_datetimes is None:
            self.position_adjust_datetimes = self.trading_dates
        else:
            self.position_adjust_datetimes = position_adjust_datetimes

        # 因子值为nan的日期
        factor_df = self.factor_series.unstack()
        self.nan_dates = factor_df[factor_df.isna().all(
            axis=1)].index.get_level_values('datetime')

    def report_dist(self) -> None:
        """因子值分布报告"""
        # 去极值，防止生成的bins过多
        series = self.factor_series.copy()
        total_len = len(series)
        series_dropna = series.dropna()
        median = np.median(series_dropna)
        mad = np.median(np.abs(series_dropna - median))
        if mad == 0:
            std = series_dropna.std()
            mad = std * 0.6745
        series_drop = series_dropna[(series_dropna >= median - 5 * mad)
                                    & (series_dropna <= median + 5 * mad)]
        statistics = series_drop.describe().to_frame().T
        statistics = statistics.rename(columns={'count': 'n_valid'})
        statistics.insert(1, 'n_nan', total_len - len(series_dropna))
        statistics.insert(2, 'n_outlier',
                          len(series_dropna) - len(series_drop))
        statistics[['n_valid', 'n_nan', 'n_outlier'
                    ]] = statistics[['n_valid', 'n_nan',
                                     'n_outlier']].astype(int).astype(str)
        self.md_writer.add_title('因子值分布（已去极值）', 2)
        plot_dist(series_drop, True, self.output_dir, 'factor_dist')
        self.md_writer.add_image('因子值分布', f'{self.output_dir}/factor_dist.svg')
        self.md_writer.add_table(statistics, float_format='.4f', index=False)

    def report_non_nan(self) -> None:
        """因子值非空比例报告"""
        # 记录每天的因子值非空比例
        self.factor_non_nan_ratio = self.factor_series.groupby(
            level='datetime').apply(lambda x: x.count())
        self.factor_non_nan_ratio = self.factor_non_nan_ratio.to_frame()
        self.factor_non_nan_ratio.columns = ['non_nan_count']
        self.factor_non_nan_ratio['total_count'] = self.factor_series.groupby(
            level='datetime').apply(lambda x: len(x))
        self.factor_non_nan_ratio['non_nan_ratio'] = self.factor_non_nan_ratio[
            'non_nan_count'] / self.factor_non_nan_ratio['total_count']
        self.factor_non_nan_ratio.to_csv(
            f'{self.output_dir}/factor_non_nan_ratio.csv', float_format='%.4f')
        plot_line(self.factor_non_nan_ratio['non_nan_ratio'],
                  'factor non-nan ratio', self.output_dir,
                  'factor_non_nan_ratio')
        self.md_writer.add_title('因子值非空比例', 2)
        self.md_writer.add_image(
            '因子值非空比例', f'{self.output_dir}/factor_non_nan_ratio.svg')
        self.md_writer.add_pagebreak()

    def analyze(self) -> None:
        """因子全面分析
        """
        self.md_writer = MarkdownWriter(
            md_path=f'{self.output_dir}/{self.name}_report.md',
            title=f'{self.name}({self.universe}) 因子报告')

        self.report_dist()
        self.report_non_nan()

        if 'IC' in self.fields:
            self.analyze_ic()

        if 'return' in self.fields or 'turnover' in self.fields:
            self.calc_groups_and_positions()

            if 'return' in self.fields:
                self.analyze_return()

            if 'turnover' in self.fields:
                self.analyze_turnover()

        self.md_writer.to_pdf(f'{self.output_dir}/{self.name}_report.pdf')

    def analyze_quantile(self, if_save=False) -> None:
        """单因子策略超额收益分析"""
        self.calc_quantile_position_weights()
        self.calc_quantile_positions(if_save=if_save)
        self.calc_quantile_return()
        self.make_quantile_return_performance()

    # ===========================================================
    # region 1. IC分析
    # ===========================================================

    def analyze_ic(self) -> None:
        """IC分析"""
        self.calc_ic()
        self.make_IC_tables()
        self.plot_ic()
        self.report_ic()

    def calc_ic(self) -> None:
        """计算IC、RankIC"""
        IC_df_dict: Dict[str, pd.Series] = {}
        Rank_IC_df_dict: Dict[str, pd.Series] = {}
        for period, forward_return in self.forward_return_df.iteritems():
            ic, rank_ic = calc_ic(self.factor_series, forward_return)
            IC_df_dict[period] = ic
            Rank_IC_df_dict[period] = rank_ic
        self.IC_df = pd.DataFrame(IC_df_dict)
        self.Rank_IC_df = pd.DataFrame(Rank_IC_df_dict)

    def make_IC_tables(self) -> None:
        """生成IC表格"""
        IC_df, rank_IC_df = self.IC_df, self.Rank_IC_df
        IC_summary_df_dict = {}

        for period in IC_df.columns:
            IC_series = IC_df[period]
            year_IC = IC_series.groupby(
                IC_series.index.year).apply(lambda x: get_IC_summary(x, 'IC'))
            year_IC = year_IC.reset_index(level=1, drop=True)

            rank_IC_series = rank_IC_df[period]
            year_rank_IC = rank_IC_series.groupby(
                rank_IC_series.index.year).apply(
                    lambda x: get_IC_summary(x, 'RankIC'))
            year_rank_IC = year_rank_IC.reset_index(level=1, drop=True)

            year_IC = pd.concat([year_IC, year_rank_IC], axis=1)

            total_IC = get_IC_summary(IC_series, 'IC')
            total_rank_IC = get_IC_summary(rank_IC_series, 'RankIC')
            total_IC = pd.concat([total_IC, total_rank_IC], axis=1)
            total_IC.index = ['total']

            IC_summary_df = pd.concat([total_IC, year_IC], axis=0).T
            IC_summary_df_dict[period] = IC_summary_df

        self.IC_tables = IC_summary_df_dict

    def plot_ic(self):
        """绘制IC图"""
        output_dir = f'{self.output_dir}/IC'
        os.makedirs(output_dir, exist_ok=True)
        for period in self.IC_df.columns:
            plot_ic_series(self.IC_df[period], 'IC', output_dir,
                           f'{self.name}_{period}')
            plot_ic_series(self.Rank_IC_df[period], 'RankIC', output_dir,
                           f'{self.name}_{period}_Rank')

    def report_ic(self):
        """生成IC报告至Markdown"""
        self.md_writer.add_title('IC分析', 2)
        for period, output_df in self.IC_tables.items():
            self.md_writer.add_title(period, 3)
            self.md_writer.add_image(
                'IC分布', f'{self.output_dir}/IC/{self.name}_{period}.svg')
            IC_df = output_df.iloc[:3, :]
            IC_df.index = ['mean', 'std', 'IR']
            IC_df.index.name = 'IC'
            self.md_writer.add_table(IC_df, float_format='.4f')

            self.md_writer.add_image(
                'RankIC分布',
                f'{self.output_dir}/IC/{self.name}_{period}_Rank.svg')
            Rank_IC_df = output_df.iloc[3:, :]
            Rank_IC_df.index = ['mean', 'std', 'IR']
            Rank_IC_df.index.name = 'RankIC'
            self.md_writer.add_table(Rank_IC_df, float_format='.4f')

            self.md_writer.add_pagebreak()
            output_df.to_csv(f'{self.output_dir}/IC/{self.name}_{period}.csv',
                             float_format='%.4f')

    # endregion =================================================

    # ===========================================================
    # region 2. 计算分组和持仓
    # ===========================================================

    def calc_groups_and_positions(self) -> None:
        """计算分组和持仓"""
        self.group_cut()
        self.calc_position_weights()
        self.calc_positions()

    def group_cut(self) -> None:
        """因子分组"""
        count_non_nan = self.factor_series.groupby(
            level='datetime').apply(lambda x: x.count())
        filtered_dates = count_non_nan[count_non_nan > 0].index
        groups = cross_sectional_group_cut(
            self.factor_series.loc[filtered_dates], self.n_group)
        groups = groups.reindex(self.factor_series.index)
        self.groups = groups

    def calc_position_weights(self) -> None:
        """计算换仓日目标持仓权重"""
        self.calc_group_position_weights()
        self.calc_factor_weighted_position_weights()
        self.calc_quantile_position_weights()

    def calc_positions(self) -> None:
        """计算每日持仓"""
        self.calc_group_positions()
        self.calc_factor_weighted_positions()
        self.calc_quantile_positions()

    def calc_group_position_weights(self) -> None:
        """计算分组持仓"""
        self._group_positions = {}
        for period, forward_return in self.forward_return_df.iteritems():
            period_int = int(period[:-1])
            adjust_datetimes = self.trading_dates[::period_int]
            for group in range(1, self.n_group + 1):
                group_positions = self.groups.loc[adjust_datetimes]
                group_positions = group_positions[group_positions == group]
                group_positions = group_positions.groupby(
                    level='datetime').transform(lambda x: 1 / len(x))
                self._group_positions[f'{period}_{group}'] = group_positions

    def calc_factor_weighted_position_weights(self) -> None:
        """计算因子加权持仓"""
        factor_series = self.factor_series[
            self.factor_series.index.get_level_values('datetime').isin(
                self.position_adjust_datetimes)]
        factor_weighted_positions = factor_series.groupby('datetime').apply(
            lambda x: x / x.abs().sum())
        # factor_weighted_positions_long = factor_series.groupby(
        #     'datetime').apply(lambda x: (x * (x > 0)) / (x * (x > 0)).sum())
        # factor_weighted_positions_short = factor_series.groupby(
        #     'datetime').apply(lambda x: -(x * (x < 0)) / (x * (x < 0)).sum())

        # 将long和short中的没有值的交易日的持仓设为等权持仓
        # factor_weighted_positions_long = factor_weighted_positions_long.groupby(
        #     level='datetime').apply(lambda x: x.fillna(1 / len(x)))
        # factor_weighted_positions_short = factor_weighted_positions_short.groupby(
        #     level='datetime').apply(lambda x: -x.fillna(1 / len(x)))

        self._factor_weighted_positions = factor_weighted_positions
        # self._factor_weighted_positions_long = factor_weighted_positions_long
        # self._factor_weighted_positions_short = factor_weighted_positions_short

    def calc_quantile_position_weights(self) -> None:
        """计算所需单因子策略持仓"""
        factor_series = self.factor_series[
            self.factor_series.index.get_level_values('datetime').isin(
                self.position_adjust_datetimes)]

        # 如果是tuple分位数，选择对应的分位数持仓
        if isinstance(self.quantile, tuple):
            quantile_positions = factor_series.groupby('datetime').transform(
                lambda x: quantile_pos(x, *self.quantile))
        # 如果是int：如果为正，选择排序在前self.quantile名的股票；如果为负，选择排序在后-self.quantile名的股票
        elif isinstance(self.quantile, int):
            if self.quantile > 0:
                quantile_positions = factor_series.groupby(
                    'datetime').transform(lambda x: n_pos(x, self.quantile))
            elif self.quantile < 0:
                quantile_positions = factor_series.groupby(
                    'datetime').transform(lambda x: n_pos(x, self.quantile))

        self._quantile_positions = quantile_positions

    def calc_group_positions(self) -> None:
        """计算分组持仓"""
        self.group_net_values = {}

        for period in self.forward_return_df.columns:
            series_list = []
            period_int = int(period[:-1])
            for group in range(1, self.n_group + 1):
                _group_positions = self._group_positions[f'{period}_{group}']
                result = calc_positions(
                    _group_positions, self.trading_dates,
                    self.group_periodic_net_values[period_int], self.nan_dates)
                series_list.append(result[1])
            group_net_values = pd.concat(series_list, axis=1)
            group_net_values.columns = [
                f'group_{i+1}' for i in range(group_net_values.shape[1])
            ]
            self.group_net_values[period] = group_net_values

    def calc_factor_weighted_positions(self) -> None:
        result_ls = calc_positions(self._factor_weighted_positions,
                                   self.trading_dates,
                                   self.periodic_net_values, self.nan_dates)
        self.factor_weighted_net_value_base = result_ls[0]
        self.total_factor_weighted_net_values = result_ls[1]
        self.factor_weighted_positions = result_ls[2]

        # try:
        #     result_l = calc_positions(self._factor_weighted_positions_long,
        #                               self.trading_dates,
        #                               self.periodic_net_values, self.nan_dates)
        #     self.factor_weighted_l_ok = True
        #     self.factor_weighted_net_value_base_l = result_l[0]
        #     self.total_factor_weighted_net_values_l = result_l[1]
        #     self.factor_weighted_positions_l = result_l[2]
        # except:
        #     self.factor_weighted_l_ok = False

        # try:
        #     result_s = calc_positions(self._factor_weighted_positions_short,
        #                               self.trading_dates,
        #                               self.periodic_net_values, self.nan_dates)
        #     self.factor_weighted_s_ok = True
        #     self.factor_weighted_net_value_base_s = result_s[0]
        #     self.total_factor_weighted_net_values_s = result_s[1]
        #     self.factor_weighted_positions_s = result_s[2]
        # except:
        #     self.factor_weighted_s_ok = False

    def calc_quantile_positions(self, if_save: bool = True) -> None:
        """计算单因子策略持仓"""
        result = calc_positions(self._quantile_positions, self.trading_dates,
                                self.periodic_net_values, self.nan_dates)

        self.quantile_net_value_base = result[0]
        self.total_quantile_net_values = result[1]
        self.quantile_positions = result[2]

        if if_save == True:
            self.quantile_positions.to_csv(
                f'{self.output_dir}/quantile_positions.csv',
                float_format='%.6f')

    # endregion =================================================

    # ===========================================================
    # region 3. 收益分析
    # ===========================================================

    def analyze_return(self) -> None:
        """收益分析"""
        self.calc_return()
        self.make_return_performance()
        self.plot_net_value()
        self.report_return_performance()

    def calc_return(self) -> None:
        """计算收益"""
        self.calc_group_returns()
        self.calc_factor_weighted_return(if_save=True)
        self.calc_quantile_return(if_save=True)

    def make_return_performance(self):
        """生成收益表现"""
        self.make_group_return_performance()
        self.make_factor_weighted_return_performance()
        self.make_quantile_return_performance()

    def plot_net_value(self):
        """绘制净值图"""
        self.plot_group_net_value()
        self.plot_factor_weighted_net_value()
        self.plot_quantile_net_value()

    def report_return_performance(self):
        """生成收益表现报告至Markdown"""
        self.md_writer.add_title('收益分析', 2)
        self.report_group_return_performance()
        self.report_factor_weighted_return_performance()
        self.md_writer.add_pagebreak()
        self.report_quantile_return_performance()
        self.md_writer.add_pagebreak()

    def calc_group_returns(self) -> None:
        """计算分组收益"""
        group_return_dict = {}
        for period, group_net_value in self.group_net_values.items():
            group_return = group_net_value.pct_change().dropna().copy()
            long_group = group_return.columns[-1]
            short_group = group_return.columns[0]
            group_return['benchmark'] = self.bench_forward_return_df[
                '1D'].unstack().iloc[:, 0]
            group_return['long_excess'] = group_return[
                long_group] - group_return['benchmark']
            group_return['short_excess'] = group_return[
                'benchmark'] - group_return[short_group]
            group_return['long_short'] = group_return[
                long_group] - group_return[short_group]
            group_return_dict[period] = group_return
            if not os.path.exists(f'{self.output_dir}/net_value'):
                os.makedirs(f'{self.output_dir}/net_value')
            group_return.to_csv(
                f'{self.output_dir}/net_value/{self.name}_{period}_return.csv',
                float_format='%.6f')
        self.group_return_dict = group_return_dict

    def calc_factor_weighted_return(self, if_save: bool = False) -> None:
        """计算因子加权收益"""
        factor_weighted_return = self.total_factor_weighted_net_values.pct_change(
        )
        # 删去第一个换仓日（包含）之前的收益nan，换仓日当天因为净值从0变为1导致收益为inf
        factor_weighted_return = factor_weighted_return.loc[
            self.position_adjust_datetimes[0]:].iloc[1:].dropna()
        factor_weighted_return = factor_weighted_return.to_frame()
        factor_weighted_return.columns = ['factor_weighted']
        factor_weighted_return['benchmark'] = self.bench_forward_return_df[
            '1D'].unstack().iloc[:, 0]
        # factor_weighted_return = factor_weighted_return[[
        #     'benchmark', 'factor_weighted_ls'
        # ]]
        factor_weighted_return['excess'] = factor_weighted_return[
            'factor_weighted'] - factor_weighted_return['benchmark']

        # if self.factor_weighted_l_ok:
        #     factor_weighted_return_l = self.total_factor_weighted_net_values_l.pct_change(
        #     )
        #     factor_weighted_return_l = factor_weighted_return_l.loc[
        #         self.position_adjust_datetimes[0]:].iloc[1:].dropna()
        #     factor_weighted_return_l = factor_weighted_return_l.to_frame()
        #     factor_weighted_return[
        #         'factor_weighted_l'] = factor_weighted_return_l
        #     factor_weighted_return[
        #         'factor_weighted_l_excess'] = factor_weighted_return[
        #             'factor_weighted_l'] - factor_weighted_return['benchmark']

        # if self.factor_weighted_s_ok:
        #     factor_weighted_return_s = self.total_factor_weighted_net_values_s.pct_change(
        #     )
        #     factor_weighted_return_s = factor_weighted_return_s.loc[
        #         self.position_adjust_datetimes[0]:].iloc[1:].dropna()
        #     factor_weighted_return_s = factor_weighted_return_s.to_frame()
        #     factor_weighted_return[
        #         'factor_weighted_s'] = factor_weighted_return_s
        #     factor_weighted_return[
        #         'factor_weighted_s_excess'] = factor_weighted_return[
        #             'factor_weighted_s'] + factor_weighted_return['benchmark']

        if if_save:
            if not os.path.exists(f'{self.output_dir}/net_value'):
                os.makedirs(f'{self.output_dir}/net_value')
            factor_weighted_return.to_csv(
                f'{self.output_dir}/net_value/{self.name}_factor_weighted_return.csv',
                float_format='%.6f')
        self.factor_weighted_return = factor_weighted_return

    def calc_quantile_return(self, if_save: bool = False) -> None:
        """计算单因子策略收益"""
        quantile_return = self.total_quantile_net_values.pct_change()
        # 删去第一个换仓日（包含）之前的收益nan，换仓日当天因为净值从0变为1导致收益为inf
        quantile_return = quantile_return.loc[
            self.position_adjust_datetimes[0]:].iloc[1:].dropna()
        quantile_return = quantile_return.to_frame()
        quantile_return.columns = ['strategy']
        quantile_return['benchmark'] = self.bench_forward_return_df[
            '1D'].unstack().iloc[:, 0]
        quantile_return['excess'] = quantile_return[
            'strategy'] - quantile_return['benchmark']
        if if_save:
            if not os.path.exists(f'{self.output_dir}/net_value'):
                os.makedirs(f'{self.output_dir}/net_value')
            quantile_return.to_csv(
                f'{self.output_dir}/net_value/{self.name}_quantile_return.csv',
                float_format='%.6f')
        self.quantile_return = quantile_return

    def make_group_return_performance(self):
        """生成分组收益表现"""
        group_return_dict = self.group_return_dict

        # calculate group return performance
        return_performance_df_dict = {}
        for period, group_returns in group_return_dict.items():
            groups = group_returns.columns.to_list()
            # [group_last, group1], [benchmark, long_excess, short_excess, long_short]
            target_groups = [groups[-5], groups[0]] + groups[-4:]

            performance_dict_list = []
            for group in target_groups:
                group_series = group_returns[group].dropna()
                performance_dict = Risk(group_series).performance()
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
                performance_dict = Risk(group_series).performance()
                performance_dict_list.append(performance_dict)
            performance_df = pd.DataFrame(performance_dict_list,
                                          index=return_df.columns)
            return performance_df

        self.factor_weighted_return_performance = get_performances(
            factor_weighted_return)

    def make_quantile_return_performance(self):
        """生成单因子策略收益表现"""
        quantile_return = self.quantile_return

        def get_performances(return_df: pd.DataFrame) -> pd.DataFrame:
            performance_dict_list = []
            for group in return_df.columns:
                group_series = return_df[group].dropna()
                performance_dict = Risk(group_series).performance()
                performance_dict_list.append(performance_dict)
            performance_df = pd.DataFrame(performance_dict_list,
                                          index=return_df.columns)
            return performance_df

        self.quantile_return_performance = get_performances(quantile_return)

    def plot_group_net_value(self):
        """绘制分组净值图"""
        output_dir = f'{self.output_dir}/net_value'
        os.makedirs(output_dir, exist_ok=True)
        for period, group_return in self.group_return_dict.items():
            period_int = int(period[:-1])
            plot_net_value(group_return, period_int, output_dir,
                           f'{self.name}_{period}')

            n = group_return.shape[1]
            cols = [group_return.columns[-5]
                    ] + ['group_1'] + group_return.columns[-4:].to_list()
            colors = [f'C{n-5}', 'C0'] + [f'C{n-4+i}' for i in range(4)]
            log_return = group_return[cols]
            log_return = log_return.apply(np.log1p)
            log_return = log_return.cumsum()
            plot_log_return(log_return,
                            output_dir,
                            f'{self.name}_{period}_log_return',
                            colors=colors)

            group_return = group_return[[
                'long_excess', 'short_excess', 'long_short'
            ]]
            year_return = group_return.groupby(group_return.index.year).apply(
                lambda x: ((x + 1).prod())**(250 / len(x)) - 1)
            year_return = year_return.round(4)
            colors = ['C6', 'C7', 'C8']
            plot_ann_return(year_return,
                            f'{self.output_dir}/net_value',
                            f'{self.name}_{period}_ann_ret',
                            colors=colors)

    def plot_factor_weighted_net_value(self):
        """绘制因子加权持仓净值图"""
        output_dir = f'{self.output_dir}/net_value'
        os.makedirs(output_dir, exist_ok=True)
        plot_net_value(self.factor_weighted_return,
                       1,
                       output_dir,
                       f'{self.name}_factor_weighted',
                       fig_size=(16, 6))

        log_return = self.factor_weighted_return.copy()
        log_return = log_return.apply(np.log1p)
        log_return = log_return.cumsum()
        plot_log_return(log_return, output_dir,
                        f'{self.name}_factor_weighted_log_return')

        year_return = self.factor_weighted_return.groupby(
            self.factor_weighted_return.index.year).apply(
                lambda x: ((x + 1).prod())**(250 / len(x)) - 1)
        # cols = ['factor_weighted_ls_excess']
        # colors = ['C2']
        # if self.factor_weighted_l_ok:
        #     cols.append('factor_weighted_l_excess')
        #     colors.append('C4')
        # if self.factor_weighted_s_ok:
        #     cols.append('factor_weighted_s_excess')
        #     colors.append('C6')
        # year_return = year_return[cols]
        plot_ann_return(
            year_return,
            output_dir,
            f'{self.name}_factor_weighted_ann_ret',
        )

    def plot_quantile_net_value(self):
        """绘制单因子策略净值图"""
        output_dir = f'{self.output_dir}/net_value'
        os.makedirs(output_dir, exist_ok=True)
        plot_net_value(self.quantile_return,
                       1,
                       output_dir,
                       f'{self.name}_quantile',
                       fig_size=(16, 6))

        log_return = self.quantile_return.copy()
        log_return = log_return.apply(np.log1p)
        log_return = log_return.cumsum()
        plot_log_return(log_return, output_dir,
                        f'{self.name}_quantile_log_return')

        year_return = self.quantile_return.groupby(
            self.quantile_return.index.year).apply(lambda x: ((x + 1).prod())**
                                                   (250 / len(x)) - 1)
        plot_ann_return(year_return, output_dir,
                        f'{self.name}_quantile_ann_ret')

    def report_group_return_performance(self):
        """生成分组收益表现报告至Markdown"""
        for period, output_df in self.group_return_performance.items():
            self.md_writer.add_title(period, 3)
            self.md_writer.add_image(
                '分层净值',
                f'{self.output_dir}/net_value/{self.name}_{period}.svg')
            self.md_writer.add_image(
                '累计对数收益率',
                f'{self.output_dir}/net_value/{self.name}_{period}_log_return.svg'
            )
            self.md_writer.add_image(
                '分层超额年化收益',
                f'{self.output_dir}/net_value/{self.name}_{period}_ann_ret.svg'
            )
            self.md_writer.add_table(output_df, float_format='.4f')
            output_df.to_csv(
                f'{self.output_dir}/net_value/{self.name}_{period}.csv',
                float_format='%.4f')
            self.md_writer.add_pagebreak()

    def report_factor_weighted_return_performance(self):
        """生成因子加权持仓收益表现报告至Markdown"""
        self.md_writer.add_title('因子加权日频收益', 3)
        self.md_writer.add_image(
            '因子加权净值',
            f'{self.output_dir}/net_value/{self.name}_factor_weighted.svg')
        self.md_writer.add_image(
            '因子加权累计对数收益率',
            f'{self.output_dir}/net_value/{self.name}_factor_weighted_log_return.svg'
        )
        self.md_writer.add_image(
            '因子加权超额年化收益',
            f'{self.output_dir}/net_value/{self.name}_factor_weighted_ann_ret.svg'
        )
        self.md_writer.add_table(self.factor_weighted_return_performance,
                                 float_format='.4f')
        self.factor_weighted_return_performance.to_csv(
            f'{self.output_dir}/net_value/{self.name}_factor_weighted.csv',
            float_format='%.4f')

    def report_quantile_return_performance(self):
        """生成单因子策略收益表现报告至Markdown"""
        if isinstance(self.quantile, tuple):
            title = f'单因子策略{self.quantile}'
        elif self.quantile > 0:
            title = f'单因子策略（前{self.quantile}支）'
        elif self.quantile < 0:
            title = f'单因子策略（后{-self.quantile}支）'
        else:
            raise ValueError('quantile不支持这个值')
        self.md_writer.add_title(f'{title}收益', 3)
        self.md_writer.add_image(
            f'{title}净值',
            f'{self.output_dir}/net_value/{self.name}_quantile.svg')
        self.md_writer.add_image(
            f'{title}累计对数收益率',
            f'{self.output_dir}/net_value/{self.name}_quantile_log_return.svg')
        self.md_writer.add_image(
            f'{title}超额年化收益',
            f'{self.output_dir}/net_value/{self.name}_quantile_ann_ret.svg')
        self.md_writer.add_table(self.quantile_return_performance,
                                 float_format='.4f')
        self.quantile_return_performance.to_csv(
            f'{self.output_dir}/net_value/{self.name}_quantile.csv',
            float_format='%.4f')

    # endregion =================================================

    # ===========================================================
    # region 4. 换手率分析
    # ===========================================================

    def analyze_turnover(self) -> None:
        """换手率分析"""
        self.calc_turnover()
        self.make_turnover_table()
        self.plot_turnover()
        self.report_turnover()

    def calc_turnover(self) -> None:
        """计算换手率"""
        self.calc_group_turnover()
        self.calc_factor_weighted_turnover()
        self.calc_quantile_turnover()

    def make_turnover_table(self):
        """生成换手率表"""
        self.make_group_turnover_table()
        self.make_factor_weighted_turnover_table()
        self.make_quantile_turnover_table()

    def plot_turnover(self):
        """绘制换手率图"""
        self.plot_group_turnover()
        self.plot_factor_weighted_turnover()
        self.plot_quantile_turnover()

    def report_turnover(self):
        """生成换手率报告至Markdown"""
        self.md_writer.add_title('换手率分析', 2)
        self.report_group_turnover()
        self.report_factor_weighted_turnover()
        self.report_quantile_turnover()
        self.md_writer.add_pagebreak()

    def calc_group_turnover(self) -> None:
        """计算分组换手率"""
        groups_turnover_dict = {}
        for period in self.forward_return_df.columns:
            dates = self._group_positions[
                f'{period}_1'].index.get_level_values('datetime').unique()
            groups = self.groups.loc[dates]
            old_groups = groups.groupby(level='order_book_id').shift(1)
            groups_df = pd.concat([groups, old_groups], axis=1)
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
            groups_turnover = groups_turnover.reindex(self.trading_dates)
            groups_turnover = groups_turnover.fillna(0)
            groups_turnover.columns.name = 'group'
            groups_turnover_dict[period] = groups_turnover
        self.groups_turnover_dict = groups_turnover_dict

    def calc_factor_weighted_turnover(self) -> None:
        """计算因子加权持仓换手率"""
        factor_weighted_positions = self._factor_weighted_positions.unstack()
        net_value_base = self.factor_weighted_net_value_base.loc[
            self.position_adjust_datetimes]
        factor_weighted_net_values = pd.concat(
            [(factor_weighted_positions[col] * net_value_base)
             for col in factor_weighted_positions.columns],
            axis=1)
        factor_weighted_turnover = factor_weighted_net_values.diff().abs().sum(
            axis=1) / factor_weighted_net_values.abs().sum(axis=1)
        factor_weighted_turnover = factor_weighted_turnover.to_frame()
        factor_weighted_turnover.columns = ['turnover']
        factor_weighted_turnover /= 2
        factor_weighted_turnover.loc[self.position_adjust_datetimes[0],
                                     'turnover'] = 0
        self.factor_weighted_turnover = factor_weighted_turnover

    def calc_quantile_turnover(self) -> None:
        """计算单因子策略持仓换手率"""
        quantile_positions = self._quantile_positions.unstack()
        net_value_base = self.quantile_net_value_base.loc[
            self.position_adjust_datetimes]
        quantile_net_values = pd.concat(
            [(quantile_positions[col] * net_value_base)
             for col in quantile_positions.columns],
            axis=1)
        quantile_turnover = quantile_net_values.diff().abs().sum(
            axis=1) / quantile_net_values.abs().sum(axis=1)
        quantile_turnover = quantile_turnover.to_frame()
        quantile_turnover.columns = ['turnover']
        quantile_turnover /= 2
        quantile_turnover.loc[self.position_adjust_datetimes[0],
                              'turnover'] = 0
        self.quantile_turnover = quantile_turnover

    def make_group_turnover_table(self):
        """生成分组换手率表"""
        turnover_table_dict = {}
        for period, groups_turnover in self.groups_turnover_dict.items():
            turnover_dict_list = []
            for group, turnover in groups_turnover.iteritems():
                turnover = turnover[turnover != 0]
                turnover_dict_list.append({
                    'group': group,
                    'turnover_count': turnover.count(),
                    'turnover_mean': turnover.mean(),
                    'turnover_std': turnover.std()
                })
            turnover_table = pd.DataFrame(turnover_dict_list).set_index(
                'group')
            turnover_table.index.name = None
            turnover_table_dict[period] = turnover_table
        self.turnover_table_dict = turnover_table_dict

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
        factor_weighted_turnover_table = pd.DataFrame(
            [turnover_dict], index=['factor_weighted'])
        self.factor_weighted_turnover_table = factor_weighted_turnover_table

    def make_quantile_turnover_table(self):
        """生成单因子策略换手率表"""
        quantile_turnover = self.quantile_turnover
        assert quantile_turnover.shape[1] == 1, '单因子策略换手率不止一列'
        quantile_turnover = quantile_turnover.iloc[:, 0]
        quantile_turnover = quantile_turnover[quantile_turnover != 0]
        turnover_dict = {
            'turnover_count': quantile_turnover.count(),
            'turnover_mean': quantile_turnover.mean(),
            'turnover_std': quantile_turnover.std()
        }
        quantile_turnover_table = pd.DataFrame([turnover_dict],
                                               index=['strategy'])
        self.quantile_turnover_table = quantile_turnover_table

    def plot_group_turnover(self):
        """绘制分组换手率图"""
        output_dir = f'{self.output_dir}/turnover'
        os.makedirs(output_dir, exist_ok=True)
        for period, groups_turnover in self.groups_turnover_dict.items():
            plot_turnover(groups_turnover, output_dir,
                          f'{self.name}_{period}_group')

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
        plot_turnover(daily_factor_weighted_turnover,
                      output_dir,
                      f'{self.name}_factor_weighted',
                      fig_size=(16, 6))

    def plot_quantile_turnover(self):
        """绘制单因子策略持仓换手率图"""
        output_dir = f'{self.output_dir}/turnover'
        os.makedirs(output_dir, exist_ok=True)
        daily_quantile_turnover = pd.DataFrame(index=self.trading_dates,
                                               columns=['turnover'])
        daily_quantile_turnover.loc[
            self.position_adjust_datetimes,
            'turnover'] = self.quantile_turnover['turnover']
        daily_quantile_turnover = daily_quantile_turnover.fillna(0)
        plot_turnover(daily_quantile_turnover,
                      output_dir,
                      f'{self.name}_quantile',
                      fig_size=(16, 6))

    def report_group_turnover(self):
        """生成分组换手率报告至Markdown"""
        for period, turnover_table in self.turnover_table_dict.items():
            self.md_writer.add_title(period, 3)
            self.md_writer.add_image(
                '分层换手率分布',
                f'{self.output_dir}/turnover/{self.name}_{period}_group.svg')
            self.md_writer.add_table(turnover_table, float_format='.4f')
            self.md_writer.add_pagebreak()
            turnover_table.to_csv(
                f'{self.output_dir}/turnover/{self.name}_{period}_group.csv',
                float_format='%.4f')

    def report_factor_weighted_turnover(self):
        """生成因子加权持仓换手率报告至Markdown"""
        self.md_writer.add_title('因子加权持仓换手率', 3)
        self.md_writer.add_image(
            '因子加权持仓换手率',
            f'{self.output_dir}/turnover/{self.name}_factor_weighted.svg')
        self.md_writer.add_table(self.factor_weighted_turnover_table,
                                 float_format='.4f')
        self.factor_weighted_turnover_table.to_csv(
            f'{self.output_dir}/turnover/{self.name}_factor_weighted.csv',
            float_format='%.4f')

    def report_quantile_turnover(self):
        """生成单因子策略持仓换手率报告至Markdown"""
        if isinstance(self.quantile, tuple):
            title = f'单因子策略{self.quantile}'
        elif self.quantile > 0:
            title = f'单因子策略（前{self.quantile}支）'
        elif self.quantile < 0:
            title = f'单因子策略（后{-self.quantile}支）'
        else:
            raise ValueError('quantile不支持这个值')
        self.md_writer.add_title(f'{title}换手率', 3)
        self.md_writer.add_image(
            f'{title}换手率',
            f'{self.output_dir}/turnover/{self.name}_quantile.svg')
        self.md_writer.add_table(self.quantile_turnover_table,
                                 float_format='.4f')
        self.quantile_turnover_table.to_csv(
            f'{self.output_dir}/turnover/{self.name}_quantile.csv',
            float_format='%.4f')

    # endregion =================================================
