#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@Author  :   shaoshijie
@File    :   summary.py
@Description    :   生成总结：1. 单个选股池下，不同因子的表现；2. 单个因子下，不同选股池的表现
'''

import re
import os
import shutil
import PyPDF2
import pandas as pd
import numpy as np
import seaborn as sns

from typing import List, Dict, Tuple, Literal

from factor_analysis.markdown_writer import MarkdownWriter
from factor_analysis.plot import plot_log_return

UNIVERSE_ORDERS = ['上证50', '沪深300', '中证500', '中证1000']
N_LINES_UPLIMIT = 6


def find_common_elements(list_of_lists: List[List]) -> List:
    """找到多个列表中的公共元素"""
    if not list_of_lists:
        return []

    common_elements = set(list_of_lists[0])

    for lst in list_of_lists[1:]:
        common_elements.intersection_update(lst)

    common_elements = list(common_elements)
    common_elements.sort()
    return common_elements


def merge_pdfs(input_pdfs, output_pdf):
    """合并多个PDF文件"""
    # 创建一个PdfWriter对象用于合并
    pdf_writer = PyPDF2.PdfWriter()

    for pdf_file in input_pdfs:
        # 打开每个PDF文件
        with open(pdf_file, 'rb') as pdf:
            # 创建一个PdfReader对象
            pdf_reader = PyPDF2.PdfReader(pdf)
            # 将所有页面复制到PdfWriter对象
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                pdf_writer.add_page(page)

    # 保存合并后的PDF到输出文件
    with open(output_pdf, 'wb') as output:
        pdf_writer.write(output)


def get_forward_period(file: str) -> int:
    """从文件名中获取预测期数"""
    return int(re.findall(r'_(\d+)D\.csv', file)[0])


def get_forward_period_return(file: str) -> int:
    """从文件名中获取预测期数"""
    return int(re.findall(r'_(\d+)D_return\.csv', file)[0])


def get_forward_period_group(file: str) -> int:
    """从文件名中获取预测期数"""
    return int(re.findall(r'_(\d+)D_group\.csv', file)[0])


def add_table_cell_color(
    df: pd.DataFrame,
    html_str: str,
    alpha: float = 0.3,
    text_align: Literal['left', 'center', 'right'] = 'right',
) -> str:
    """为df中的每个单元格添加颜色"""
    assert 0 <= alpha <= 1, 'alpha必须在0和1之间'
    color_df = df.copy()
    for c in color_df.columns:
        color_df[c] = color_df[c]
        _max = color_df[c].max()
        _min = color_df[c].min()
        color_df[c] = color_df[c].apply(lambda x: round(
            (x - _min) / (_max - _min), 2) if _max != _min else 0.5)
        color_df[c] = color_df[c].apply(lambda x: 0.5 + (x - 0.5) * 0.3)
        color_df[c] = color_df[c].apply(lambda x: sns.color_palette(
            "RdBu_r", as_cmap=True)(x)[:3] + (alpha, ))
        color_df[c] = color_df[c].apply(
            lambda x: tuple([int(i * 255) for i in x]))
    html_lines = html_str.split('\n')
    row_pos = 0
    for idx in color_df.index:
        for pos, line in enumerate(html_lines):
            if line.endswith('<tr>'):
                if html_lines[pos + 1].endswith(f'<th>{idx}</th>'):
                    row_pos = pos + 1
                    break
        for i, c in enumerate(color_df.columns):
            html_lines[row_pos + i + 1] = re.sub(
                r'<td>(.*?)</td>',
                f'<td style="text-align:{text_align}; background-color:rgba{color_df.loc[idx, c]}">{df.loc[idx, c]}</td>',
                html_lines[row_pos + i + 1],
            )
    return '\n'.join(html_lines)


class Summary:
    def __init__(self, analysis_output_dir: str):
        """比较选股池中不同因子的表现，以及一个因子在不同的选股池中的表现

        Parameters
        ----------
        analysis_output_dir : str
            因子分析结果的输出目录，比如'/home/shaoshijie/factor_analysis_project/factor_analysis_output'
        """
        if not os.path.exists(analysis_output_dir):
            raise FileNotFoundError(f'{analysis_output_dir} not found')
        self.analysis_output_dir = analysis_output_dir

        # 得到analysis_output_dir下的所有的文件夹，每个文件夹代表一个选股池
        self.universe_list = [
            u.name for u in os.scandir(analysis_output_dir) if u.is_dir()
        ]
        if set(self.universe_list).issubset(set(UNIVERSE_ORDERS)):
            self.universe_list = [
                u for u in UNIVERSE_ORDERS if u in self.universe_list
            ]
        else:
            self.universe_list.sort()

        # 对于每个选股池，得到其下的所有因子
        self.universe_factors: Dict[str, List[str]] = {}
        for universe in self.universe_list:
            universe_dir = os.path.join(analysis_output_dir, universe)
            factors = [
                f.name for f in os.scandir(universe_dir)
                if f.is_dir() and '+' not in f.name
            ]
            factors.sort()
            self.universe_factors[universe] = factors

        # 找到所有选股池中的公共因子
        self.common_factors = find_common_elements(
            list(self.universe_factors.values()))

        # 找到每个因子对应的预测期数
        self.forward_periods: Dict[Tuple[str], List[int]] = {}
        for universe in self.universe_list:
            for factor in self.universe_factors[universe]:
                factor_dir = os.path.join(analysis_output_dir, universe,
                                          factor)
                IC_dir = os.path.join(factor_dir, 'IC')
                IC_csv_files = [
                    f.name for f in os.scandir(IC_dir)
                    if f.name.endswith('.csv')
                ]
                forward_periods = [get_forward_period(f) for f in IC_csv_files]
                forward_periods.sort()
                self.forward_periods[(universe, factor)] = forward_periods
        self.universe_common_forward_periods = find_common_elements(
            list(self.forward_periods.values()))
        self.universe_common_forward_periods.sort()

        # 生成markdown writer
        self.md_writer = MarkdownWriter(
            md_path=os.path.join(analysis_output_dir, 'summary.md'))
        self.md_writer.add_title('因子分析报告', level=1)

        # 待删除
        self.to_delete_files = set()
        # self.to_delete_files.add(
        #     os.path.join(analysis_output_dir, 'summary.md'))
        # self.to_delete_files.add(
        #     os.path.join(analysis_output_dir, 'summary.html'))

    def compare_across_factors(
            self,
            period: int = None,
            IC_type: str = 'RankIC',
            IC_show_fields: List[str] = ['mean', 'std', 'IR'],
            IC_rank_by: str = 'IR',
            return_rank_by: str = 'sharpe'):
        for universe in self.universe_list:
            self._compare_across_factors(universe=universe,
                                         period=period,
                                         IC_type=IC_type,
                                         IC_show_fields=IC_show_fields,
                                         IC_rank_by=IC_rank_by,
                                         return_rank_by=return_rank_by)

    def compare_across_universes(
        self,
        period: int = None,
        IC_type: str = 'RankIC',
        IC_show_fields: List[str] = ['mean', 'std', 'IR'],
        IC_rank_by: str = 'IR',
        return_rank_by: str = 'sharpe',
    ):
        for factor in self.common_factors:
            self._compare_across_universes(factor=factor,
                                           period=period,
                                           IC_type=IC_type,
                                           IC_show_fields=IC_show_fields,
                                           IC_rank_by=IC_rank_by,
                                           return_rank_by=return_rank_by)

    def to_pdf(self, concat_pdfs: bool = False):
        output_pdf_path = os.path.join(self.analysis_output_dir, 'summary.pdf')
        self.md_writer.to_pdf(output_pdf_path)
        print(f'生成summary.pdf文件: {output_pdf_path}')
        self.delete_files()
        if concat_pdfs:
            self.concat_pdfs()

    def _compare_across_factors(
        self,
        universe: str,
        period: int = None,
        IC_type: str = 'RankIC',
        IC_show_fields: List[str] = ['mean', 'std', 'IR'],
        IC_rank_by: str = 'IR',
        return_rank_by: str = 'sharpe',
    ):
        """比较一个选股池中不同因子的表现"""
        assert IC_type in ['RankIC', 'IC'], 'IC_type必须为RankIC或者IC'
        assert all([f in ['mean', 'std', 'IR']
                    for f in IC_show_fields]), 'IC_show_fields只能包含mean、std或者IR'
        IC_show_fields = [f'{IC_type}_{f}' for f in IC_show_fields]
        if period is None:
            period = self.universe_common_forward_periods[0]
        self.md_writer.add_title(
            f'{universe}，{len(self.universe_factors[universe])}个因子', level=2)
        self.md_writer.add_title(f'{IC_type}比较，根据(T={period},{IC_rank_by})降序',
                                 level=3)
        self.compare_IC_across_factors(universe=universe,
                                       show_fields=IC_show_fields,
                                       rank_by_period=period,
                                       rank_by_value=IC_rank_by)
        # self.plot_IC_across_factors()
        self.md_writer.add_title(f'因子(T={period})收益比较，根据{return_rank_by}降序',
                                 level=3)
        self.compare_return_across_factors(universe=universe,
                                           period=period,
                                           rank_by_value=return_rank_by)
        self.md_writer.add_title(f'单因子策略收益比较，根据{return_rank_by}降序', level=3)
        self.compare_quantile_across_factors(universe=universe,
                                             rank_by_value=return_rank_by)
        self.md_writer.add_pagebreak()
        self.md_writer.add_title('因子相关性热力图', level=3)
        self.compare_corr_across_factors(universe=universe)
        self.md_writer.add_title('因子等权组合优化', level=3)
        self.report_optimization(universe=universe,
                                 period=period,
                                 IC_type=IC_type,
                                 IC_show_fields=IC_show_fields,
                                 IC_rank_by=IC_rank_by,
                                 return_rank_by=return_rank_by)
        self.md_writer.add_pagebreak()

    def _compare_across_universes(
            self,
            factor: str,
            period: int = None,
            IC_type: str = 'RankIC',
            IC_show_fields: List[str] = ['mean', 'std', 'IR'],
            IC_rank_by: str = 'IR',
            return_rank_by: str = 'sharpe'):
        """比较一个因子在不同选股池中的表现"""
        assert len(self.universe_list) > 1, 'universe_list长度必须大于1'
        assert IC_type in ['RankIC', 'IC'], 'IC_type必须为RankIC或者IC'
        assert all([f in ['mean', 'std', 'IR']
                    for f in IC_show_fields]), 'IC_show_fields只能包含mean、std或者IR'
        IC_show_fields = [f'{IC_type}_{f}' for f in IC_show_fields]
        if period is None:
            period = self.universe_common_forward_periods[0]
        self.md_writer.add_title(f'{factor}，{len(self.universe_list)}个选股池',
                                 level=2)
        self.md_writer.add_title(f'{IC_type}比较，根据(T={period},{IC_rank_by})降序',
                                 level=3)
        self.compare_IC_across_universes(factor=factor,
                                         show_fields=IC_show_fields,
                                         rank_by_period=period,
                                         rank_by_value=IC_rank_by)
        self.md_writer.add_title(f'因子(T={period})收益比较，根据{return_rank_by}降序',
                                 level=3)
        self.compare_return_across_universes(factor=factor,
                                             period=period,
                                             rank_by_value=return_rank_by)
        self.md_writer.add_title(f'单因子策略收益比较，根据{return_rank_by}降序', level=3)
        self.compare_quantile_across_universes(factor=factor,
                                               rank_by_value=return_rank_by)
        self.md_writer.add_pagebreak()

    def compare_IC_across_factors(
        self,
        universe: str,
        show_fields: List[str],
        rank_by_period: int,
        rank_by_value: str = 'IR',
    ):
        """比较选股池中不同因子的表现
        """
        factor_ic_dict: Dict[str, pd.DataFrame] = {}
        for factor in self.universe_factors[universe]:
            factor_dir = os.path.join(self.analysis_output_dir, universe,
                                      factor)
            IC_dir = os.path.join(factor_dir, 'IC')
            IC_csv_files = [
                f.name for f in os.scandir(IC_dir) if f.name.endswith('.csv')
            ]
            IC_csv_files = [
                f for f in IC_csv_files if get_forward_period(f) in
                self.universe_common_forward_periods
            ]
            IC_csv_files.sort(key=get_forward_period)
            factor_ic_dict[factor] = pd.concat(
                [
                    pd.read_csv(
                        os.path.join(IC_dir, f),
                        index_col=0).loc[show_fields, ['total']].rename(
                            columns={'total': f'T={get_forward_period(f)}'})
                    for f in IC_csv_files
                ],
                axis=1,
            )
            factor_ic_dict[factor].index = [
                idx.split('_')[-1] for idx in factor_ic_dict[factor].index
            ]
            factor_ic_dict[factor] = factor_ic_dict[factor].unstack().to_frame(
            ).T
            factor_ic_dict[factor].index = [factor]
        total_df = pd.concat(factor_ic_dict.values(), axis=0)
        total_df = total_df.sort_values(by=(f'T={rank_by_period}',
                                            rank_by_value),
                                        ascending=False,
                                        key=abs)
        html_str = add_table_cell_color(total_df, total_df.to_html())
        self.md_writer.write(html_str)

    def compare_IC_across_universes(self,
                                    factor: str,
                                    show_fields: List[str],
                                    rank_by_period: int,
                                    rank_by_value: str = 'IR'):
        """比较选股池中不同因子的表现
        """
        factor_ic_dict: Dict[str, pd.DataFrame] = {}
        for universe in self.universe_list:
            factor_dir = os.path.join(self.analysis_output_dir, universe,
                                      factor)
            IC_dir = os.path.join(factor_dir, 'IC')
            IC_csv_files = [
                f.name for f in os.scandir(IC_dir) if f.name.endswith('.csv')
            ]
            IC_csv_files = [
                f for f in IC_csv_files if get_forward_period(f) in
                self.universe_common_forward_periods
            ]
            IC_csv_files.sort(key=get_forward_period)
            factor_ic_dict[universe] = pd.concat(
                [
                    pd.read_csv(
                        os.path.join(IC_dir, f),
                        index_col=0).loc[show_fields, ['total']].rename(
                            columns={'total': f'T={get_forward_period(f)}'})
                    for f in IC_csv_files
                ],
                axis=1,
            )
            factor_ic_dict[universe].index = [
                idx.split('_')[-1] for idx in factor_ic_dict[universe].index
            ]
            factor_ic_dict[universe] = factor_ic_dict[universe].unstack(
            ).to_frame().T
            factor_ic_dict[universe].index = [universe]
        total_df = pd.concat(factor_ic_dict.values(), axis=0)
        total_df = total_df.sort_values(by=(f'T={rank_by_period}',
                                            rank_by_value),
                                        ascending=False)
        html_str = add_table_cell_color(total_df, total_df.to_html())
        self.md_writer.write(html_str)

    def compare_return_across_factors(self,
                                      universe: str,
                                      period: int,
                                      rank_by_value: str = 'sharpe'):
        """比较选股池中不同因子的表现
        """
        _type_dict = {
            'long_excess': '多头超额',
            'short_excess': '空头超额',
            'long_short': '多空组合'
        }

        def compare_return_performance(_type: Literal['long_excess',
                                                      'short_excess',
                                                      'long_short']):
            factor_return_dict: Dict[str, pd.DataFrame] = {}
            for factor in self.universe_factors[universe]:
                factor_dir = os.path.join(self.analysis_output_dir, universe,
                                          factor)
                return_dir = os.path.join(factor_dir, 'net_value')
                turnover_dir = os.path.join(factor_dir, 'turnover')
                return_csv = [
                    f.name for f in os.scandir(return_dir)
                    if f.name.endswith('D.csv')
                    and get_forward_period(f.name) == period
                ][0]
                turnover_csv = [
                    f.name for f in os.scandir(turnover_dir)
                    if f.name.endswith('D_group.csv')
                    and get_forward_period_group(f.name) == period
                ][0]
                return_df = pd.read_csv(os.path.join(return_dir, return_csv),
                                        index_col=0).loc[_type].to_frame().T
                return_df.columns = [('return', c) for c in return_df.columns]
                return_df = return_df.iloc[:, :-3]

                turnover_df = pd.read_csv(
                    os.path.join(turnover_dir, turnover_csv),
                    index_col=0)[['turnover_count', 'turnover_mean']]

                if _type == 'long_short':
                    count = turnover_df.loc[
                        'group_1',
                        'turnover_count'] + turnover_df.loc['group_5',
                                                            'turnover_count']
                    mean = (
                        turnover_df.loc['group_1', 'turnover_mean'] *
                        turnover_df.loc['group_1', 'turnover_count'] +
                        turnover_df.loc['group_5', 'turnover_mean'] *
                        turnover_df.loc['group_5', 'turnover_count']) / count
                    turnover_df = pd.DataFrame(
                        [[count, mean]],
                        index=[_type],
                        columns=['turnover_count', 'turnover_mean'],
                    )
                    turnover_df['turnover_mean'] = turnover_df[
                        'turnover_mean'].round(4)
                elif _type == 'long_excess':
                    turnover_df = turnover_df.loc['group_5'].to_frame().T
                    turnover_df['turnover_count'] = turnover_df[
                        'turnover_count'].astype(int)
                    turnover_df.index = [_type]
                elif _type == 'short_excess':
                    turnover_df = turnover_df.loc['group_1'].to_frame().T
                    turnover_df['turnover_count'] = turnover_df[
                        'turnover_count'].astype(int)
                    turnover_df.index = [_type]
                turnover_df.columns = [('turnover', c.split('_')[-1])
                                       for c in turnover_df.columns]

                merge_df = pd.concat([return_df, turnover_df], axis=1)
                merge_df.columns = pd.MultiIndex.from_tuples(merge_df.columns)
                merge_df.index = [factor]
                factor_return_dict[factor] = merge_df

            total_df = pd.concat(factor_return_dict.values(), axis=0)
            total_df = total_df.sort_values(by=('return', rank_by_value),
                                            ascending=False)
            html_str = add_table_cell_color(total_df, total_df.to_html())
            self.md_writer.write(html_str)

        def compare_log_return(_type: Literal['long_excess', 'short_excess',
                                              'long_short']):

            factor_return_df_list = []
            for factor in self.universe_factors[universe]:
                factor_dir = os.path.join(self.analysis_output_dir, universe,
                                          factor)
                return_dir = os.path.join(factor_dir, 'net_value')
                return_csv = [
                    f.name for f in os.scandir(return_dir)
                    if f.name.endswith('D_return.csv')
                    and get_forward_period_return(f.name) == period
                ][0]
                factor_return_df = pd.read_csv(os.path.join(
                    return_dir, return_csv),
                                               index_col=0)[[_type]]
                factor_return_df.columns = [factor]
                factor_return_df_list.append(factor_return_df)

            total_df = pd.concat(factor_return_df_list, axis=1).sort_index()
            total_df = total_df.apply(lambda x: np.log(x + 1))
            log_return_df = total_df.cumsum()
            log_return_df.index = pd.to_datetime(log_return_df.index)
            # path = os.path.join(os.path.dirname(self.analysis_output_dir),
            #                     'log_return')
            # os.makedirs(path, exist_ok=True)
            path = self.analysis_output_dir
            plot_log_return(log_return_df, path, f'{universe}_{_type}')
            fig_path = os.path.join(path, f'{universe}_{_type}.svg')
            self.to_delete_files.add(fig_path)
            self.md_writer.add_image(f'{universe}{_type_dict[_type]}对数收益率',
                                        fig_path)

        for _type in ['long_short', 'long_excess', 'short_excess']:
            self.md_writer.add_title(_type_dict[_type], level=4)
            compare_return_performance(_type)
            compare_log_return(_type)

    def compare_return_across_universes(self,
                                        factor: str,
                                        period: int,
                                        rank_by_value: str = 'sharpe'):
        """比较因子在不同选股池中的表现
        """
        _type_dict = {
            'long_excess': '多头超额',
            'short_excess': '空头超额',
            'long_short': '多空组合'
        }

        def compare_return_performance(_type: Literal['long_excess',
                                                      'short_excess',
                                                      'long_short']):
            factor_return_dict: Dict[str, pd.DataFrame] = {}
            for universe in self.universe_list:
                factor_dir = os.path.join(self.analysis_output_dir, universe,
                                          factor)
                return_dir = os.path.join(factor_dir, 'net_value')
                turnover_dir = os.path.join(factor_dir, 'turnover')
                return_csv = [
                    f.name for f in os.scandir(return_dir)
                    if f.name.endswith('D.csv')
                    and get_forward_period(f.name) == period
                ][0]
                turnover_csv = [
                    f.name for f in os.scandir(turnover_dir)
                    if f.name.endswith('D_group.csv')
                    and get_forward_period_group(f.name) == period
                ][0]
                return_df = pd.read_csv(os.path.join(return_dir, return_csv),
                                        index_col=0).loc[_type].to_frame().T
                return_df.columns = [('return', c) for c in return_df.columns]
                return_df = return_df.iloc[:, :-3]

                turnover_df = pd.read_csv(
                    os.path.join(turnover_dir, turnover_csv),
                    index_col=0)[['turnover_count', 'turnover_mean']]

                if _type == 'long_short':
                    count = turnover_df.loc[
                        'group_1',
                        'turnover_count'] + turnover_df.loc['group_5',
                                                            'turnover_count']
                    mean = (
                        turnover_df.loc['group_1', 'turnover_mean'] *
                        turnover_df.loc['group_1', 'turnover_count'] +
                        turnover_df.loc['group_5', 'turnover_mean'] *
                        turnover_df.loc['group_5', 'turnover_count']) / count
                    turnover_df = pd.DataFrame(
                        [[count, mean]],
                        index=[_type],
                        columns=['turnover_count', 'turnover_mean'],
                    )
                    turnover_df['turnover_mean'] = turnover_df[
                        'turnover_mean'].round(4)
                elif _type == 'long_excess':
                    turnover_df = turnover_df.loc['group_5'].to_frame().T
                    turnover_df['turnover_count'] = turnover_df[
                        'turnover_count'].astype(int)
                    turnover_df.index = [_type]
                elif _type == 'short_excess':
                    turnover_df = turnover_df.loc['group_1'].to_frame().T
                    turnover_df['turnover_count'] = turnover_df[
                        'turnover_count'].astype(int)
                    turnover_df.index = [_type]
                turnover_df.columns = [('turnover', c.split('_')[-1])
                                       for c in turnover_df.columns]

                merge_df = pd.concat([return_df, turnover_df], axis=1)
                merge_df.columns = pd.MultiIndex.from_tuples(merge_df.columns)
                merge_df.index = [universe]
                factor_return_dict[universe] = merge_df

            total_df = pd.concat(factor_return_dict.values(), axis=0)
            total_df = total_df.sort_values(by=('return', rank_by_value),
                                            ascending=False)
            html_str = add_table_cell_color(total_df, total_df.to_html())
            self.md_writer.write(html_str)

        def compare_log_return(_type: Literal['long_excess', 'short_excess',
                                              'long_short']):

            factor_return_df_list = []
            for universe in self.universe_list:
                factor_dir = os.path.join(self.analysis_output_dir, universe,
                                          factor)
                return_dir = os.path.join(factor_dir, 'net_value')
                return_csv = [
                    f.name for f in os.scandir(return_dir)
                    if f.name.endswith('D_return.csv')
                    and get_forward_period_return(f.name) == period
                ][0]
                factor_return_df = pd.read_csv(os.path.join(
                    return_dir, return_csv),
                                               index_col=0)[[_type]]
                factor_return_df.columns = [universe]
                factor_return_df_list.append(factor_return_df)

            total_df = pd.concat(factor_return_df_list, axis=1).sort_index()
            total_df = total_df.apply(lambda x: np.log(x + 1))
            log_return_df = total_df.cumsum()
            log_return_df.index = pd.to_datetime(log_return_df.index)
            # path = os.path.join(os.path.dirname(self.analysis_output_dir),
            #                     'log_return')
            # os.makedirs(path, exist_ok=True)
            path = self.analysis_output_dir
            plot_log_return(log_return_df, path, f'{factor}_{_type}')
            fig_path = os.path.join(path, f'{factor}_{_type}.svg')
            self.to_delete_files.add(fig_path)
            self.md_writer.add_image(f'{factor}{_type_dict[_type]}对数收益率',
                                        fig_path)

        for _type in ['long_short', 'long_excess', 'short_excess']:
            self.md_writer.add_title(_type_dict[_type], level=4)
            compare_return_performance(_type)
            compare_log_return(_type)

    def compare_quantile_across_factors(self,
                                        universe: str,
                                        rank_by_value: str = 'sharpe'):
        """比较选股池中不同因子的表现
        """
        factor_return_dict: Dict[str, pd.DataFrame] = {}
        for factor in self.universe_factors[universe]:
            factor_dir = os.path.join(self.analysis_output_dir, universe,
                                      factor)
            return_dir = os.path.join(factor_dir, 'net_value')
            return_csv = os.path.join(return_dir, f'{factor}_quantile.csv')
            turnover_dir = os.path.join(factor_dir, 'turnover')
            turnover_csv = os.path.join(turnover_dir, f'{factor}_quantile.csv')

            return_df = pd.read_csv(return_csv, index_col=0).loc[['excess'], [
                'ann_ret', 'max_dd', 'sharpe', 'calmar', 'win_rate', 'ann_vol'
            ]]
            return_df.index = [factor]
            return_df.columns = [('excess_return', c)
                                 for c in return_df.columns]
            turnover_df = pd.read_csv(turnover_csv, index_col=0)
            turnover_df.index = [factor]
            turnover_df.columns = [('turnover', c.split('_')[-1])
                                   for c in turnover_df.columns]
            factor_return_dict[factor] = pd.concat([return_df, turnover_df],
                                                   axis=1)
            factor_return_dict[factor].columns = pd.MultiIndex.from_tuples(
                factor_return_dict[factor].columns)

        total_df = pd.concat(factor_return_dict.values(), axis=0)
        total_df = total_df.sort_values(by=('excess_return', rank_by_value),
                                        ascending=False)
        html_str = add_table_cell_color(total_df, total_df.to_html())
        self.md_writer.write(html_str)

        factor_return_df_list = []
        for factor in self.universe_factors[universe]:
            factor_dir = os.path.join(self.analysis_output_dir, universe,
                                      factor)
            return_dir = os.path.join(factor_dir, 'net_value')
            return_csv = [
                f.name for f in os.scandir(return_dir)
                if f.name.endswith('quantile_return.csv')
            ][0]
            factor_return_df = pd.read_csv(os.path.join(
                return_dir, return_csv),
                                           index_col=0)[['excess']]
            factor_return_df.columns = [factor]
            factor_return_df_list.append(factor_return_df)

        total_df = pd.concat(factor_return_df_list, axis=1).sort_index()
        total_df = total_df.apply(lambda x: np.log(x + 1))
        log_return_df = total_df.cumsum()
        log_return_df.index = pd.to_datetime(log_return_df.index)
        # path = os.path.join(os.path.dirname(self.analysis_output_dir),
        #                     'log_return')
        # os.makedirs(path, exist_ok=True)
        path = self.analysis_output_dir
        plot_log_return(log_return_df, path, f'{universe}_quantile_log_return')
        fig_path = os.path.join(path, f'{universe}_quantile_log_return.svg')
        self.to_delete_files.add(fig_path)
        self.md_writer.add_image(f'{universe}单因子策略对数收益率',
                                    fig_path)

    def compare_quantile_across_universes(self,
                                          factor: str,
                                          rank_by_value: str = 'sharpe'):
        """比较因子在不同选股池中的表现
        """
        factor_return_dict: Dict[str, pd.DataFrame] = {}
        for universe in self.universe_list:
            factor_dir = os.path.join(self.analysis_output_dir, universe,
                                      factor)
            return_dir = os.path.join(factor_dir, 'net_value')
            return_csv = os.path.join(return_dir, f'{factor}_quantile.csv')
            turnover_dir = os.path.join(factor_dir, 'turnover')
            turnover_csv = os.path.join(turnover_dir, f'{factor}_quantile.csv')

            return_df = pd.read_csv(return_csv, index_col=0).loc[['excess'], [
                'ann_ret', 'max_dd', 'sharpe', 'calmar', 'win_rate', 'ann_vol'
            ]]
            return_df.index = [universe]
            return_df.columns = [('excess_return', c)
                                 for c in return_df.columns]
            turnover_df = pd.read_csv(turnover_csv, index_col=0)
            turnover_df.index = [universe]
            turnover_df.columns = [('turnover', c.split('_')[-1])
                                   for c in turnover_df.columns]
            factor_return_dict[universe] = pd.concat([return_df, turnover_df],
                                                     axis=1)
            factor_return_dict[universe].columns = pd.MultiIndex.from_tuples(
                factor_return_dict[universe].columns)

        total_df = pd.concat(factor_return_dict.values(), axis=0).sort_index()
        total_df = total_df.sort_values(by=('excess_return', rank_by_value),
                                        ascending=False)
        html_str = add_table_cell_color(total_df, total_df.to_html())
        self.md_writer.write(html_str)

        factor_return_df_list = []
        for universe in self.universe_list:
            factor_dir = os.path.join(self.analysis_output_dir, universe,
                                      factor)
            return_dir = os.path.join(factor_dir, 'net_value')
            return_csv = [
                f.name for f in os.scandir(return_dir)
                if f.name.endswith('quantile_return.csv')
            ][0]
            factor_return_df = pd.read_csv(os.path.join(
                return_dir, return_csv),
                                           index_col=0)[['excess']]
            factor_return_df.columns = [universe]
            factor_return_df_list.append(factor_return_df)

        total_df = pd.concat(factor_return_df_list, axis=1).sort_index()
        total_df = total_df.apply(lambda x: np.log(x + 1))
        log_return_df = total_df.cumsum()
        log_return_df.index = pd.to_datetime(log_return_df.index)
        # path = os.path.join(os.path.dirname(self.analysis_output_dir),
        #                     'log_return')
        # os.makedirs(path, exist_ok=True)
        path = self.analysis_output_dir
        plot_log_return(log_return_df, path, f'{factor}_quantile_log_return')
        fig_path = os.path.join(path, f'{factor}_quantile_log_return.svg')
        self.to_delete_files.add(fig_path)
        self.md_writer.add_image(f'{factor}单因子策略对数收益率',
                                    fig_path)

    def compare_corr_across_factors(self, universe: str):
        corr_dir = os.path.join(self.analysis_output_dir, universe,
                                'factor_corr.svg')
        self.md_writer.add_image(f'{universe}因子相关性', corr_dir)

    def report_optimization(
        self,
        universe: str,
        period: int,
        IC_type: str = 'RankIC',
        IC_show_fields: List[str] = ['mean', 'std', 'IR'],
        IC_rank_by: str = 'IR',
        return_rank_by: str = 'sharpe',
    ):
        """生成因子等权组合优化的报告"""
        universe_dir = os.path.join(self.analysis_output_dir, universe)
        optimized_factor = [
            factor for factor in os.listdir(universe_dir) if '+' in factor
        ]
        if not optimized_factor:
            self.md_writer.add_title('没有找到优化因子', level=4)
            return
        assert len(optimized_factor) == 1, '一个选股池只能有一个优化因子'
        optimized_factor = optimized_factor[0]
        components = optimized_factor.split('+')
        components_str = ', '.join(components)
        self.md_writer.add_title(f'优化因子组合: {components_str}', level=4)

        factor_dir = os.path.join(universe_dir, optimized_factor)
        IC_dir = os.path.join(factor_dir, 'IC')
        IC_csv_files = [
            f.name for f in os.scandir(IC_dir) if f.name.endswith('.csv')
        ]
        IC_csv_files = [
            f for f in IC_csv_files
            if get_forward_period(f) in self.universe_common_forward_periods
        ]
        IC_csv_files.sort(key=get_forward_period)
        IC_df = pd.concat(
            [
                pd.read_csv(
                    os.path.join(IC_dir, f),
                    index_col=0).loc[IC_show_fields, ['total']].rename(
                        columns={'total': f'T={get_forward_period(f)}'})
                for f in IC_csv_files
            ],
            axis=1,
        )
        IC_df.index = [idx.split('_')[-1] for idx in IC_df.index]
        IC_df = IC_df.unstack().to_frame().T
        IC_df.index = [universe]
        IC_df.columns = pd.MultiIndex.from_tuples(IC_df.columns)
        IC_df = IC_df.sort_values(by=(f'T={period}', IC_rank_by),
                                  ascending=False,
                                  key=abs)
        html_str = add_table_cell_color(IC_df, IC_df.to_html())
        self.md_writer.add_title(IC_type, level=4)
        self.md_writer.write(html_str)

        factor_dir = os.path.join(self.analysis_output_dir, universe,
                                  optimized_factor)

        factor_return_dict: Dict[str, pd.DataFrame] = {}
        return_dir = os.path.join(factor_dir, 'net_value')
        return_csv = os.path.join(return_dir,
                                  f'{optimized_factor}_quantile.csv')
        turnover_dir = os.path.join(factor_dir, 'turnover')
        turnover_csv = os.path.join(turnover_dir,
                                    f'{optimized_factor}_quantile.csv')

        return_df = pd.read_csv(return_csv, index_col=0).loc[
            ['excess'],
            ['ann_ret', 'max_dd', 'sharpe', 'calmar', 'win_rate', 'ann_vol']]
        return_df.index = [universe]
        return_df.columns = [('excess_return', c) for c in return_df.columns]
        turnover_df = pd.read_csv(turnover_csv, index_col=0)
        turnover_df.index = [universe]
        turnover_df.columns = [('turnover', c.split('_')[-1])
                               for c in turnover_df.columns]
        factor_return_dict[universe] = pd.concat([return_df, turnover_df],
                                                 axis=1)
        factor_return_dict[universe].columns = pd.MultiIndex.from_tuples(
            factor_return_dict[universe].columns)

        total_df = pd.concat(factor_return_dict.values(), axis=0).sort_index()
        total_df = total_df.sort_values(by=('excess_return', return_rank_by),
                                        ascending=False)
        html_str = add_table_cell_color(total_df, total_df.to_html())
        self.md_writer.add_title('单因子策略收益与换手率', level=4)
        self.md_writer.write(html_str)

    def concat_pdfs(self):
        """将每个因子的pdf合并成一个pdf"""
        merge_pdf = os.path.join(self.analysis_output_dir, 'merge.pdf')
        total_pdfs = []
        for universe in self.universe_list:
            for factor in self.universe_factors[universe]:
                total_pdfs.append((universe, factor))
        factor_pdfs = [
            os.path.join(self.analysis_output_dir, universe, factor,
                         f'{factor}_report.pdf')
            for universe, factor in total_pdfs
        ]
        merge_pdfs(factor_pdfs, merge_pdf)

    def delete_files(self):
        """删除中间文件"""
        for file in self.to_delete_files:
            if os.path.isfile(file):
                os.remove(file)
            elif os.path.isdir(file):
                shutil.rmtree(file)


if __name__ == '__main__':
    summary = Summary(
        analysis_output_dir=
        '/home/shaoshijie/factor_analysis_project/score_comb_1000')
    summary.compare_across_factors(period=20)
    # summary.compare_across_universes(period=20)
    summary.to_pdf()
    # summary.concat_pdfs()
