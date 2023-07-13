#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@Author  :   shaoshijie
@File    :   plot.py
@Description    :   绘图（IC、净值曲线、换手率）
'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import datetime
from typing import Union, Tuple


def _non_nan_mean(x: pd.Series) -> float:
    """得到非nan的均值

    Parameters
    ----------
    x : pd.Series
        序列

    Returns
    -------
    float
        非nan的均值
    """
    non_nan_values = x[~np.isnan(x)]
    return non_nan_values.mean() if non_nan_values.size > 0 else np.nan


def _set_xticks(ax: plt.Axes, xtick_freq: str) -> None:
    """设置x轴刻度

    Parameters
    ----------
    ax : plt.Axes
        坐标轴
    xtick_freq : str
        刻度频率，可选值为'year', 'month', 'day'
    """
    if xtick_freq == 'year':
        years = mdates.YearLocator()
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    elif xtick_freq == 'month':
        months = mdates.MonthLocator()
        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    elif xtick_freq == 'day':
        days = mdates.DayLocator()
        ax.xaxis.set_major_locator(days)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=90)


def plot_net_value(
    returns: Union[pd.DataFrame, pd.Series],
    forward_period: int,
    fig_dir: str,
    fig_name: str,
    fig_size: Tuple[int, int] = (16, 7),
    dpi: int = 150,
    xtick_freq: str = 'year',
):
    """
    绘制净值曲线

    Parameters
    ----------
    returns : Union[pd.DataFrame, pd.Series]
        收益率序列，若为pd.DataFrame，则每一列为一条曲线
    forward_period : int
        滚动周期
    fig_dir : str
        图片保存文件夹路径，e.g. 'factor_analysis/report/figs'
    fig_name : str
        图片名称，e.g. 'net_value'
    fig_size : Tuple[int, int], optional
        图片大小, by default (16, 7)
    dpi : int, optional
        图片dpi, by default 150
    xtick_freq : str, optional
        x轴刻度频率，可选值为'year', 'month', 'day', by default 'year'
    """
    # 检查returns的index是否为datetime index
    assert returns.index.name == 'datetime', 'index of returns must be datetime'
    assert isinstance(returns.index,
                      pd.DatetimeIndex), 'index of returns must be datetime'

    datetimes = returns.index.to_list()

    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    # 如果是series则直接绘制净值曲线
    if isinstance(returns, pd.Series):
        ax.plot(datetimes, (returns + 1).cumprod(), label=returns.name)
    else:
        # 绘制每一列的净值曲线
        for group in returns.columns:
            # 除以forward_period是为了滚仓，比如预测未来5天的收益率，
            # 那么将资金划分成5份，每天的收益率都是当天收益率1/5
            net_value = (returns[group] / forward_period + 1).cumprod()
            # 后移forward_period来将某天的forward_return变成当天的实际return
            net_value = net_value.shift(forward_period)
            ax.plot(datetimes, net_value, label=group)

    # 设置x轴刻度
    if xtick_freq in ['year', 'month']:
        _set_xticks(ax, xtick_freq)
    else:
        raise ValueError(
            f'xtick_freq must be one of ["year", "month"], but got {xtick_freq}'
        )

    # 设置标题、保存路径、图例、网格
    start_date = datetimes[0].strftime('%Y-%m-%d')
    end_date = datetimes[-1].strftime('%Y-%m-%d')
    title = f'Net Value: ({start_date} ~ {end_date})'
    save_path = os.path.join(fig_dir, f'{fig_name}.png')
    plt.legend(loc="upper left")
    plt.grid()
    plt.title(title)
    plt.tight_layout()  # 防止x轴刻度被裁剪
    plt.savefig(save_path)
    plt.close()


def plot_ic_series(
        IC_series: pd.Series,
        fig_dir: str,
        fig_name: str,
        fig_size: Tuple[int, int] = (16, 7),
        dpi: int = 150,
):
    """
    绘制IC序列

    Parameters
    ----------
    IC_series : pd.Series
        IC序列
    fig_dir : str
        图片保存文件夹路径，e.g. 'factor_analysis/report/figs'
    fig_name : str
        图片名称，e.g. 'ic'
    fig_size : Tuple[int, int], optional
        图片大小, by default (16, 7)
    dpi : int, optional
        图片dpi, by default 150
    """
    # 检查IC_series的index是否为datetime index
    assert IC_series.index.name == 'datetime', 'index of IC_series must be datetime'
    assert isinstance(IC_series.index,
                      pd.DatetimeIndex), 'index of IC_series must be datetime'

    datetimes = IC_series.index.to_list()

    # 计算周度IC（日度IC画出来过于密集）
    weekly_IC_series = IC_series.resample('W').mean()

    # 画周度IC
    fig, ax1 = plt.subplots(figsize=fig_size, dpi=dpi)
    ax1.bar(weekly_IC_series.index,
            weekly_IC_series,
            label='weekly IC mean',
            width=-5,
            align='edge')
    ax1.set_ylabel('IC')
    ax1.set_xlabel('datetime')

    # 画IC平均值
    ax1.axhline(y=IC_series.mean(),
                color='purple',
                linestyle='--',
                linewidth=2,
                label='IC mean')
    handles, labels = plt.gca().get_legend_handles_labels()

    # 调整legend顺序并显示legend
    order = [1, 0]
    ax1.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
               loc="upper left")

    # 在y=0处画一条水平线
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)

    # 画IC累计值
    ax2 = ax1.twinx()
    ax2.plot(datetimes,
             IC_series.cumsum(),
             label='daily IC cumsum',
             color='red')
    ax2.set_ylabel('IC cumsum')
    ax2.legend(loc="upper right")

    plt.grid()  # 显示网格

    # 计算IC的均值、标准差、IR、胜率并以图例的形式显示
    ax3 = ax1.twinx()
    ax3.set_yticklabels([])
    IC_mean = IC_series.mean()
    IC_std = IC_series.std()
    IC_IR = IC_mean / IC_std

    if IC_mean > 0:
        IC_win_rate = (IC_series > 0).sum() / len(IC_series)
        IC_win_rate_label = f'IC (+)rate: {IC_win_rate:.4f}'
    elif IC_mean < 0:
        IC_win_rate = (IC_series < 0).sum() / len(IC_series)
        IC_win_rate_label = f'IC (-)rate:  {IC_win_rate:.4f}'
    else:
        IC_win_rate = (IC_series == 0).sum() / len(IC_series)
        IC_win_rate_label = f'IC (0)rate: {IC_win_rate:.4f}'

    IC_mean = f' {IC_mean:.4f}' if IC_mean >= 0 else f'{IC_mean:.4f}'
    IC_std = f' {IC_std:.4f}' if IC_std >= 0 else f'{IC_std:.4f}'
    IC_IR = f' {IC_IR:.4f}' if IC_IR >= 0 else f'{IC_IR:.4f}'

    IC_mean_legend = plt.Line2D([], [],
                                color='black',
                                linestyle='',
                                label=f'IC mean:  {IC_mean}        ')
    IC_std_legend = plt.Line2D([], [],
                               color='black',
                               linestyle='',
                               label=f'IC std:      {IC_std}        ')
    ICIR_legend = plt.Line2D([], [],
                             color='black',
                             linestyle='',
                             label=f'IC IR:        {IC_IR}        ')
    IC_win_rate_legend = plt.Line2D(
        [],
        [],
        color='black',
        linestyle='',
        label=IC_win_rate_label,
    )

    # 添加图例到图中
    plt.legend(handles=[
        IC_mean_legend, IC_std_legend, ICIR_legend, IC_win_rate_legend
    ],
               loc='upper center')

    # 设置标题、保存图片
    start_date = datetimes[0].strftime('%Y-%m-%d')
    end_date = datetimes[-1].strftime('%Y-%m-%d')
    title = f'IC: ({start_date} ~ {end_date})'
    save_path = os.path.join(fig_dir, f'{fig_name}.png')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_turnover(
    turnover_df: pd.DataFrame,
    fig_dir: str,
    fig_name: str,
    fig_size: Tuple[int, int] = (16, 5),
    dpi: int = 150,
    xtick_freq: str = 'year',
    MA_window: int = 20,
):
    """
    画换手率图

    Parameters
    ----------
    turnover_df : pd.DataFrame
        每一列为一个组合的换手率时间序列
    fig_dir : str
        图片保存文件夹路径，e.g. 'factor_analysis/report/figs'
    fig_name : str
        图片名称，e.g. 'turnover'
    fig_size : Tuple[int, int], optional
        图片大小，by default (16, 5)
    dpi : int, optional
        图片分辨率，by default 150
    xtick_freq : str, optional
        x轴刻度频率，by default 'year'
    MA_window : int, optional
        换手率的移动平均窗口大小，by default 20
    """
    # 检查turnover_df的index是否为datetime
    assert turnover_df.index.name == 'datetime', 'index of turnover_df must be datetime'
    assert isinstance(
        turnover_df.index,
        pd.DatetimeIndex), 'index of turnover_df must be datetime'

    datetimes = turnover_df.index.to_list()

    n_groups = len(turnover_df.columns)

    # 画每组的换手率，竖着将若干组的换手率柱状图组合在一起
    subplots = []
    for i in range(n_groups):

        ax = plt.subplot(n_groups, 1, i + 1)
        # 绘制日换手率柱状图
        ax.bar(datetimes,
               turnover_df.iloc[:, i],
               label=turnover_df.columns[i],
               align='center')

        # 绘制换手率的移动平均线
        turnover_MA = turnover_df.iloc[:, i].rolling(MA_window).apply(
            _non_nan_mean, raw=True)
        ax.plot(datetimes,
                turnover_MA,
                label=f'{MA_window}-day MA',
                color='red')
        ax.set_ylabel('turnover')

        # 设置图例显示顺序
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [1, 0]
        ax.legend([handles[idx] for idx in order],
                  [labels[idx] for idx in order],
                  loc='upper left')

        # 设置x轴的刻度
        _set_xticks(ax, xtick_freq)

        # 设置x轴的刻度标签旋转90度
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)

        ax.grid()
        subplots.append(ax)

        # 添加每组的换手率统计信息（换手率非0的天数、换手率均值、换手率标准差）
        turnover_series = turnover_df.iloc[:, i]
        turnover_series = turnover_series[turnover_series != 0]
        turnover_count = turnover_series.count()
        turnover_mean = turnover_series.mean()
        turnover_std = turnover_series.std()

        turnover_count = f' {turnover_count:.0f}' if turnover_count >= 0 else f'{turnover_count:.0f}'
        turnover_mean = f' {turnover_mean:.4f}' if turnover_mean >= 0 else f'{turnover_mean:.4f}'
        turnover_std = f' {turnover_std:.4f}' if turnover_std >= 0 else f'{turnover_std:.4f}'
        turnover_count = f'turnover count: {turnover_count}       '
        turnover_mean = f'turnover mean: {turnover_mean}       '
        turnover_std = f'turnover std:     {turnover_std}       '

        # 添加第二个legend到图中
        ax2 = ax.twinx()
        ax2.legend(
            handles=[
                plt.Line2D([], [],
                           color='black',
                           linestyle='',
                           label=turnover_count),
                plt.Line2D([], [],
                           color='black',
                           linestyle='',
                           label=turnover_mean),
                plt.Line2D([], [],
                           color='black',
                           linestyle='',
                           label=turnover_std)
            ],
            loc='upper right',
        )

        if i == 0:
            # 设置整个图的标题
            start_date = datetimes[0].strftime('%Y-%m-%d')
            end_date = datetimes[-1].strftime('%Y-%m-%d')
            title = f'Turnover: ({start_date} ~ {end_date})'
            plt.title(title)

    # 设置子图之间的间隔
    plt.subplots_adjust(hspace=0.5)

    # 设置整个图的大小
    fig_size = (fig_size[0], fig_size[1] * n_groups)
    plt.gcf().set_size_inches(fig_size)

    plt.tight_layout()

    # 保存图
    save_path = os.path.join(fig_dir, f'{fig_name}.png')
    plt.savefig(save_path, dpi=dpi)
    plt.close()


if __name__ == '__main__':
    import pandas as pd
    from datetime import timedelta

    np.random.seed(0)
    datetimes = [datetime(2010, 1, 1) + timedelta(days=i) for i in range(2500)]
    returns = pd.DataFrame(
        np.random.randn(2500, 3) / 100 + 0.01,
        index=datetimes,
        columns=['group1', 'group2', 'group3'],
    )
    returns.index.name = 'datetime'

    plot_net_value(
        returns=returns['group1'],
        forward_period=1,
        fig_dir='/home/shaoshijie/factor_analysis_project/test',
        fig_name='net_value',
    )

    datetimes = [datetime(2010, 1, 1) + timedelta(days=i) for i in range(2500)]
    IC_series = pd.Series(
        np.random.randn(2500) / 100,
        index=datetimes,
        name='IC',
    )
    IC_series.index.name = 'datetime'

    plot_ic_series(
        IC_series=IC_series,
        fig_dir='/home/shaoshijie/factor_analysis_project/test',
        fig_name='IC',
    )

    datetimes = [datetime(2010, 1, 1) + timedelta(days=i) for i in range(2500)]
    turnover_df = pd.DataFrame(
        np.random.randn(2500, 5) / 100,
        index=datetimes,
        columns=['group1', 'group2', 'group3', 'group4', 'group5'],
    )
    turnover_df.index.name = 'datetime'

    plot_turnover(
        turnover_df=turnover_df,
        fig_dir='/home/shaoshijie/factor_analysis_project/test',
        fig_name='turnover',
    )
