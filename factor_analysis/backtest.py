#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@Author  :   shaoshijie
@File    :   backtest.py
@Description    :   因子回测类
'''

import os
import sys
import time
import json
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from multiprocessing import Manager, Process, Queue
from typing import Union, List, Dict, Tuple
from tqdm import tqdm

from factor_analysis.utils import assert_std_index, assert_index_equal, calc_return
from factor_analysis.factor import Factor
from factor_analysis.factor_calc import CalcPool
from factor_analysis.postprocess import Postprocess, PostprocessQueue
from factor_analysis.optimize import FactorOptimizer


def _analyze_shared(factor: Factor, shared_list: List[Factor], queue: Queue):
    """多进程实现Factor.analyze()的辅助函数，用于将Factor对象存入共享列表中
    ，并将进程号存入队列中"""
    factor.analyze()
    shared_list.append(factor)
    queue.put(os.getpid())


class FactorBacktest:
    def __init__(
        self,
        universe: str,
        bar_df: pd.DataFrame,
        factor_df: pd.DataFrame = None,
        factor_expressions: Union[List[str], Dict[str, str]] = None,
        benchmark_df: pd.DataFrame = None,
        forward_periods: Union[int, List[int]] = [1, 5, 10, 20],
        position_adjust_datetimes: List[pd.Timestamp] = None,
        postprocess_queue: PostprocessQueue = None,
        choice: Union[Tuple[float, float], int] = (0.8, 1.0),
        analyze_fields: List[str] = ['IC', 'return', 'turnover'],
        output_dir: str = None,
        n_groups: int = 5,
        n_jobs: int = 1,
    ):
        """因子回测类（因子可以直接提供factor_df，也可以经过表达式factor_expressions计算得到）

        Parameters
        ----------
        universe : str
            股票池名称，比如全A股或者沪深300等
        bar_df : pd.DataFrame
            <后复权>行情数据，multi-index[datetime, order_book_id]，columns=['open', 'high', 'low', 'close', 'volume', ...]
            其中除必须包含的开高低收量之外，还可以包含其他任意列，这些列都可用于用于计算因子
        factor_df : pd.DataFrame, optional
            因子数据，multi-index[datetime, order_book_id]，columns=['factor1', 'factor2', ...]，默认为None，但是不能和factor_expressions同时为None
        factor_expressions : Union[List[str], Dict[str, str]], optional
            因子表达式，可以是一个列表，也可以是一个字典；
            列表中的每个元素都是一个因子表达式，例：['(ts_corr(rank(ts_delta(log(volume), 2)), rank(div(diff(close, open), open)), 6))']，会自动命名为factor1、factor2、...；
            字典中的每个键值对都是一个因子表达式，键为因子名称，值为因子表达式，例：{'factor1': '(ts_corr(rank(ts_delta(log(volume), 2)), rank(div(diff(close, open), open)), 6))'}
            关于可以使用哪些函数和变量，请参考factor_analysis.factor_calc文件；
            默认为None，但是不能和factor_df同时为None
        benchmark_df : pd.DataFrame, optional
            基准数据，multi-index[datetime, order_book_id]，columns=['open', 'high', 'low', 'close', 'volume', ...]，默认为None；
            如果为None，表示基准为每天等权持仓所有股票，日度换仓；
        forward_periods : Union[int, List[int]], optional
            前瞻期数，可以是一个整数，也可以是一个整数列表，默认为[1, 5, 10, 20]
        position_adjust_datetimes : List[pd.Timestamp], optional
            调仓日期列表，每个元素为一个pd.Timestamp对象，默认为None，表示每天都调仓；
            可以在当调仓日为某些特定时刻时使用，比如每年的财务日：每年4月、8月、10月最后一天；
        postprocess_queue : PostprocessQueue, optional
            后处理队列，用于对因子数据进行后处理，默认为None，表示不进行后处理
        choice : Union[Tuple[float, float], int], optional
            用于单因子回测的分位数选择，可以是一个元组，也可以是一个整数，默认为(0.8, 1.0)；
            例：(0.8, 1,0)即为每次换仓时，选择因子值处于80%~100%分位数的股票等权买入作为多头；
            当为整数时，表示每次换仓时，选择因子值处于前choice名的股票等权买入作为多头；
        analyze_fields : List[str]
            因子分析时需要分析的部分，可以是['IC', 'return', 'turnover']中的任意组合
        output_dir : str, optional
            回测结果输出目录，默认为None，表示在当前目录下创建一个factor_analysis_output文件夹
        n_groups : int, optional
            分组数，默认为5
        n_jobs : int, optional
            多进程数量，默认为1，表示单进程运行，如果为-1，则使用所有可用的CPU核心数
        """
        assert_std_index(bar_df, 'bar_df', 'df')
        if benchmark_df is not None:
            assert_std_index(benchmark_df, 'benchmark_df', 'df')
        if factor_df is not None:
            assert_std_index(factor_df, 'factor_df', 'df')
            assert_index_equal(bar_df, factor_df, 'bar_df', 'factor_df')

        self.universe = universe
        self.bar_df = bar_df.sort_index()
        self.forward_periods = forward_periods
        self.benchmark_df = benchmark_df
        self.n_groups = n_groups
        self.choice = choice
        self.analyze_fields = analyze_fields
        self.trading_dates = self.bar_df.index.get_level_values(
            'datetime').unique()

        # 因子后处理步骤队列
        if postprocess_queue is not None:
            self.postprocess_queue = postprocess_queue
        else:
            self.postprocess_queue = PostprocessQueue()

        # 检验调仓日是否全都为交易日
        if position_adjust_datetimes is not None:
            position_adjust_datetimes = pd.to_datetime(
                position_adjust_datetimes)
            is_trading_days = position_adjust_datetimes.isin(
                self.trading_dates)
            if not is_trading_days.all():
                error_datetimes = position_adjust_datetimes[
                    ~is_trading_days].tolist()
                raise ValueError(
                    f"position_adjust_datetimes must be trading days, but {error_datetimes} are not"
                )
        else:
            position_adjust_datetimes = self.bar_df.index.get_level_values(
                'datetime').unique().tolist()
        self.position_adjust_datetimes = position_adjust_datetimes

        # 因子数据
        if factor_df is None and factor_expressions is None:
            raise ValueError(
                "factor_df and factor_expressions can't be both None")
        if factor_expressions is None:
            factor_expressions = {}
        if isinstance(factor_expressions, list):
            factor_expressions = {
                f'factor{i}': expression
                for i, expression in enumerate(factor_expressions, 1)
            }
        self._factor_df = factor_df if factor_df is not None else pd.DataFrame(
        )
        self.factor_names: List[str] = self._factor_df.columns.append(
            pd.Index(factor_expressions.keys())).tolist()
        self.factor_expressions = factor_expressions

        # 设定进程数
        if n_jobs == -1:
            self.n_jobs = min(os.cpu_count(), len(self.factor_names))
        elif n_jobs > 0:
            self.n_jobs = min(n_jobs, len(self.factor_names), os.cpu_count())
        else:
            raise ValueError("n_jobs must be -1 or positive integer")

        # 创建文件夹
        if output_dir is None:
            current_dir = os.getcwd()
            output_dir = os.path.join(current_dir, 'factor_analysis_output')
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

        # 用于多进程的共享变量
        self.manager = Manager()
        self.name_space = self.manager.Namespace()

        # 记录参数
        self.config = {}
        self.config['universe'] = self.universe
        self.config['datetime_range'] = (
            self.trading_dates.min().strftime('%Y-%m-%d'),
            self.trading_dates.max().strftime('%Y-%m-%d'))
        self.config['analyze_fields'] = self.analyze_fields
        self.config['forward_periods'] = self.forward_periods
        self.config['choice'] = self.choice
        self.config['n_groups'] = self.n_groups
        self.config['n_jobs'] = self.n_jobs
        self.config['postprocess_queue'] = self.postprocess_queue.info
        self.config['factor_expressions'] = self.factor_expressions
        self.config['position_adjust_datetimes'] = [
            dt.strftime('%Y-%m-%d') for dt in self.position_adjust_datetimes
        ]

        # 导出参数为JSON文件
        self.config_path = os.path.join(self.output_dir, universe,
                                        'config.json')
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)

    def run(self, try_optimize: bool = True, auto_standardize: bool = True):
        """运行回测

        Parameters
        ----------
        try_optimize : bool, optional
            是否尝试优化，如果为True，则会尝试将已有因子进行相加组合，以寻找更好的因子，
            如果为False，则不会进行优化，默认为True
        auto_standardize : bool, optional
            优化时是否自动进行截面标准化，默认为True
        """
        try:
            terminal_width = os.get_terminal_size().columns
        except OSError:
            terminal_width = 80
        text = ' Factor Analysis '
        padding_len = (terminal_width - len(text)) // 2
        print(f'\n{"*"*padding_len}{text}{"*"*padding_len}')
        print(f"### Backtest config: universe={self.universe}, "
              f"datetime_range={self.config['datetime_range']}, "
              f"n_jobs={self.n_jobs}")
        start = time.time()
        if self.factor_expressions:
            self.calc_factors()  # 计算因子
        self.calc_forward_returns()  # 计算前瞻收益
        self.calc_periodic_net_values()  # 计算阶段净值

        self.factor_list: List[Factor] = []
        for factor_name, factor_series in self._factor_df.iteritems():
            factor = Factor(
                name=factor_name,
                universe=self.universe,
                fields=self.analyze_fields,
                factor_series=factor_series,
                forward_return_df=self.forward_return_df,
                bench_forward_return_df=self.bench_return_df,
                group_periodic_net_values=self.group_periodic_net_values,
                periodic_net_values=self.periodic_net_values,
                position_adjust_datetimes=self.position_adjust_datetimes,
                n_group=self.n_groups,
                quantile=self.choice,
                output_dir=self.output_dir,
            )
            self.factor_list.append(factor)
        self.analyze_factors()  # 分析因子

        self.plot_corr()  # 绘制因子相关性热力图

        if try_optimize is True:
            self.optimize_combination(
                auto_standardize=auto_standardize)  # 优化因子组合
        print(
            f">>> Output directory: {os.path.join(self.output_dir, self.universe)}"
        )
        end = time.time()
        print(f">>> Total time for running backtest: {end-start:.2f}s")
        print(f'{"*"*terminal_width}\n')

    @property
    def factor_df(self) -> pd.DataFrame:
        """因子数据"""
        if self._factor_df.columns.tolist() == self.factor_names:
            return self._factor_df
        else:
            self.calc_factors()
            return self._factor_df.copy()

    def calc_periodic_net_values(self):
        """计算阶段净值"""
        print(">>> Calculating periodic net values:", end=' ')
        sys.stdout.flush()
        start_time = time.time()

        self.group_periodic_net_values = {}
        for period in self.forward_periods:
            position_adjust_datetimes = self.trading_dates[::period]
            returns = self.forward_return_df['1D'].unstack()
            returns.loc[position_adjust_datetimes,
                        'adjust_datetime'] = position_adjust_datetimes
            returns['adjust_datetime'] = returns['adjust_datetime'].ffill(
            ).fillna(returns.index[0])
            periodic_net_values = returns.groupby('adjust_datetime').apply(
                lambda x: (x.drop(columns='adjust_datetime') + 1).cumprod())
            # 最后一天用倒数第二天填充，因为最后一天的收益是无法计算的
            periodic_net_values.iloc[-1] = periodic_net_values.iloc[-2]
            self.group_periodic_net_values[period] = periodic_net_values

        returns = self.forward_return_df['1D'].unstack()
        returns.loc[self.position_adjust_datetimes,
                    'adjust_datetime'] = self.position_adjust_datetimes
        returns['adjust_datetime'] = returns['adjust_datetime'].ffill().fillna(
            returns.index[0])
        periodic_net_values = returns.groupby('adjust_datetime').apply(
            lambda x: (x.drop(columns='adjust_datetime') + 1).cumprod())
        # 最后一天用倒数第二天填充，因为最后一天的收益是无法计算的
        periodic_net_values.iloc[-1] = periodic_net_values.iloc[-2]
        self.periodic_net_values = periodic_net_values

        end_time = time.time()
        print(f"Done in {end_time-start_time:.2f}s")

    def calc_factors(self):
        """计算因子"""
        calc_pool = CalcPool(
            df=self.bar_df,
            expressions=self.factor_expressions,
            n_jobs=1,
        )
        new_factor_df = calc_pool.calc_factors()
        factor_df = pd.concat([self._factor_df, new_factor_df],
                              axis=1,
                              sort=False)
        if self.postprocess_queue is not None:
            factor_df = self.postprocess_queue(factor_df)
        self._factor_df = factor_df

    def calc_forward_returns(self):
        """计算前瞻收益"""
        print(">>> Calculating forward returns:", end=' ')
        sys.stdout.flush()

        # 如果forward_periods中不包含1，则将1加入到forward_periods中
        if 1 not in self.forward_periods:
            forward_periods = [1] + self.forward_periods
        else:
            forward_periods = self.forward_periods

        start_time = time.time()
        forward_return_df = calc_return(self.bar_df, forward_periods)
        forward_return_df = forward_return_df.loc[self.factor_df.index]
        self.forward_return_df = forward_return_df
        if self.benchmark_df is not None:
            self.bench_return_df = calc_return(self.benchmark_df,
                                               forward_periods)
        else:
            self.bench_return_df = forward_return_df.groupby(
                level='datetime').mean()
            self.bench_return_df.index = pd.MultiIndex.from_product(
                [self.bench_return_df.index, ['equal_weight']],
                names=['datetime', 'order_book_id'])
        end_time = time.time()
        print(f"Done in {end_time-start_time:.2f}s")

    def analyze_factors(self) -> None:
        """分析所有因子"""
        if self.n_jobs == 1:
            for factor in tqdm(self.factor_list, desc='>>> Analyzing factors'):
                factor.analyze()
        else:
            processes: List[Process] = []  # 进程列表，用于存储正在运行的进程
            queue = Queue()  # 进程队列，用于存储已完成的进程
            shared_factor_list = Manager().list()  # 用于多进程的共享变量

            tqdm_obj = tqdm(total=len(self.factor_list),
                            desc='>>> Analyzing factors')

            for factor in self.factor_list:
                p = Process(target=_analyze_shared,
                            args=(factor, shared_factor_list, queue))

                # 当进程数达到最大值时，等待其中一个进程结束并移除
                if len(processes) == self.n_jobs:
                    completed_process_pid = queue.get()  # 等待进程基本结束并获取其pid
                    for process in processes:
                        if process.pid == completed_process_pid:
                            process.join()  # 等待进程完全结束
                            processes.remove(process)  # 移除已完成的进程
                            tqdm_obj.update()

                # 在进程数未达到最大值时，直接加入新进程
                processes.append(p)
                p.start()

            # 等待所有进程结束
            for process in processes:
                process.join()
                tqdm_obj.update()
            tqdm_obj.close()

            self.factor_list = list(shared_factor_list)

    def plot_corr(self) -> None:
        """绘制因子相关性热力图"""
        start = time.time()
        print(">>> Plotting factor correlation:", end=' ')
        sys.stdout.flush()
        factor_corr = self.factor_df.corr()
        plt.figure(figsize=(16, 13), dpi=150)
        ax = sns.heatmap(factor_corr,
                         annot=True,
                         cmap='RdBu_r',
                         linewidths=0.2,
                         vmin=-1,
                         vmax=1,
                         cbar_kws={'use_gridspec': False})
        ytick_labels = ax.get_yticklabels()
        ax.set_yticklabels(ytick_labels, va='center')
        plt.title('Factor Correlation')
        plt.savefig(
            os.path.join(self.output_dir, self.universe, 'factor_corr.png'))
        plt.savefig(
            os.path.join(self.output_dir, self.universe, 'factor_corr.svg'))
        plt.close()
        end = time.time()
        print(f"Done in {end-start:.2f}s")

    def optimize_combination(
        self,
        compare_method: List[str] = ['excess', 'sharpe'],
        auto_standardize: bool = True,
    ) -> None:
        """优化因子组合（比较超额收益的夏普值）：先两两组合，选出最优组合，比较相对于最优的单个因子是否有边际提升，
        如果有提升，再尝试向组合中逐个添加新因子至三个因子，比较三个因子的组合相对于两个因子的组合是否有边际提升，
        依次类推，直到没有边际提升为止"""

        factor_names = self.factor_names
        if len(factor_names) == 1:
            print(">>> Only one factor, skip optimizing factor combination")
            return
        print(">>> Optimizing factor combination:")

        # 截面标准化，使得不同因子在截面上具有可加性
        if auto_standardize:
            factor_addable = False
            for postprocess in self.postprocess_queue.queue:
                if postprocess.func in [
                        Postprocess.standardize, Postprocess.normalize
                ]:
                    factor_addable = True
                    break
            if not factor_addable:
                factor_list = []
                for factor in self.factor_list:
                    factor = copy.copy(factor)
                    factor.factor_series = Postprocess.standardize(
                        factor.factor_series)
                    factor_list.append(factor)
            else:
                factor_list = self.factor_list
        else:
            factor_list = self.factor_list

        # 得到单因子的超额收益表现
        old_sharpe_list = [
            factor.quantile_return_performance.loc[compare_method[0],
                                                   compare_method[1]]
            for factor in self.factor_list
        ]
        old_best = np.argmax(old_sharpe_list)
        old_best_combination = self.factor_list[old_best]
        old_best_sharpe = old_sharpe_list[old_best]
        print(
            f'(n=1) Best {compare_method[0]} {compare_method[1]}: {old_best_sharpe:.4f}, {old_best_combination.name}'
        )

        factor_optimizer = FactorOptimizer(
            factor_list=factor_list,
            universe=self.universe,
            analyze_fields=self.analyze_fields,
            forward_return_df=self.forward_return_df,
            bench_return_df=self.bench_return_df,
            group_periodic_net_values=self.group_periodic_net_values,
            periodic_net_values=self.periodic_net_values,
            n_groups=self.n_groups,
            n_jobs=self.n_jobs,
            output_dir=self.output_dir,
            position_adjust_datetimes=self.position_adjust_datetimes)

        factor_optimizer.optimize(row='excess', col='sharpe')
