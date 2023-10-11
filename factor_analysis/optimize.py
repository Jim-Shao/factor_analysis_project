#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@Author  :   shaoshijie
@File    :   optimize.py
@Description    :   对因子进行组合优化
'''

import os
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Process, Queue, Manager
from itertools import combinations
from typing import List, Literal

from factor_analysis.factor import Factor


def get_linear_factor_name(
    coef_list: List[float],
    factor_list: List[str],
):
    expression_parts = []

    for coeff, variable in zip(coef_list, factor_list):
        if coeff >= 0:
            expression_parts.append(f"+{coeff}*{variable}")
        else:
            expression_parts.append(f"{coeff}*{variable}")

    final_expression = ''.join(expression_parts)

    if final_expression.startswith('+'):
        final_expression = final_expression[1:]  # 去掉开头的多余'+'

    return final_expression


def _analyze_quantile_shared(factor: Factor, shared_list: List[Factor],
                             queue: Queue):
    """多进程实现Factor.analyze_quantile()的辅助函数，用于将Factor对象存入共享列表中
    ，并将进程号存入队列中"""
    factor.analyze_quantile()
    shared_list.append(factor)
    queue.put(os.getpid())


def mp_analyze_quantile(
    factor_list: List[Factor],
    n_jobs: int,
    desc: str,
) -> List[Factor]:
    """多进程实现Factor.analyze_quantile()"""
    processes: List[Process] = []  # 进程列表，用于存储正在运行的进程
    queue = Queue()  # 进程队列，用于存储已完成的进程

    # 多进程时，使用共享变量来存储因子分析结果
    manager = Manager()
    shared_factor_list = manager.list()

    tqdm_obj = tqdm(total=len(factor_list), desc=desc)

    for factor in factor_list:
        p = Process(target=_analyze_quantile_shared,
                    args=(factor, shared_factor_list, queue))

        # 当进程数达到最大值时，等待其中一个进程结束并移除
        if len(processes) == n_jobs:
            completed_process_pid = queue.get()
            for process in processes:
                if process.pid == completed_process_pid:
                    process.join()
                    processes.remove(process)
                    tqdm_obj.update()

        # 在进程数未达到最大值时，直接加入新进程
        processes.append(p)
        p.start()

    # 等待所有进程结束
    for process in processes:
        process.join()
        tqdm_obj.update()

    tqdm_obj.close()
    factor_list = list(shared_factor_list)
    return factor_list


class FactorOptimizer:
    def __init__(
        self,
        factor_list: List[Factor],
        universe: str,
        analyze_fields: List[str],
        forward_return_df: pd.DataFrame,
        bench_return_df: pd.DataFrame,
        group_periodic_net_values: pd.DataFrame,
        periodic_net_values: pd.DataFrame,
        n_groups: int,
        n_jobs: int,
        output_dir: str,
        position_adjust_datetimes: List[str],
    ):
        """因子优化类，目前实现了等权优化方法，后续可添加其他优化方法；
        参数详见Factor类的__init__方法
        """
        # 两两组合
        factor_names = [factor.name for factor in factor_list]
        factor_values = [factor.factor_series for factor in factor_list]
        factor_value_dict = dict(zip(factor_names, factor_values))
        self.factor_names = factor_names
        self.factor_value_dict = factor_value_dict

        self.factor_list = factor_list
        self.universe = universe
        self.analyze_fields = analyze_fields
        self.forward_return_df = forward_return_df
        self.bench_return_df = bench_return_df
        self.group_periodic_net_values = group_periodic_net_values
        self.periodic_net_values = periodic_net_values
        self.n_groups = n_groups
        self.n_jobs = n_jobs
        self.output_dir = output_dir
        self.position_adjust_datetimes = position_adjust_datetimes

    def equal_weight_optimize(self,
                              row: str = 'excess',
                              col: str = 'sharpe') -> None:
        """对因子进行优化"""
        self.combine_2_factors(row, col)
        if self.is_improved:
            self.combine_n_factors(row, col)
        self.analyze_best_comb()

    def optimize(
        self,
        row: str = 'excess',
        col: str = 'sharpe',
        mode: Literal['linear', 'lasso', 'ridge'] = 'linear',
        train_end_date: str = '2019-12-31',
        forward_period: int = 20,
    ):
        """对因子进行优化"""
        self.equal_weight_optimize(row, col)
        # self.linear_optimize(mode, train_end_date, forward_period)

    def combine_2_factors(
        self,
        row: str = 'excess',
        col: str = 'sharpe',
    ):
        """从所有因子中，选出最优的两个因子进行组合"""
        old_performance_list = [
            factor.quantile_return_performance.loc[row, col]
            for factor in self.factor_list
        ]
        old_best_idx = np.argmax(old_performance_list)
        old_best_performance = old_performance_list[old_best_idx]
        old_best_combination = self.factor_list[old_best_idx]
        factor_name_comb = list(combinations(self.factor_names, 2))

        # 计算两两组合的因子
        factor_list: List[Factor] = []
        for factor1_name, factor2_name in factor_name_comb:
            factor1_value = self.factor_value_dict[factor1_name]
            factor2_value = self.factor_value_dict[factor2_name]
            factor_name_comb = factor1_name + '+' + factor2_name  # 因子组合
            factor_value_comb = factor1_value + factor2_value  # 因子组合
            factor = Factor(
                name=factor_name_comb,
                universe=self.universe,
                fields=self.analyze_fields,
                factor_series=factor_value_comb,
                forward_return_df=self.forward_return_df,
                bench_forward_return_df=self.bench_return_df,
                group_periodic_net_values=self.group_periodic_net_values,
                periodic_net_values=self.periodic_net_values,
                n_group=self.n_groups,
                output_dir=self.output_dir,
                position_adjust_datetimes=self.position_adjust_datetimes,
            )
            factor_list.append(factor)

        # 分析优化因子（不需要分层回测和计算IC，只需要查看单因子策略的超额收益表现）
        if self.n_jobs == 1:
            for factor in tqdm(factor_list, desc='Combining 2 factors'):
                factor.analyze_quantile()
        else:
            factor_list = mp_analyze_quantile(factor_list, self.n_jobs,
                                              'Combining 2 factors')

        # 两两组合的因子分析完毕，开始比较两两组合的因子的超额收益表现
        performance_list = [
            factor.quantile_return_performance.loc[row, col]
            for factor in factor_list
        ]
        comp_factor_names = [factor.name for factor in factor_list]
        done_combinations = dict(zip(comp_factor_names, performance_list))
        best_combination = max(done_combinations, key=done_combinations.get)
        best_performance = done_combinations[best_combination]

        # 比较两两组合的因子的超额收益表现与单因子的超额收益表现
        if best_performance > max(old_performance_list):
            print(
                f"(n=2) Best {row} {col}: {best_performance:.4f}, {best_combination}"
            )
            is_improved = True
        else:
            # print(
            #     f"(n=2) No improvement, Best {row} {col}: {old_best_performance:.4f}, {old_best_combination.name}"
            # )
            print("(n=2) No improvement")
            best_combination = old_best_combination.name
            best_performance = old_best_performance
            is_improved = False

        self.best_combination = best_combination
        self.best_performance = best_performance
        self.is_improved = is_improved

    def combine_n_factors(
        self,
        row: str = 'excess',
        col: str = 'sharpe',
    ) -> None:
        """在选出的最优因子组合基础上，逐个添加剩下的因子，生成三因子、四因子、...组合，
        并比较新组合的因子的超额收益表现与当前最优组合的因子的超额收益表现，当新组合的因子的
        超额收益表现优于当前最优组合的因子的超额收益表现时，更新最优组合"""
        factor_list: List[Factor] = []

        # 从剩下的因子中，逐个添加到最优的两个因子中，生成三因子、四因子、...组合
        best_combination = self.best_combination.split('+')
        rest_factor_names = list(
            set(self.factor_names) - set(best_combination))
        if len(rest_factor_names) == 0:
            return

        # for n in range(3, len(self.factor_list) + 1):
        for factor_name in rest_factor_names:
            combination = best_combination + [factor_name]
            factor_combination = sum(
                [self.factor_value_dict[factor] for factor in combination])
            factor = Factor(
                name='+'.join(combination),
                universe=self.universe,
                fields=self.analyze_fields,
                factor_series=factor_combination,
                forward_return_df=self.forward_return_df,
                bench_forward_return_df=self.bench_return_df,
                group_periodic_net_values=self.group_periodic_net_values,
                periodic_net_values=self.periodic_net_values,
                n_group=self.n_groups,
                output_dir=self.output_dir,
                position_adjust_datetimes=self.position_adjust_datetimes,
            )
            factor_list.append(factor)

        n = len(best_combination) + 1
        # 对新组合的因子进行分析
        if self.n_jobs == 1:
            for factor in tqdm(factor_list, desc=f'Combining {n} factors'):
                factor.analyze_quantile()
        else:
            factor_list = mp_analyze_quantile(factor_list, self.n_jobs,
                                              f'Combining {n} factors')

        # 比较新组合的因子的超额收益表现与当前最优组合的因子的超额收益表现
        performance_list = [
            factor.quantile_return_performance.loc[row, col]
            for factor in factor_list
        ]
        best_performance_idx = np.argmax(performance_list)
        best_performance = performance_list[best_performance_idx]
        best_performance_name = factor_list[best_performance_idx].name

        # 如果新组合的因子的超额收益表现优于当前最优组合的因子的超额收益表现，则更新最优组合
        if best_performance > self.best_performance:
            pop_name = best_performance_name.split('+')[-1]
            pop_idx = [
                i for i, name in enumerate(rest_factor_names)
                if name == pop_name
            ][0]
            best_combination = tuple(best_combination) + (
                rest_factor_names.pop(pop_idx), )
            best_combination = '+'.join(best_combination)
            print(
                f"(n={n}) Best {row} {col}: {best_performance:.4f}, {best_combination}"
            )
            self.best_performance = best_performance
            self.best_combination = best_combination

            self.combine_n_factors(row, col)
        # 如果新组合的因子的超额收益表现不优于当前最优组合的因子的超额收益表现，则停止添加新因子
        else:
            best_combination = '+'.join(best_combination)
            # print(
            #     f"(n={n}) No improvement, Best {row} {col}: {self.best_performance:.4f}, {self.best_combination}"
            # )
            print(f"(n={n}) No improvement")

    def analyze_best_comb(self):
        """对最优组合的因子进行全面分析（包括因子IC、收益率、换手率等）"""
        start_time = time.time()
        if '+' not in self.best_combination:
            return
        print('>>> Analyzing best factor combination:', end=' ')
        sys.stdout.flush()
        # 对最优组合的因子进行全面分析（包括因子IC、收益率、换手率等）
        best_factor = Factor(
            name=self.best_combination,
            universe=self.universe,
            fields=self.analyze_fields,
            factor_series=sum([
                self.factor_value_dict[factor]
                for factor in self.best_combination.split('+')
            ]),
            forward_return_df=self.forward_return_df,
            bench_forward_return_df=self.bench_return_df,
            group_periodic_net_values=self.group_periodic_net_values,
            periodic_net_values=self.periodic_net_values,
            n_group=self.n_groups,
            output_dir=self.output_dir,
            position_adjust_datetimes=self.position_adjust_datetimes,
        )
        best_factor.analyze()
        end_time = time.time()
        print(f'Done in {end_time - start_time:.2f}s')

    def linear_optimize(
        self,
        mode: Literal['linear', 'lasso', 'ridge'] = 'linear',
        train_end_date: str = '2019-12-31',
        forward_period: int = 20,
        row: str = 'excess',
        col: str = 'sharpe',
    ):
        """对最优组合的因子进行线性组合，生成新的因子"""
        print('>>> Linearly combining best factor combination:')
        sys.stdout.flush()

        # 训练集和测试集划分
        train_end_date = pd.Timestamp(train_end_date)

        # 对最优组合的因子进行线性组合，生成新的因子
        forward_return = self.forward_return_df.loc[:, [f'{forward_period}D']]
        factor_values = pd.DataFrame(self.factor_value_dict)
        merged_df = pd.merge(forward_return,
                             factor_values,
                             left_index=True,
                             right_index=True)
        merged_df_dropna = merged_df.dropna()
        # forward_return_dropna = forward_return.loc[merged_df_dropna.index]

        train_df = merged_df.loc[
            merged_df.index.get_level_values(0) <= train_end_date, :]
        test_df = merged_df.loc[
            merged_df.index.get_level_values(0) > train_end_date, :]
        train_df_dropna = merged_df_dropna.loc[
            merged_df_dropna.index.get_level_values(0) <= train_end_date, :]
        # test_df_dropna = merged_df_dropna.loc[
        #     merged_df_dropna.index.get_level_values(0) > train_end_date, :]

        X_train = train_df.iloc[:, 1:]
        # y_train = train_df.iloc[:, 0]
        X_test = test_df.iloc[:, 1:]
        # y_test = test_df.iloc[:, 0]
        X_train_dropna = train_df_dropna.iloc[:, 1:]
        y_train_dropna = train_df_dropna.iloc[:, 0]
        # X_test_dropna = test_df_dropna.iloc[:, 1:]
        # y_test_dropna = test_df_dropna.iloc[:, 0]

        if mode == 'linear':
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        elif mode == 'lasso':
            from sklearn.linear_model import LassoCV
            model = LassoCV(alphas=[1e-4, 1e-5, 1e-6, 1e-7])
        elif mode == 'ridge':
            from sklearn.linear_model import RidgeCV
            model = RidgeCV(alphas=[1e-4, 1e-5, 1e-6, 1e-7])
        model.fit(X_train_dropna, y_train_dropna)
        coef = model.coef_

        # 记录系数
        coef_df = pd.DataFrame(coef.reshape(-1, 1),
                               index=X_train.columns,
                               columns=[mode])
        coef_df.to_csv(
            os.path.join(self.output_dir, self.universe,
                         f'{mode}_comb_coef.csv'))

        coef_max = coef_df.abs().max().max()
        coef_df = coef_df / coef_max
        coef_df = coef_df.sort_values(by=coef_df.columns[0],
                                      ascending=False,
                                      key=abs)
        coef_df = coef_df.round(2).T

        comb_train = np.dot(X_train, coef.T)  # 计算新因子的值
        comb_test = np.dot(X_test, coef.T)
        comb_train = pd.Series(comb_train,
                               index=X_train.index,
                               name=f'{mode}_train')
        comb_test = pd.Series(comb_test,
                              index=X_test.index,
                              name=f'{mode}_test')

        forward_return_df_train = forward_return.loc[
            forward_return.index.get_level_values(0) <= train_end_date, :]
        forward_return_df_test = forward_return.loc[
            forward_return.index.get_level_values(0) > train_end_date, :]
        bench_return_df_train = self.bench_return_df.loc[
            self.bench_return_df.index.get_level_values(0
                                                        ) <= train_end_date, :]
        bench_return_df_test = self.bench_return_df.loc[
            self.bench_return_df.index.get_level_values(0) > train_end_date, :]
        group_periodic_net_values_train = self.group_periodic_net_values[
            forward_period].loc[self.group_periodic_net_values[forward_period].
                                index <= train_end_date, :]
        group_periodic_net_values_test = self.group_periodic_net_values[
            forward_period].loc[self.group_periodic_net_values[forward_period].
                                index > train_end_date, :]
        periodic_net_values_train = self.periodic_net_values.loc[
            self.periodic_net_values.index <= train_end_date, :]
        periodic_net_values_test = self.periodic_net_values.loc[
            self.periodic_net_values.index > train_end_date, :]
        position_adjust_datetimes_train = [
            datetime for datetime in self.position_adjust_datetimes
            if datetime <= train_end_date
        ]
        position_adjust_datetimes_test = [
            datetime for datetime in self.position_adjust_datetimes
            if datetime > train_end_date
        ]

        coef_list = coef_df.iloc[0, :].tolist()
        factor_list = coef_df.columns.tolist()
        factor_name = get_linear_factor_name(coef_list, factor_list)
        if mode == 'linear':
            prefix = f'({mode} regression)'
        if mode in ['lasso', 'ridge']:
            prefix = f'({mode} regression, alpha={model.alpha_})'
        print(f'{prefix} {factor_name}')

        # 分析新因子
        for split in ['train', 'test']:
            factor = Factor(
                name=f'{mode}_{split}',
                universe=self.universe,
                fields=self.analyze_fields,
                factor_series=eval(f'comb_{split}'),
                forward_return_df=eval(f'forward_return_df_{split}'),
                bench_forward_return_df=eval(f'bench_return_df_{split}'),
                group_periodic_net_values={
                    forward_period: eval(f'group_periodic_net_values_{split}')
                },
                periodic_net_values=eval(f'periodic_net_values_{split}'),
                n_group=self.n_groups,
                output_dir=self.output_dir,
                position_adjust_datetimes=eval(
                    f'position_adjust_datetimes_{split}'),
            )
            factor.analyze()

            print(
                f'({split}) Best {row} {col}: {factor.quantile_return_performance.loc[row, col]:.4f}'
            )
