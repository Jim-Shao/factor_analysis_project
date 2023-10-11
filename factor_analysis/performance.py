#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@Author  :   shaoshijie
@File    :   performance.py
@Description    :   计算收益率表现
'''

import numpy as np
import warnings

from functools import cached_property

warnings.filterwarnings('ignore')

YEAR = 'year'
MONTH = 'month'
WEEK = 'week'
DAY = 'day'

FREQUENCY_LIST = [YEAR, MONTH, WEEK, DAY]

MONTHS_PER_YEAR = 12
WEEKS_PER_YEAR = 52
DAYS_PER_YEAR = 252

ANNUALIZATION_FACTORS = {
    YEAR: 1,
    MONTH: MONTHS_PER_YEAR,
    WEEK: WEEKS_PER_YEAR,
    DAY: DAYS_PER_YEAR,
}

ANNUAL_RISK_FREE_RATE = 0
RISK_FREE_RATES = {
    YEAR: ANNUAL_RISK_FREE_RATE,
    MONTH: (1 + ANNUAL_RISK_FREE_RATE)**(1 / MONTHS_PER_YEAR) - 1,
    WEEK: (1 + ANNUAL_RISK_FREE_RATE)**(1 / WEEKS_PER_YEAR) - 1,
    DAY: (1 + ANNUAL_RISK_FREE_RATE)**(1 / DAYS_PER_YEAR) - 1,
}


def calc_return(returns: np.ndarray) -> float:
    return np.expm1(np.log1p(returns).sum())


def calc_annual_return(
    returns: np.ndarray,
    frequency: str = DAY,
) -> float:
    return_rate = calc_return(returns)
    period_count = len(returns)
    annual_factor = ANNUALIZATION_FACTORS[frequency]
    return (1 + return_rate)**(annual_factor / period_count) - 1


def calc_sharpe(
    returns: np.ndarray,
    risk_free_rate: float = 0,
) -> float:
    mean_excess_return = np.mean(returns) - risk_free_rate
    std_excess_return = np.std(returns, ddof=1)
    return mean_excess_return / std_excess_return


def calc_annual_sharpe(
    returns: np.ndarray,
    risk_free_rate: float = 0,
    frequency: str = DAY,
) -> float:
    sharpe = calc_sharpe(returns, risk_free_rate)
    annual_factor = ANNUALIZATION_FACTORS[frequency]
    return sharpe * np.sqrt(annual_factor)


def calc_win_rate(returns: np.ndarray) -> float:
    n_win = len(returns[returns > 0])
    n_loss = len(returns[returns < 0])
    return n_win / (n_win + n_loss)


def calc_tracking_win_rate(
    returns: np.ndarray,
    bench_returns: np.ndarray,
) -> float:
    return calc_win_rate(returns - bench_returns)


def calc_profit_loss_ratio(returns: np.ndarray) -> float:
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]
    if len(positive_returns) == 0:
        return np.nan
    if len(negative_returns) == 0:
        return np.inf
    return abs(np.mean(positive_returns) / np.mean(negative_returns))


def calc_max_drawdown(returns: np.ndarray) -> float:
    net_values = np.insert(np.exp(np.log1p(returns).cumsum(), 0, 1))
    cummax_net_values = np.maximum.accumulate(net_values)
    return np.abs(np.min(net_values / cummax_net_values - 1))


def calc_information_ratio(
    returns: np.ndarray,
    bench_returns: np.ndarray,
) -> float:
    tracking_returns = returns - bench_returns
    if len(tracking_returns) < 2:
        return np.nan
    mean_tracking_return = np.mean(tracking_returns)
    tracking_error = np.std(tracking_returns, ddof=1)
    if tracking_error == 0:
        return np.nan
    information_ratio = mean_tracking_return / tracking_error
    return information_ratio


def calc_annual_information_ratio(
    returns: np.ndarray,
    bench_returns: np.ndarray,
    frequency: str = DAY,
) -> float:
    information_ratio = calc_information_ratio(returns, bench_returns)
    annual_factor = ANNUALIZATION_FACTORS[frequency]
    return information_ratio * np.sqrt(annual_factor)


def calc_volatility(returns: np.ndarray) -> float:
    std_return = np.std(returns, ddof=1)
    return std_return


def calc_annual_volatility(
    returns: np.ndarray,
    frequency: DAY,
) -> float:
    volatility = calc_volatility(returns)
    annual_factor = ANNUALIZATION_FACTORS[frequency]
    return volatility * np.sqrt(annual_factor)


def calc_downside_deviation(
    returns: np.ndarray,
    risk_free_rate: float = 0,
) -> float:
    if len(returns) < 2:
        return 0
    return np.std(returns[returns < risk_free_rate], ddof=1)


def calc_annual_downside_deviation(
    returns: np.ndarray,
    risk_free_rate: float = 0,
    frequency: str = DAY,
) -> float:
    downside_deviation = calc_downside_deviation(returns, risk_free_rate)
    annual_factor = ANNUALIZATION_FACTORS[frequency]
    return downside_deviation * np.sqrt(annual_factor)


def calc_tracking_error(
    returns: np.ndarray,
    bench_returns: np.ndarray,
) -> float:
    tracking_returns = returns - bench_returns
    return np.std(tracking_returns, ddof=1)


def calc_annual_tracking_error(
    returns: np.ndarray,
    bench_returns: np.ndarray,
    frequency: str = DAY,
) -> float:
    tracking_error = calc_tracking_error(returns, bench_returns)
    annual_factor = ANNUALIZATION_FACTORS[frequency]
    return tracking_error * np.sqrt(annual_factor)


def calc_bench_corr(
    returns: np.ndarray,
    bench_returns: np.ndarray,
) -> float:
    return np.corrcoef(returns, bench_returns)[0, 1]


def calc_sortino(
    returns: np.ndarray,
    risk_free_rate: float = 0,
) -> float:
    mean_excess_return = np.mean(returns) - risk_free_rate
    downside_deviation = calc_downside_deviation(returns, risk_free_rate)
    if downside_deviation == 0:
        return np.nan
    sortino = mean_excess_return / downside_deviation
    return sortino


def calc_annual_sortino(
    returns: np.ndarray,
    risk_free_rate: float = 0,
    frequency: str = DAY,
) -> float:
    sortino = calc_sortino(returns, risk_free_rate)
    annual_factor = ANNUALIZATION_FACTORS[frequency]
    return sortino * np.sqrt(annual_factor)


def calc_calmar(
    returns: np.ndarray,
    risk_free_rate: float = 0,
) -> float:
    max_drawdown = calc_max_drawdown(returns)
    mean_excess_return = np.mean(returns) - risk_free_rate
    return mean_excess_return / max_drawdown


def calc_annual_calmar(
    returns: np.ndarray,
    risk_free_rate: float = 0,
    frequency: str = DAY,
) -> float:
    calmar = calc_calmar(returns, risk_free_rate)
    annual_factor = ANNUALIZATION_FACTORS[frequency]
    return calmar * np.sqrt(annual_factor)


def calc_beta(
    returns: np.ndarray,
    bench_returns: np.ndarray,
) -> float:
    period_count = len(returns)
    if period_count < 2:
        return np.nan
    cov_matrix = np.cov(returns, bench_returns)
    variance = cov_matrix[1][1]
    if variance == 0:
        return np.nan
    covariance = cov_matrix[0][1]
    return covariance / variance


def calc_alpha(
    returns: np.ndarray,
    bench_returns: np.ndarray,
    risk_free_rate: float = 0,
) -> float:
    beta = calc_beta(returns, bench_returns)
    excess_bench_returns = bench_returns - risk_free_rate
    alpha = np.mean(returns - beta * excess_bench_returns) - risk_free_rate
    return alpha


def calc_annual_alpha(
    returns: np.ndarray,
    bench_returns: np.ndarray,
    risk_free_rate: float = 0,
    frequency: str = DAY,
) -> float:
    alpha = calc_alpha(returns, bench_returns, risk_free_rate)
    annual_factor = ANNUALIZATION_FACTORS[frequency]
    return alpha * np.sqrt(annual_factor)


class ReturnPerformance:
    def __init__(
        self,
        returns: np.ndarray,
        bench_returns: np.ndarray = None,
        frequency: str = DAY,
    ) -> None:
        """The performance of a portfolio.

        Parameters
        ----------
        returns : np.ndarray
            The returns of the portfolio.
        bench_returns : np.ndarray, optional
            The returns of the benchmark, by default None
        frequency : str, optional
            The frequency of the returns, by default DAY
        """

        if not isinstance(returns, np.ndarray) or returns.ndim != 1:
            raise TypeError('returns must be a 1d numpy array')
        if bench_returns is not None:
            if not isinstance(bench_returns,
                              np.ndarray) or bench_returns.ndim != 1:
                raise TypeError('bench_returns must be a 1-d numpy array')
            if len(returns) != len(bench_returns):
                raise ValueError(
                    'returns and bench_returns must have the same length')
            self._with_bench = True
        else:
            bench_returns = np.zeros_like(returns)
            self._with_bench = False
        if frequency not in FREQUENCY_LIST:
            raise ValueError(f'frequency must be one of {FREQUENCY_LIST}')

        # define the attributes
        self._returns = returns
        self._bench_returns = bench_returns
        self._frequency = frequency
        self._risk_free_rate = RISK_FREE_RATES[frequency]
        self._tracking_returns = returns - bench_returns
        self._annual_factor = ANNUALIZATION_FACTORS[frequency]

    @cached_property
    def period_count(self) -> int:
        return len(self._returns)

    @cached_property
    def net_values(self) -> np.ndarray:
        return np.insert(np.exp(np.log1p(self._returns).cumsum()), 0, 1)

    @cached_property
    def profit(self) -> np.ndarray:
        return self._returns[self._returns > 0]

    @cached_property
    def loss(self) -> np.ndarray:
        return self._returns[self._returns < 0]

    @cached_property
    def mean_return(self) -> float:
        return np.mean(self._returns)

    @cached_property
    def std_return(self) -> float:
        return np.std(self._returns, ddof=1)

    @cached_property
    def mean_excess_return(self) -> float:
        return self.mean_return - self._risk_free_rate

    @cached_property
    def return_rate(self) -> float:
        return calc_return(self._returns)

    @cached_property
    def annual_return(self) -> float:
        return (1 + self.return_rate)**(self._annual_factor /
                                        self.period_count) - 1

    @cached_property
    def sharpe(self) -> float:
        if self.std_return == 0:
            return np.nan
        return self.mean_excess_return / self.std_return

    @cached_property
    def annual_sharpe(self) -> float:
        return self.sharpe * np.sqrt(self._annual_factor)

    @cached_property
    def win_rate(self) -> float:
        if len(self.profit) == 0 and len(self.loss) == 0:
            return np.nan
        return len(self.profit) / (len(self.profit) + len(self.loss))

    @cached_property
    def avg_win_return(self) -> float:
        return np.mean(self.profit) if len(self.profit) > 0 else np.nan

    @cached_property
    def avg_loss_return(self) -> float:
        return np.mean(self.loss) if len(self.loss) > 0 else np.nan

    @cached_property
    def profit_loss_ratio(self) -> float:
        if len(self.profit) == 0:
            return np.nan
        elif len(self.loss) == 0:
            return np.inf
        else:
            return -self.avg_win_return / self.avg_loss_return

    @cached_property
    def max_drawdown(self) -> float:
        cummax_net_values = np.maximum.accumulate(self.net_values)
        return -np.min(self.net_values / cummax_net_values - 1)

    @cached_property
    def volatility(self) -> float:
        return self.std_return

    @cached_property
    def annual_volatility(self) -> float:
        return self.volatility * np.sqrt(self._annual_factor)

    @cached_property
    def bench_return(self) -> float:
        return calc_return(self._bench_returns)

    @cached_property
    def annual_bench_return(self) -> float:
        return (1 + self.bench_return)**(self._annual_factor /
                                         self.period_count) - 1

    @cached_property
    def tracking_return(self) -> float:
        return calc_return(self._tracking_returns)

    @cached_property
    def annual_tracking_return(self) -> float:
        return (1 + self.tracking_return)**(self._annual_factor /
                                            self.period_count) - 1

    @cached_property
    def tracking_win_rate(self) -> float:
        profit = self._tracking_returns[self._tracking_returns > 0]
        loss = self._tracking_returns[self._tracking_returns < 0]
        return len(profit) / (len(profit) + len(loss))

    @cached_property
    def tracking_error(self) -> float:
        return np.std(self._tracking_returns, ddof=1)

    @cached_property
    def annual_tracking_error(self) -> float:
        return self.tracking_error * np.sqrt(self._annual_factor)

    @cached_property
    def information_ratio(self) -> float:
        return self.tracking_return / self.tracking_error

    @cached_property
    def annual_information_ratio(self) -> float:
        return self.information_ratio * np.sqrt(self._annual_factor)

    @cached_property
    def beta(self) -> float:
        return np.cov(self._returns, self._bench_returns)[0, 1] / np.var(
            self._bench_returns)

    @cached_property
    def alpha(self) -> float:
        return self.mean_excess_return - self.beta * (
            self.annual_bench_return - self._risk_free_rate)

    @cached_property
    def annual_alpha(self) -> float:
        return self.alpha * self._annual_factor

    @cached_property
    def bench_corr(self) -> float:
        return np.corrcoef(self._returns, self._bench_returns)[0, 1]

    @cached_property
    def downside_deviation(self) -> float:
        return np.std(self.loss, ddof=1)

    @cached_property
    def annual_downside_deviation(self) -> float:
        return self.downside_deviation * np.sqrt(self._annual_factor)

    @cached_property
    def sortino(self) -> float:
        return self.mean_excess_return / self.downside_deviation

    @cached_property
    def annual_sortino(self) -> float:
        return self.sortino * np.sqrt(self._annual_factor)

    @cached_property
    def calmar(self) -> float:
        if self.max_drawdown == 0:
            return np.nan
        return self.annual_return / self.max_drawdown

    @cached_property
    def annual_calmar(self) -> float:
        return self.calmar * np.sqrt(self._annual_factor)

    def performance(self) -> dict:
        # default output of performance
        performance = {
            'annual_return': self.annual_return,
            'max_drawdown': self.max_drawdown,
            'annual_sharpe': self.annual_sharpe,
            'annual_calmar': self.annual_calmar,
            'win_rate': self.win_rate,
            'avg_win_return': np.mean(self.profit),
            'avg_loss_return': np.mean(self.loss),
            'profit_loss_ratio': self.profit_loss_ratio,
            'annual_volatility': self.annual_volatility,
            'annual_downside_deviation': self.annual_downside_deviation,
            'annual_sortino': self.annual_sortino,
        }

        # if the benchmark is set, add the benchmark related performance
        if self._with_bench == True:
            performance.update({
                'annual_bench_return': self.annual_bench_return,
                'annual_tracking_return': self.annual_tracking_return,
                'tracking_win_rate': self.tracking_win_rate,
                'tracking_error': self.tracking_error,
                'annual_tracking_error': self.annual_tracking_error,
                'annual_information_ratio': self.annual_information_ratio,
                'bench_corr': self.bench_corr,
                'beta': self.beta,
                'alpha': self.alpha,
                'annual_alpha': self.annual_alpha,
            })

        performance = {k: round(v, 4) for k, v in performance.items()}
        return performance


if __name__ == '__main__':
    returns = np.random.randn(100000) * 0.02
    bench_returns = np.random.randn(100000) * 0.02
    performance = ReturnPerformance(returns, bench_returns, frequency=DAY)
    print(performance.performance())
