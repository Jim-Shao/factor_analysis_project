# factor_analysis 因子回测报告生成工具

## 1. 介绍
该工具用于生成markdown与pdf格式的因子回测报告

功能：
1. 因子值分布分析
2. 因子多周期IC分析
3. 因子多周期分层回测、换手率分析
4. 因子值加权回测、换手率分析
5. 单因子策略（可选特定比例或排名靠前、靠后的股票）、换手率分析
6. 多因子相关性分析
7. 因子等权组合优化
8. 报告总结功能，可以将多个因子的回测结果汇总到一份报告中，从而查看某个选股池中不同因子的表现对比，以及单个因子在不同选股池中的表现对比

特性：
1. 支持指定调仓日期（1D，5D，10D，20D，...），以及特定日期调仓（尤其是财务因子，可在四月末、八月末、十月末调仓）
2. 支持多种因子值后处理，包括去极值、标准化、中性化处理（行业、市值、行业+市值）等
3. 支持多进程生成报告，提高运行效率
<!-- 4. 对于用于生成因子的底层特征如开高低收量等，使用共享内存，从而避免当底层特征数量过多或进程数过多时，内存占用过大的问题 -->

限制：
1. 仅支持因子值为日度的因子，暂不支持分钟级别或更低的因子


## 2. 安装
- 编译环境
    - Python >= 3.7

- 第三方依赖
    - pandas == 1.3.5
    - numpy == 1.22.1
    - matplotlib == 3.3.4
    - seaborn == 0.11.1
    - tqdm == 4.59.0
    - scipy == 1.6.2
    - tabulate == 0.9.0
    - markdown == 3.4.3
    - bs4 == 0.0.1
    - pypdf2 == 3.0.1
    - pyppeteer == 1.0.2 (用于生成pdf报告，如果系统中未安装Google Chrome，运行程序时会自动安装)

- 如何安装

    克隆项目到本地后移动到项目根目录
    ```shell
    cd factor_analysis_project
    ```
    创建conda环境
    ```shell
    conda create -n factor_analysis python=3.8
    ```
    激活conda环境
    ```shell
    conda activate factor_analysis
    ```
    安装第三方依赖（该步骤受到网络环境影响，可能会失败，建议多尝试几次）
    ```shell
    python setup.py develop
    ```
    当看到以下信息时，说明安装成功
    ```shell
    Finished processing dependencies for factor-analysis==0.0.0
    ```
    将当前目录下的代码包factor_analysis安装到 Python 环境中
    ```shell
    pip install -e .
    ```

## 3. 目录结构
```
factor_analysis_project
├── demo
│   └── demo.py
├── factor_analysis
│   ├── __init__.py
│   ├── backtest.py         # 多因子回测
│   ├── factor_calc.py      # 因子计算
│   ├── factor.py           # 单因子回测
│   ├── markdown_writer.py  # 报告生成
│   ├── optimize.py         # 因子优化
│   ├── plot.py             # 因子分析图表绘制
│   ├── postprocess.py      # 因子值后处理
│   ├── risk.py             # 收益分析
│   ├── summary.py          # 因子回测总结
│   └── utils.py            # 工具
├── .gitignore
├── README.md
├── setup.cfg
├── setup.py
└── template.html           # 报告模板
```



## 4. 快速入门

打开[./demo/demo.py](./demo/demo.py)运行查看效果。

运行过程中demo在Console中输出示例如下：
```

******************************************* Factor Analysis *******************************************

******************************************* Factor Analysis *******************************************
### Backtest config: universe=RandomGenerated1, datetime_range=('2010-01-04', '2010-10-08'), n_jobs=2
>>> Calculating factors: 100%|████████████████████████████████████████████| 2/2 [00:00<00:00,  5.77it/s]
>>> Calculating forward returns: Done in 0.03s
>>> Calculating periodic net values: Done in 0.18s
>>> Analyzing factors: 100%|██████████████████████████████████████████████| 3/3 [00:33<00:00, 11.05s/it]
>>> Plotting factor correlation: Done in 0.44s
>>> Optimizing factor combination:
(n=1) Best excess sharpe: 2.3360, open
Combining 2 factors: 100%|████████████████████████████████████████████████| 3/3 [00:00<00:00, 10.05it/s]
(n=2) No improvement
>>> Output directory: /home/shaoshijie/factor_analysis_project/factor_analysis_output/RandomGenerated1
>>> Total time for running backtest: 35.32s
********************************************************************************************************


******************************************* Factor Analysis *******************************************
### Backtest config: universe=RandomGenerated2, datetime_range=('2010-01-04', '2010-10-08'), n_jobs=2
>>> Calculating factors: 100%|████████████████████████████████████████████| 2/2 [00:01<00:00,  1.91it/s]
>>> Calculating forward returns: Done in 0.03s
>>> Calculating periodic net values: Done in 0.19s
>>> Analyzing factors: 100%|██████████████████████████████████████████████| 3/3 [00:34<00:00, 11.38s/it]
>>> Plotting factor correlation: Done in 0.48s
>>> Optimizing factor combination:
(n=1) Best excess sharpe: 1.1758, open
Combining 2 factors: 100%|████████████████████████████████████████████████| 3/3 [00:00<00:00,  9.95it/s]
(n=2) No improvement
>>> Output directory: /home/shaoshijie/factor_analysis_project/factor_analysis_output/RandomGenerated2
>>> Total time for running backtest: 37.16s
********************************************************************************************************

生成summary.pdf文件: /home/shaoshijie/factor_analysis_project/factor_analysis_output/summary.pdf
```

## 5. 其他

1. 回测对于传入的数据格式有一定要求:
    - bar_df, factor_df, benchmark_df的索引格式为
    ```MultiIndex['datetime'(level=0, pd.Timestamp), 'order_book_id'(level=1, str)]```
    - bar_df, factor_df的索引需要完全一致
    - benchmark_df的时间范围与bar_df, factor_df的时间范围需要完全一致
    - bar_df需要至少包含后复权收盘价(close), 用以计算收益率
2. 分层回测会对每天因子值为非nan的股票进行分层：假设上证50中只有30个因子具有某因子值，则只对这30个股票进行分五层时，另外20个股票不会被分入任何一层
3. 分层回测中，如果遇到多个股票因子值相同，那么这些股票内部的相对顺序是随机的，但是相对于其他不同因子值的股票，这些股票的顺序是固定的，因为这些相同因子值的股票的顺序是靠某一个随机种子控制的
4. 在分层回测中，因子值大小的排序方式是从小到大，因此因子值越大的股票，其分层数越大。比如划分5组，因子值最大的股票为第5组，因子值最小的股票为第1组。对应的，多空组合(long_short)为第5组减第1组，多头超额(long_excess)为第五组减基准，空头超额(short_excess)为基准减第一组
5. 分层回测中的换仓日是根据前瞻周期自动确定的，比如前瞻周期为20天就每20个交易日取一天为换仓日；但是在因子加权回测与单因子回测中，换仓日是通过传入的参数```position_adjust_datetimes```来确定的，可以为任意日期的组合

