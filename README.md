# factor_analysis 因子回测报告生成工具

## 1. 介绍
该工具用于生成markdown与pdf格式的因子回测报告

- [点击查看markdown示例报告](/factor_analysis_output/factor1/factor1_report.md)

- [点击查看pdf示例报告](/factor_analysis_output/factor1/factor1_report.pdf)

功能：
1. 因子多周期IC分析
2. 因子收益分析（分层、因子值加权、单因子策略指定quantile或前几支）
3. 因子换手率分析
4. 多因子相关性分析
5. 因子组合优化

特性：
1. 支持指定调仓日期，以适应不同的调仓周期，以及特定日期调仓（尤其是财务因子）
2. 内置多种因子值后处理，包括中性化处理（行业、市值、行业+市值）
3. 支持多进程计算因子和生成报告，提高运行效率
4. 对于用于生成因子的底层特征如开高低收量等，使用共享内存，从而避免当底层特征数量过多或进程数过多时，内存占用过大的问题

限制：
1. 仅支持因子值为日度的因子，暂不支持分钟级别或更低的因子


## 2. 安装
- 编译环境
    - Python >= 3.7
    - Google Chrome (用于生成pdf报告，如果未安装，运行程序时会自动安装浏览器)

- 第三方依赖
    - pandas == 1.3.5
    - numpy == 1.22.1
    - matplotlib == 3.3.4
    - seaborn == 0.11.1
    - tqdm == 4.59.0
    - scipy == 1.6.2
    - tabulate == 0.9.0
    - markdown == 3.4.3
    - pyppeteer == 1.0.2
    - bs4 == 0.0.1

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

## 3. 快速入门

打开[./demo/demo.py](./demo/demo.py)运行查看效果。

## 4. 其他

1. 回测对于传入的数据格式有一定要求，主要在于pd.DataFrame或pd.Series的索引为
```MultiIndex['datetime'(level=0, DatetimeIndex), 'order_book_id'(level=1, Index)]```
2. 多进程模式在Windows下运行会出现异常（单进程模式正常），建议在Linux下运行多进程模式


