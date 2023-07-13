# factor_analysis 因子回测报告生成工具

## 1. 介绍
该工具用于生成Markdown格式的因子回测报告，主要包括以下功能：
1. 因子多周期IC分析
2. 因子收益分析（分层、因子值加权、指定quantile组）
3. 因子换手率分析
4. 多因子相关性分析
5. 因子组合优化

特性：
1. 支持多进程计算因子和生成报告，提高运行效率
2. 支持指定调仓日期，以适应不同的调仓周期，以及特定日期调仓（尤其是财务因子）
3. 内置多种因子值后处理，包括中性化处理（行业、市值、行业+市值）

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
    安装第三方依赖
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

1. 多进程模式在windows下运行会出现异常（单进程模式正常），建议在linux下运行多进程模式


