#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@Author  :   shaoshijie
@File    :   markdown_writer.py
@Description    :   编写markdown文件的类
'''

import os
import pandas as pd


def get_relative_path(path_a, path_b) -> str:
    """
    获取path_b相对于path_a的相对路径

    Parameters
    ----------
    path_a : str
        路径a
    path_b : str
        路径b

    Returns
    -------
    relative_path : str
        相对路径
    """
    common_prefix = os.path.commonprefix([path_a, path_b])
    if common_prefix:
        relative_path = os.path.relpath(path_b, common_prefix)
    else:
        relative_path = path_b

    return relative_path


class MarkdownWriter:
    def __init__(self, md_path: str, title: str = None):
        """
        编写markdown文件

        Parameters
        ----------
        md_path : str
            markdown文件路径（例："factor_analysis/report/report.md"）
        title : str, optional
            标题（例："因子分析报告"），默认为None
        """
        # 检查路径是否为markdown文件
        if not md_path.endswith(".md"):
            raise ValueError("md_path must be a markdown file path")
        # 检查路径是否存在
        if not os.path.exists(os.path.dirname(md_path)):
            os.makedirs(os.path.dirname(md_path))
        self.md_path = md_path

        # 添加标题
        if title is not None:
            self.add_title(title)

    def write(self, content: str):
        """
        添加内容

        Parameters
        ----------
        content : str
            内容
        """
        with open(self.md_path, "a", encoding="utf-8") as f:
            f.write(content + "\n")

    def add_title(self, title: str, level: int = 1):
        """
        添加标题

        Parameters
        ----------
        title : str
            标题
        level : int, optional
            标题级别（1-6），默认为1
        """
        with open(self.md_path, "a", encoding="utf-8") as f:
            f.write("#" * level + " " + title + "\n")

    def add_table(self, df: pd.DataFrame, float_format: str = "%.4f"):
        """
        添加表格

        Parameters
        ----------
        df : pd.DataFrame
            表格数据
        """
        with open(self.md_path, "a", encoding="utf-8") as f:
            f.write(df.to_markdown(floatfmt=float_format) + "\n\n")

    def add_image(self, image_name: str, image_path: str):
        """
        添加图片

        Parameters
        ----------
        image_name : str
            图片名称
        image_path : str
            图片路径
        """
        image_path = get_relative_path(self.md_path, image_path)
        with open(self.md_path, "a", encoding="utf-8") as f:
            f.write(f"![{image_name}]({image_path})\n")

    def add_code(self, code: str, language: str = "python"):
        """
        添加代码

        Parameters
        ----------
        code : str
            代码
        language : str, optional
            代码语言，可选值为"python"、"sql"、"r"、"c"、"cpp"、"java"、"javascript"、"typescript"、"html"、"css"、"json"、"yaml"、"xml"、"bash"、"shell"、"markdown"、"latex"、"text"、"none"，默认为"python"
        """
        with open(self.md_path, "a", encoding="utf-8") as f:
            f.write(f"```{language}\n{code}\n```\n")

    def add_link(self, link_name: str, link_url: str):
        """
        添加链接

        Parameters
        ----------
        link_name : str
            链接名称
        link_url : str
            链接地址
        """
        with open(self.md_path, "a", encoding="utf-8") as f:
            f.write(f"[{link_name}]({link_url})\n")

    def add_list(self, list: list, ordered: bool = False):
        """
        添加列表

        Parameters
        ----------
        list : list
            列表
        ordered : bool, optional
            是否有序，默认为False
        """
        with open(self.md_path, "a", encoding="utf-8") as f:
            if ordered:
                f.write(
                    "\n".join([f"{i}. {item}"
                               for i, item in enumerate(list)]) + "\n")
            else:
                f.write("\n".join([f"- {item}" for item in list]) + "\n")

    def add_newline(self):
        """
        添加换行
        """
        with open(self.md_path, "a", encoding="utf-8") as f:
            f.write("\n")

    def add_pagebreak(self):
        """
        添加分页符
        """
        with open(self.md_path, "a", encoding="utf-8") as f:
            f.write("<div style=\"page-break-after: always;\"></div>\n")

    def add_math(self, math: str):
        """
        添加数学公式

        Parameters
        ----------
        math : str
            数学公式
        """
        with open(self.md_path, "a", encoding="utf-8") as f:
            f.write(f"$$\n{math}\n$$\n")

    def add_blockquote(self, blockquote: str):
        """
        添加引用

        Parameters
        ----------
        blockquote : str
            引用
        """
        with open(self.md_path, "a", encoding="utf-8") as f:
            f.write(f"> {blockquote}\n")

    def add_bold(self, bold: str):
        """
        添加加粗

        Parameters
        ----------
        bold : str
            加粗
        """
        with open(self.md_path, "a", encoding="utf-8") as f:
            f.write(f"**{bold}**\n")

    def add_italic(self, italic: str):
        """
        添加斜体

        Parameters
        ----------
        italic : str
            斜体
        """
        with open(self.md_path, "a", encoding="utf-8") as f:
            f.write(f"*{italic}*\n")


if __name__ == '__main__':
    md_writer = MarkdownWriter(
        "/home/shaoshijie/factor_analysis_project/test.md")
    md_writer.make_html("/home/shaoshijie/factor_analysis_project/test.html")