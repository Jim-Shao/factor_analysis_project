#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@Author  :   shaoshijie
@File    :   markdown_writer.py
@Description    :   编写markdown文件的类
'''

import os
import asyncio
import subprocess
import markdown
import pandas as pd
from bs4 import BeautifulSoup
from pyppeteer import launch

HTML_TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                  "template.html")


def get_chrome_executable_path():
    """获取chrome路径"""
    if os.name == 'posix':  # Linux 或 macOS
        # 尝试查找Chrome执行文件
        try:
            chrome_path = subprocess.check_output(['which', 'google-chrome'
                                                   ]).decode().strip()
            if chrome_path:
                return chrome_path
            else:
                chrome_path = subprocess.check_output(['which', 'chrome'
                                                       ]).decode().strip()
                if chrome_path:
                    return chrome_path
        except subprocess.CalledProcessError:
            pass
    elif os.name == 'nt':  # Windows
        # 尝试查找Chrome执行文件
        try:
            chrome_path = subprocess.check_output(['where',
                                                   'chrome']).decode().strip()
            if chrome_path:
                return chrome_path
        except subprocess.CalledProcessError:
            pass

    # 如果在上述系统中未找到Chrome执行文件，返回None
    return None


def markdown_to_html(markdown_path, html_template_path=None):
    """
    将markdown文件转换为html文件

    Parameters
    ----------
    markdown_path : str
        markdown文件路径
    html_template_path : str, optional
        html模板文件路径，默认为None

    Returns
    -------
    html_path : str
        html文件路径
    """
    # 读取markdown文件
    with open(markdown_path, 'r', encoding='utf-8') as file:
        markdown_text = file.read()

    # 使用 Markdown 库进行转换
    html_output = markdown.markdown(text=markdown_text,
                                    output_format="html5",
                                    extensions=["tables", "toc"])

    # 使用自定义的模板
    with open(html_template_path, 'r', encoding='utf-8') as f:
        template = f.read()
    html_output = template.replace("{content}", html_output)

    # 将html写入文件
    html_path = markdown_path.replace(".md", ".html")
    with open(html_path, 'w', encoding='utf-8') as file:
        file.write(html_output)

    return html_path


async def html_to_pdf(html_path, pdf_path):
    browser = await launch()
    page = await browser.newPage()

    # Convert the local file path to URL format
    file_url = 'file://' + os.path.abspath(html_path)

    await page.goto(file_url, waitUntil='load', timeout=120000)
    await page.pdf(
        options={
            'path': pdf_path,
            'margin': {
                'top': '1cm',
                'bottom': '1cm',
                'left': '0.8cm',
                'right': '0.8cm'
            },
            'printBackground': True,
        })

    await browser.close()


def relpath_to_abspath(html_file, base_dir):
    # Read the HTML file
    with open(html_file, 'r', encoding='utf-8') as file:
        html_content = file.read()

    # Create a BeautifulSoup object to parse the HTML
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all image tags with relative paths
    img_tags = soup.find_all(
        'img', src=lambda src: src and not src.startswith('http'))

    # Replace relative paths with absolute paths
    for img_tag in img_tags:
        src = img_tag['src']
        absolute_path = os.path.join(base_dir, src)
        img_tag['src'] = absolute_path

    # Save the modified HTML back to the file
    with open(html_file, 'w', encoding='utf-8') as file:
        file.write(soup.prettify())


def get_relative_path(path_a: str, path_b: str) -> str:
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
    if not common_prefix.endswith("/"):
        common_prefix = os.path.dirname(common_prefix)
    if common_prefix:
        relative_path = os.path.relpath(path_b, common_prefix)
    else:
        relative_path = path_b
    # 对于windows系统，将路径中的反斜杠替换为正斜杠
    if os.name == "nt":
        relative_path = relative_path.replace("\\", "/")
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
        self.md_dir = os.path.dirname(md_path)
        # 如果markdown文件已经存在，将其删除
        if os.path.exists(md_path):
            os.remove(md_path)

        # 添加标题
        if title is not None:
            self.add_title(title)

    def to_pdf(self, pdf_path: str):
        """
        将markdown文件转换为pdf文件

        Parameters
        ----------
        pdf_path : str
            pdf文件路径
        """
        html_path = markdown_to_html(self.md_path,
                                     html_template_path=HTML_TEMPLATE_PATH)
        # relpath_to_abspath(html_path, self.md_dir)
        asyncio.get_event_loop().run_until_complete(
            html_to_pdf(html_path, pdf_path))

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

    def add_table(self,
                  df: pd.DataFrame,
                  float_format: str = ".4f",
                  index: bool = True):
        """
        添加表格

        Parameters
        ----------
        df : pd.DataFrame
            表格数据
        """
        with open(self.md_path, "a", encoding="utf-8") as f:
            f.write(
                df.to_markdown(floatfmt=float_format, index=index) + "\n\n")

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
            f.write(f"![{image_name}]({image_path})\n\n")

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
    md_path = '/home/shaoshijie/factor_analysis_project/factor_analysis_output/open/open_report.md'
    html_path = '/home/shaoshijie/factor_analysis_project/factor_analysis_output/ROE/ROE_report.html'
    pdf_path = '/home/shaoshijie/factor_analysis_project/factor_analysis_output/ROE/ROE_report.pdf'

    # markdown_to_html(md_path)
    # relpath_to_abspath(html_path, os.path.dirname(md_path))
    asyncio.get_event_loop().run_until_complete(
        html_to_pdf(html_path, pdf_path))
