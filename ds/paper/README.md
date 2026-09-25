# 论文图表目录

此目录提供论文手可直接使用的图表生成工具。生成器只依赖 Python 标准库，不需要安装 `numpy`、`pandas`、`matplotlib` 或 `seaborn`。

## 运行

```powershell
python C:\work\数模\smgs_repo\ds\paper\generate_paper_figures.py
```

也可以先查看将生成哪些文件：

```powershell
python C:\work\数模\smgs_repo\ds\paper\generate_paper_figures.py --list
```

指定输出目录：

```powershell
python C:\work\数模\smgs_repo\ds\paper\generate_paper_figures.py --output C:\temp\ds_paper_output
```

## 输出

```text
paper/
├─ figures/                  # SVG 矢量图，可直接插入 Word/LaTeX 或转 PNG
├─ tables/                   # 论文可排版 CSV
└─ FIGURE_CATALOG.md         # 图名、图注、数据来源与写作建议
```

## 设计原则

- 所有图表从 `ds/artifacts` 或 `ds/submission` 的冻结结果重新读取，不硬编码实验数值；
- 图片为 SVG，文本可编辑，适合中文论文排版；
- CSV 保留原始数值和字段名，便于论文手自行重绘；
- 不修改训练结果、不重跑模型、不读取附件 3/4 标签；
- 图表中的中文建议使用 `Microsoft YaHei` 或 `SimSun` 字体渲染。

## 文件映射

`FIGURE_CATALOG.md` 由生成器自动维护。论文中引用图片时，建议同时保留图注和来源文件说明，以体现结果可复核性。
