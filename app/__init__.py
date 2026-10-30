# -*- coding: utf-8 -*-
"""QuestEdu 学问 —— 苏格拉底式数学伴学。

模块划分（对应大纲第三节）：
    app.core.mathcheck   确定性计算层：SymPy 兜底，所有数字的唯一来源
    app.core.judge       答案判定层：算 → 对表述 → 才谈对错
    app.core.engine      教学状态机：推不推进、给哪级提示，全由代码决定
    app.core.llm         统一 LLM 适配层：cloud / local / mock 一行切换
    app.api              HTTP 接口（路由薄，逻辑都在 core）
    app.data             题库：加题只改数据，不动引擎
    app.web              移动端单页前端 + 本地 vendor（KaTeX）
"""

__version__ = "0.2.0"
