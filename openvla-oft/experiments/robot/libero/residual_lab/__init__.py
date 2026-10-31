# -*- coding: utf-8 -*-
"""residual_lab — RefCloth 部署残差感知优化：探索/验证/实现（计划 2026-10-31）。

阶段链：Gate A 成对测量 → 阶段二校准数据 → Gate B 子空间诊断 →
Gate C 原生单步 → Gate D 单种子训练 → 阶段六冻结闭环验证。
数学核心见 residual_core.py（材料包原样收编，CPU 单测通过）。
"""
