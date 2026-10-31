# -*- coding: utf-8 -*-
"""config.py — 实验配置装载与 fail-fast 路径校验。

所有路径显式；缺项/不存在直接抛错，不做静默回退（计划§14）。
配置文件：configs/refcloth_residual/experiment.yaml
"""
import json
import os

import yaml

D = "/data/shaozhen.liu"
REPO = f"{D}/code/tex3d"

REQUIRED_PATHS = [
    "paths.source_checkpoint", "paths.source_pool", "paths.matched_reference_cache",
    "paths.cloth_mesh", "paths.base_texture", "paths.output_root",
]
OPTIONAL_PATHS = ["paths.gain_map"]


class Cfg(dict):
    """点访问的配置 dict。"""
    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) else v


def load_config(path):
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = Cfg(raw)
    for key in REQUIRED_PATHS:
        node = cfg
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                raise ValueError(f"config 缺项: {key}")
            node = node[part]
        if not node or not isinstance(node, str):
            raise ValueError(f"config 缺项: {key}")
        if not os.path.exists(node):
            raise FileNotFoundError(f"config 路径不存在: {key} = {node}")
    for key in OPTIONAL_PATHS:
        node = cfg
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                node = None
                break
            node = node[part]
        if node and not os.path.exists(node):
            raise FileNotFoundError(f"config 可选路径不存在: {key} = {node}")
    return cfg


def baseline_lock(cfg, extra=None):
    """baseline_lock.json 内容（计划§3.3）。调用方补 commit/环境版本。"""
    import hashlib

    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        return h.hexdigest()

    lock = {
        "paths": dict(cfg.paths),
        "source_pool_sha256": sha(cfg.paths.source_pool),
        "matched_reference_cache_sha256": sha(cfg.paths.matched_reference_cache),
        "base_texture_sha256": sha(cfg.paths.base_texture),
        "cloth_mesh_sha256": sha(cfg.paths.cloth_mesh),
        "camera_order": list(cfg.source.cameras),
        "feature_layout": dict(cfg.source.feature_layout),
        "training_recipe": dict(cfg.training),
    }
    if cfg.paths.get("gain_map"):
        lock["gain_map_sha256"] = sha(cfg.paths.gain_map)
    if extra:
        lock.update(extra)
    return lock


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str, ensure_ascii=False)
