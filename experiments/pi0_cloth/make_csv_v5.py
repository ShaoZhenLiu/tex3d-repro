#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""make_csv_v5.py — v4 → v5（第13轮第0步，老师 §6.3/附录A.2）
1) 新增 5 格的 texture_sha256 由 16 位前缀补成完整 64 位（完整值取自部署 PNG 实测）；
2) 新增 noise_policy_id 列：历史行 = legacy_independent（独立进程/全局流，无共噪声证据，
   不追认共噪声配对）；第12轮新增 5 格 = query_seed_logged（逐查询种子已记录于
   41 edpa_logs/qseed_*.csv，种子=f(task,ep,query序)；"记录"不等于"按该种子生成"，
   共噪声复核为后续可选项）。v4 保留不覆盖。"""
import csv

CSV4 = "交给老师_材料包_20261021/04_数据/episodes_merged_all_v4.csv"
OUT = "交给老师_材料包_20261021/04_数据/episodes_merged_all_v5.csv"
FULL = {
    "B2v2rms": "c24a8ebeb582d88abbfc8254e2adb0e1ae977e6a4165a2df9ab863fe6898dc55",
    "B3rms": "a1eaa967e27b476d9fc2a51842955f6638304b9f6a19bed9a3fc380f7b9dbeb0",
    "C1": "57828087be7b92c2a31390ca8f4166fea5d5623b9d3391e6cbd381f54c3b84cf",
    "C1rms": "40ce034c068dbedfb91d9153702a001dd0fa39c4a222048c2aa37e0aa9940f53",
}
NEW = set(FULL)

rows = list(csv.DictReader(open(CSV4, encoding="utf-8")))
fields = ["method", "target", "suite", "cond", "task", "episode_index",
          "init_state_id", "success", "protocol_id", "texture_sha256",
          "noise_policy_id"]
n_new = 0
for r in rows:
    m = r["method"]
    if m in NEW:
        assert len(r["texture_sha256"]) == 16
        assert r["texture_sha256"] == FULL[m][:16], (m, r["texture_sha256"])
        r["texture_sha256"] = FULL[m]
        r["noise_policy_id"] = "query_seed_logged"
        n_new += 1
    else:
        if m == "F0" and r["cond"] in ("O", "P"):
            pass  # 无纹理行保持空（不伪造哈希）
        r["noise_policy_id"] = "legacy_independent"

with open(OUT, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print("v4=%d 行 → v5=%d 行（补全 SHA %d 行，加 noise_policy_id）→ %s"
      % (len(rows), len(rows), n_new, OUT))
