# 给 Tex3D 作者的 Issue 草稿(提交流程与中文摘要)

> **中文摘要(给你过目)**:本 issue 包含三条互相独立的发现,均有行号级证据和量化实验支撑:
> ① 评测脚本末行打印的 "Attack success rate" 实际是"受攻击下机器人任务成功率",语义相反——这大概率是社区里"OFT 复现 95% vs 0%"之争的根源,两边可能在引用同一个数字的不同读法;
> ② π0 的纹理烘焙导出与训练渲染路径数学不等价,零扰动时导出也无法恢复原纹理(酒标资产实测 PSNR 3.2dB),影响部署保真度;我们验证过一个零误差的修复(只烘噪声场);
> ③ 用官方 RLinf 评测栈 + 你们发布的 RLinf π0 权重复测,clean 失败率 33-66%,与论文 Table 的 3.5-7.2% 差 30-60pp;我们自建栈与官方栈完全一致,说明不是复现环境问题——想确认论文 clean 数字的权重与口径。
> 提交建议:可以一条 issue 全发,或拆成三条(①③更值得单开)。英文正文如下。

---

## Title

Three reproducibility findings: inverted metric label in eval scripts, π0 bake/train mismatch, and clean-baseline gap with released checkpoints

## Body

Hi, thanks for releasing the code — we've completed a full reproduction over ~3 weeks (OpenVLA / OpenVLA-OFT / π0, all four LIBERO suites, 50 episodes per task, plus cross-model transfer). The attack pipeline itself reproduces well for us. Along the way we hit three issues that other reproducers will likely encounter too, so we're reporting them with code-line evidence and measurements. Happy to share any logs.

### 1. `[DONE] Attack success rate` prints the *task success* rate, not the attack success rate

In `openvla/experiments/robot/libero/attack_openvla.py:1344` and `openvla-oft/experiments/robot/libero/attack_oft.py:1576`:

```python
avg_sr = total_successes / total_episodes   # successes = env.step(...) done flags = TASK successes
print(f"\n[DONE] Episodes: {total_episodes} | Attack success rate: {avg_sr:.2%}")
```

The numerator counts episodes where the **robot completed the task**, so the printed number is the policy's task-success rate under attack. The true attack success rate is `1 - printed`. Concretely, our OFT spatial run prints `Attack success rate: 0-2%` for most tasks — meaning the robot almost never completes the task, i.e. the attack is nearly fully effective (~98-100%) — but the label reads as if the attack failed.

We suspect this label is a major source of the split reports in the issues (e.g. "OFT reproduces at 95%" vs "OFT attack is ~0%"): both camps may be describing the same runs, one quoting the printed label and the other the complement. We initially misread it ourselves, which reversed several conclusions in our own report until we recomputed everything from raw episode lines.

Suggested fix: print both, e.g. `task success under attack: X% | attack success (task failure): Y%`.

### 2. π0: baked texture ≠ rendered texture, even with zero perturbation

`pi/attack_pi0.py` renders with per-pixel full-resolution texture sampling (`dr.texture(orig_texture, tex_uv)`, ~line 364) during training and built-in eval, but `get_baked_adv_texture()` (~line 385) samples the original texture **once per vertex**, adds the vertex noise, and interpolates those vertex colors across the UV atlas. The two are not equivalent: the exported texture is a piecewise-linear approximation of the original pattern.

Measured on the released LIBERO assets (zero perturbation, PSNR of baked vs original): `porcelain_mug` 47.4 dB, `akita_black_bowl` 25.1 dB, `moka_pot` 20.8 dB, `cream_cheese` (24 vertices) 16.8 dB, `wine_bottle` label (27 vertices, fine text) **3.2 dB** — the label is essentially destroyed.

White-box numbers are unaffected (evaluation goes through `render()`), but rollout videos, the MuJoCo-deployment path, and any physical-print fidelity are affected. A fix that we verified to be exact: bake only the interpolated vertex-noise field into UV space and add it to the untouched original texture (zero error at zero perturbation on all assets).

(For reference, the OpenVLA/OFT renderer does not have this inconsistency — it uses vertex colors consistently everywhere — at the cost of never rendering the original texture at full detail.)

### 3. π0 clean baselines: released checkpoints + official RLinf eval reproduce 33-66% failure, not 3.5-7.2%

We ran the official stack verbatim (openpi `scripts/serve_policy.py` + `examples/libero/main.py`, default protocol: replan 5, 10 settle steps, 50 episodes/task, seed 7, per-suite step budgets 220/280/300/520) with the released `RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT` and `RLinf-Pi0-LIBERO-Long-SFT`, 10 tasks × 50 episodes per suite, zero exceptions:

| Suite | Official stack clean failure | Our in-process stack | Paper "No Attack" |
|---|---|---|---|
| spatial | 37.6% | 38.4% | 3.5% |
| object | 33.0% | 29.4% | 2.3% |
| goal | 47.0% | 41.2% | 5.2% |
| long | 66.2% | 66.0% | 7.2% |

Our own evaluation loop (the repo's in-process path) agrees with the official stack within 6 pp on every suite — so this doesn't appear to be an environment issue on our side (RTX 4090, EGL, torch 2.6.0+cu124, openpi @215abfb; we also ruled out image resolution/preprocessing and step budgets with single-variable experiments). Could you confirm which checkpoint revision and protocol produced the paper's No Attack numbers? If the released weights differ from the ones used in the paper, that would explain the gap.

---

Environment for all measurements: 4× RTX 4090 48GB, MuJoCo EGL, per-GPU isolated workspaces, HF_HUB_OFFLINE. Full logs and per-episode records available on request — thanks again for the work!
