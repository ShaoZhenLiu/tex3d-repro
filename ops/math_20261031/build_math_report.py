# -*- coding: utf-8 -*-
"""Build the Chinese review report from complete immutable result JSONs."""
import argparse
import html
import json
from pathlib import Path


def build(folder):
    results=folder/'results'
    summary=json.loads((results/'math_summary.json').read_text(encoding='utf-8'))
    metric=json.loads((results/'metric/metric_reassessment.json').read_text(encoding='utf-8'))
    original=json.loads((results/'metric/metric_cache_report.json').read_text(encoding='utf-8'))
    def e(x):return html.escape(str(x))
    def table(headers,rows):
        return '<table><thead><tr>'+''.join('<th>'+e(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join(
            '<tr>'+''.join('<td>'+e(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
    names=dict(init='初始化',moderate='中幅度纹理',frozen_mr='冻结MR纹理')
    rows=[]
    for rec in summary['geometry']:
        candidate=rec['methods']['maximin']
        rows.append([names[rec['start']],rec['eta'],rec['normalization'],
            format(rec['best_baseline_minimum'],'.6g'),format(candidate['minimum'],'.6g'),
            format(rec['best_baseline_mean'],'.6g'),format(candidate['mean'],'.6g'),
            '通过' if rec['gate_setting'] else '未通过'])
    geometry=table(['起点','步长上限','分组尺度','基线中最大的最差增益','LP最差增益','基线中最大的平均增益','LP平均增益','单组判门'],rows)
    rows=[]
    for key in ['init','moderate','frozen_mr','all']:
        r=summary['fidelity'][key]
        rows.append([names.get(key,'全部起点，按信号能量汇总'),format(r['action_AA_relative_error'],'.4f'),
            format(r['action_interaction_relative_error'],'.4f'),r['near_zero_native_interaction_count']])
    fidelity=table(['起点','普通AA位移相对误差','四角交互相对误差','原生交互RMS<1e-6数量'],rows)
    rows=[]
    for key,label in [('direct_action','直接动作位移'),('generalized','广义特征度量'),
                      ('robust_absolute','绝对椭球下界'),('robust_relative','相对椭球下界')]:
        r=metric['validation']['methods'][key]
        rows.append([label,r['zero_score_count'],r['all_zero_states'],r['defined_state_spearman_count'],
            format(r['mean_tie_expected_native_energy'],'.6g'),
            format(r['mean_within_state_spearman_zero_for_undefined'],'.4f')])
    uncertainty=table(['方法','零分数/64','全零状态/16','可定义排序状态/16','并列公平选择的原生能量均值','全状态平均排序相关'],rows)
    raw=summary['geometry_gate']['raw']['settings_passed']
    normalized=summary['geometry_gate']['l1']['settings_passed']
    sources=[
        ('Adversarial Attack Generation Empowered by Min-Max Optimization','https://arxiv.org/abs/1906.03563','跨输入、模型与变换的最坏域攻击和simplex加权已有先例。'),
        ('Generating Adversarial Examples with Task Oriented Multi-Objective Optimization','https://arxiv.org/abs/2304.13229','L2多目标组合及L∞中的sign执行是必须区分的先例。'),
        ('Improving Generalization of Universal Adversarial Perturbation via Dynamic Maximin Optimization','https://arxiv.org/abs/2503.12793','通用扰动的maximin大方向已被占据；本轮不宣称首次。'),
        ('Fool the Hydra','https://arxiv.org/abs/2312.00173','共享多视角patch及梯度聚合不是新增贡献。'),
        ('Interpreting Attributions and Interactions of Adversarial Attacks','https://arxiv.org/abs/2108.06895','扰动交互与因子差分已有基础。'),
        ('Jailbreak in pieces','https://arxiv.org/abs/2307.14539','联合输入触发不能作为全新大概念。')]
    source_html=''.join('<li><a href="'+url+'">'+e(title)+'</a>：'+e(note)+'</li>' for title,url,note in sources)
    body='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RefCloth 数学方法探索与原生验证</title><style>
body{font:16px/1.8 "Microsoft YaHei",system-ui,sans-serif;background:#f2f5f6;color:#24323b;margin:0}main{max-width:1140px;margin:32px auto;background:white;padding:36px 42px;border:1px solid #d9e1e5}h1{font-size:29px}h2{font-size:21px;margin-top:34px}table{border-collapse:collapse;width:100%;font-size:13px;margin:18px 0}th,td{padding:9px;border:1px solid #d1dce1;text-align:left}th{background:#edf3f5}.lead{background:#edf5f7;border-left:4px solid #216b80;padding:15px 20px}.minor{color:#5b6c76;font-size:14px}a{color:#176c85}pre{background:#f0f4f5;padding:18px;font:14px/1.65 Consolas,monospace;white-space:pre-wrap}img{width:100%;height:auto}li{margin-bottom:9px}@media print{body{background:white}main{border:0;margin:0;padding:0}h2{break-after:avoid}table,img{break-inside:avoid}}
</style></head><body><main>
<p class="minor">2026年10月31日 · math-innovation-20261031 · 起点提交 f892860</p>
<h1>RefCloth：从数学构造到真实原生更新</h1>
<div class="lead"><strong>本轮完成了三条数学路线的推导、实现和可证伪实验，尚未得到稳定优于强基线的新主方法。</strong>最终更新盒LP有局部正结果，但跨起点不稳定；四角交互未改善原生预测保真；椭球下界大量退化为零。没有启动完整训练或生成新的闭环成功率，也不能宣称审稿创新性已提高。</div>
<p><a href="mathematical_notes.pdf">六页数学推导PDF</a> · <a href="mathematical_notes.tex">对应LaTeX源码</a> · <a href="https://github.com/ShaoZhenLiu/tex3d-repro/tree/math-innovation-20261031">代码分支</a></p>
<h2>实验范围与可比较性</h2>
<p>所有新模型实验只查询冻结的源OpenVLA-OFT。初态5的四个任务{0,2,6,8}提供梯度，初态6的同四任务仅做独立原生验证。三种固定纹理起点、两种参数步长上限和12种更新策略，共288条验证记录；它们来自四个保留状态，不是288个独立场景。旧locked_test未读取，没有用目标策略选择新纹理。</p>
<p>GPU0和GPU2并行运行完整源模型，GPU1单独运行动作头缓存分析；GPU4未使用。09当时显存与计算均繁忙，用于同步和CPU复核。完整模型的峰值分配显存约19.08GB。所有新数学组件22项测试通过。零扰动PNG逐像素恢复、原生A/A、A→B→A以及输入时点检查均通过；八个新源状态的旧obs与实时渲染差异均为0。</p>
<p>动作端点固定为前五个执行动作的三个平移坐标。按生产q01/q99做仿射反归一化，三轴scale约为(0.84149,0.76875,0.93482)；偏移在差分中抵消。它们是控制量，不是米制轨迹误差，也不是任务成功率。</p>
<h2>尝试一：最终执行更新盒中的共同上升</h2>
<p>原问题是一个静态纹理要同时影响多任务，而平均梯度可能牺牲个别任务。常见多目标梯度在L2空间中组合后再取sign，也不能无条件保留共同上升。下面给出严格反例，不依赖VLA实验：</p>
<pre>g₁ = (1,0),  g₂ = (−2,1),  δ ∈ [−1,1]²
平均梯度的sign：δ = (−1,+1)，任务1增益 = −1
PCGrad后再sign：δ = (+1,+1)，任务2增益 = −1
最终盒内的max–min原始解：δ = (1/3,1)，两任务增益均为1/3</pre>
<p>因此直接求下式并保留原始分数坐标，而不是再次sign：</p>
<pre>max δ,t  t
subject to  qᵢᵀδ ≥ t  for every task i
            max(−η,−5−θⱼ) ≤ δⱼ ≤ min(η,5−θⱼ)</pre>
<p>q可为原始总目标梯度，也可按该任务fit梯度的L1范数归一化。总目标保留生产正则：J=10FD−0.1幅度MSE−0.1纹理TV。零梯度任务不删除；没有正共同上升裕量时不执行任意LP端点步。对fit中完全无梯度的坐标固定零更新，避免用求解器的任意解注入无目的纹理。</p>
<p>数学成果是明确原始/对偶形式、可检查的对偶间隙、sign丢共同上升的反例，以及条件性原生增益界。存在最优基本解只需至多N−1个非端点自由坐标；固定坐标不计入该性质。它们是标准凸优化和线性规划的应用，不作为全新数学理论。原论文匹配参考命题讨论的是一个固定目标的局部sign优势，本反例不推翻那个命题。</p>
<p><strong>原生结果：</strong>原始梯度版通过 RAW_PASS/6 组设置，归一化版通过 NORM_PASS/6；预设要求至少4/6。强对照包含均值sign、PCGrad后sign及保留方向的PCGrad向量。下表完整列出所有起点、步长与尺度。两个基线列分别取三种基线在该指标上的上包络，不一定来自同一策略。归一化行用对应fit任务L1尺度换算保留状态的原生J增益；没有在保留集重估尺度。</p>
GEOMETRY_TABLE
<p>局部正例值得保留：中幅度起点、η=0.05时，raw LP的最差原生J增益约0.01055，对照为0.003591；均值约0.02226，对照为0.01556。冻结MR起点上的均值反而降低，不能仅挑中幅度结果。未通过跨设置门槛，因此没有据此启动完整训练。</p>
<img src="results/figures/box_native_screen.png" alt="全部三起点两步长下原始与归一化更新的原生最差任务增益">
<h2>尝试二：四角交互作为渲染误差不变量</h2>
<p>在固定状态、指令和机器人输入下，分别组合Plain与扰动图像，定义：</p>
<pre>I = a_AA − a_AP − a_PA + a_PP</pre>
<p>若合成与原生的动作误差可写为任意可加形式E(u,v)=b+e₁(u)+e₂(v)，四角差分会精确消掉它。在“对所有这类未知偏差不变、且AA系数固定为1”的线性对比中，(1,−1,−1,1)是唯一解。它提供了一个明确的辨识问题，而不只是任意加损失。</p>
<p>同时，||联合位移||≥max(0,||I||−||主视角单独位移||−||腕视角单独位移||)。因此，只有交互足够大且单视角效应受控，才有有用的联合动作下界。对于光滑理想模型，I可表示为混合Hessian的二重积分；实际PNG量化和BF16不自动满足光滑假设。即使交互误差的绝对量是二阶，I本身也可能二阶小，不能推出相对误差改善。</p>
<p><strong>实验结果：</strong>12个fit状态—起点组合中，整体动作位移相对误差0.3317，而交互相对误差0.5496，反而约为1.66倍；原生交互近零的排除数为0。按起点如下，未通过预设的保真门。</p>
FIDELITY_TABLE
<p>进一步仍做了诊断性原生一步对照：直接联合动作、纯交互、单视角惩罚、对所有fit单视角一阶变化施加约束的LP。独立初态6上，受保护交互未在任何一组设置中达到“至少3/4状态联合效应增加10%，且双单视角MSE不增加超过1e−8”的门；六组均未通过。有些设置减少了单视角效应，却同时减少联合效应，不能叫联合失效。全部动作下界与交互前后值保留在验证JSON中。几何路线的策略共用同一批源梯度；融合路线只作相同步长诊断，各目标的反传次数不同，没有声称计算量匹配的性能优势。</p>
<h2>尝试三：动作空间误差椭球与精确最坏情形</h2>
<p>由旧源缓存拟合原生与合成动作位移误差的二阶矩，并分离形状拟合状态和半径校准状态。比较绝对半径、随位移幅度缩放的相对半径，以及广义特征值度量对照：</p>
<pre>Lrob(u) = min e  ||u+e||²  subject to eᵀC⁻¹e ≤ r(u)²
绝对版 r(u)=ρ；相对版 r(u)=ρ||u||
相对版包络梯度 = 2(y* − μρ²u)</pre>
<p>内层最优解通过一个单调标量根方程精确求得。相对半径依赖u，不能漏掉梯度中的第二项。数值检查覆盖KKT、可行性、有限差分、齐次性和条件下界。下界只在真实误差属于拟合椭球时成立；有限源随机纹理的经验覆盖不认证未来优化纹理。</p>
<p><strong>64个验证状态—纹理对：</strong>绝对版覆盖96.9%，但59/64分数为零；相对版覆盖92.2%，64/64分数为零。覆盖样本中没有条件下界违例，但相对版没有可用方向。广义特征度量的平均状态内排序相关约0.588，低于直接动作位移约0.688，未提供强于直接动作基线的证据。</p>
UNCERTAINTY_TABLE
<p>特别修正了全零argmax的伪收益：全零时字典序第一张纹理可能碰巧很好，原始选中/oracle平均约0.897；按所有并列纹理公平选择后只有约0.486。该方法完全没有辨别能力，不能拿前一个数宣称成功。最终报告同时保留原始输出和修正后的tie-aware复评，没有覆盖历史记录。</p>
<img src="results/figures/mathematical_diagnostics.png" alt="sign反例、交互保真和椭球退化诊断">
<h2>这些结果对创新性意味着什么</h2>
<p>本轮最明确的数学产物是：把共同上升条件落实到最终更新盒；界定四角交互可辨识的误差族及唯一性；推导正确的相对椭球包络梯度并识别其退化。它们让后续方法有可检查的条件和失败标准，但实验还不支持新的主算法。</p>
<p>现有多目标攻击、LP攻击、多视角共享patch、误差稳健优化和物理纹理冲突研究占据了多数宽泛概念。特别是MinMax攻击、TA-MOO、DM-UAP，以及代理核查中记录的MLA-LP与JMOF，限制了“首次max–min”“首次处理梯度冲突”等说法。数学笔记和完整先例记录给出了边界，不以搜索未找到完全相同公式来证明新颖性。</p>
<p>当前不将三个候选写成主方法，也不把局部负结果写成这些路线永远无效。若继续，最需要补的是拟议更新的原生增益如何被源侧有限查询可靠约束，以及这种收益能否跨更多初态、任务和纹理种子保留；只有原生门稳定通过后才进入训练和目标闭环。原论文及主表未改写。</p>
<h2>文件、复现与审计</h2>
<p>results/中保留逐项数据、初态与纹理标识、JSON汇总和PNG/PDF图；code/中有源码和跑前计划；三个agent目录含完整推导、先例与数值检查。实际源缓存、梯度bank和模型仍在服务器，未将权重或大型.pt推入公开Git。实验位置为/data/shaozhen.liu/code/tex3d/experiments/refcloth_math/20261031。</p>
<p>第一轮收集之后补充了实时输入一致性审查，并重新收集。复查中旧obs与实时图像差异均为0；首轮结果被保留，不重复计入样本。共享XML原本为patched，结束时应保持原状态；所有模型作业完成后才做最终同步。详见run_manifest.json和sync_manifest.json。</p>
<ul>SOURCES</ul>
</main></body></html>'''
    body=body.replace('RAW_PASS',str(raw)).replace('NORM_PASS',str(normalized))
    body=body.replace('GEOMETRY_TABLE',geometry).replace('FIDELITY_TABLE',fidelity).replace('UNCERTAINTY_TABLE',uncertainty).replace('SOURCES',source_html)
    (folder/'数学探索报告.html').write_text(body,encoding='utf-8')
    print('math report written')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);build(ap.parse_args().folder)
