# -*- coding: utf-8 -*-
"""Standalone scientific plots and a Chinese report from native raw results."""
import argparse
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


def main(root,delivery):
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    meta=json.loads((root/'frozen_candidates.json').read_text(encoding='utf-8'))
    val=summary['summaries']['validation']
    figdir=delivery/'figures';figdir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'savefig.bbox':'tight'})
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.0))
    x=np.array([25,50,75,100])
    families=[('greedy','Native action greedy','#176983'),('ranking','One-shot ranking','#8b5aa8'),
              ('visibility','Native visibility','#c88725')]
    for family,label,color in families:
        points=[val['%s_%02d'%(family,k)] for k in [4,8,12]]+[val['full']]
        for ax,key in zip(axes,['pooled_relative_fidelity','pooled_energy_retention']):
            ax.plot(x,[p[key] for p in points],'-o',label=label,color=color,linewidth=1.6,markersize=4)
    for ax,key in zip(axes,['pooled_relative_fidelity','pooled_energy_retention']):
        random=np.array([[val['random%d_%02d'%(s,k)][key] for s in range(101,106)] for k in [4,8,12]])
        ax.plot(x[:3],np.median(random,axis=1),'--o',color='#6d747b',label='Random: median of 5 masks',markersize=4)
        ax.fill_between(x[:3],random.min(axis=1),random.max(axis=1),color='#9aa1a8',alpha=.18,label='Random: min/max')
        ax.set_xlabel('Retained cloth surface (%)');ax.set_xticks(x);ax.grid(alpha=.18)
    axes[0].set_ylabel('Action fidelity error / full attack signal energy')
    axes[0].set_title('Held-out native source fidelity (lower is better)')
    axes[0].axhline(.1,linestyle=':',color='#ba4b47',linewidth=1)
    axes[1].set_ylabel('Action displacement energy / full attack energy')
    axes[1].set_title('Preserved action magnitude (direction checked separately)')
    axes[1].axhline(1,linestyle=':',color='#30383e',linewidth=1)
    axes[1].legend(frameon=False,fontsize=8)
    fig.text(.5,-.035,'16 fixed source states, 8 task trajectories; source actions only. Random bands are mask ranges, not confidence intervals.',ha='center',fontsize=8)
    fig.tight_layout()
    for ext in ['png','pdf']: fig.savefig(figdir/('source_fidelity.'+ext),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(10.6,3.3))
    ranking=summary['ranking']
    for ax,key,title in zip(axes,['fit_scores','validation_scores','visibility'],
                           ['Fit deletion influence','Held-out deletion influence','Native camera exposure']):
        a=np.array(ranking[key]).reshape(4,4)
        ax.imshow(a,cmap='viridis');ax.set_title(title);ax.set_xticks([]);ax.set_yticks([])
        for j in range(16): ax.text(j%4,j//4,str(j),ha='center',va='center',color='white',fontsize=9)
    fig.suptitle('Block indices follow exported PNG rows; these are not task-failure attributions',fontsize=10)
    fig.tight_layout()
    for ext in ['png','pdf']: fig.savefig(figdir/('surface_regions.'+ext),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(11.6,3.1))
    for ax,name,label in zip(axes,['full','greedy_12','greedy_08','greedy_04'],['Full texture','Greedy: 75%','Greedy: 50%','Greedy: 25%']):
        ax.imshow(Image.open(root/'textures'/(name+'.png')));ax.axis('off');ax.set_title(label)
    fig.tight_layout()
    for ext in ['png','pdf']: fig.savefig(figdir/('frozen_textures.'+ext),dpi=180)
    plt.close(fig)
    rows=''
    for k in [12,8,4]:
        g=val['greedy_%02d'%k]; v=val['visibility_%02d'%k]
        rr=[val['random%d_%02d'%(s,k)]['pooled_relative_fidelity'] for s in range(101,106)]
        rows+='<tr><td>%d%%</td><td>%.4f</td><td>%.4f</td><td>%.4f</td><td>%.1f%%</td><td>%.4f</td></tr>'%(
            k/16*100,g['pooled_relative_fidelity'],float(np.median(rr)),v['pooled_relative_fidelity'],
            g['pooled_energy_retention']*100,g['mean_displacement_cosine'])
    gate_rows=''
    for g in summary['gates']:
        gate_rows+='<tr><td>%d%%</td><td>%s</td><td>%s</td></tr>'%(g['area_fraction']*100,
            '通过' if g['pass_gate'] else '未通过','；'.join(k for k,v in g['flags'].items() if not v) or '全部满足')
    conclusion=('至少一个保留不超过一半区域的候选通过预设源侧门槛。需冻结所有候选后继续目标模型闭环评测；目前尚不能声称迁移攻击性能保持。'
                if summary['source_gate_passed'] else
                '没有保留不超过一半区域的候选通过预设源侧门槛。当前结果不足以支持“可迁移且保真精简”的新增方法主张；本轮不据此启动目标闭环大评测。')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>RefCloth 固定表面区域精简探索</title>
<style>body{max-width:1080px;margin:40px auto;padding:0 24px;color:#24343c;font:16px/1.75 "Microsoft YaHei",sans-serif}h1{font-size:28px}h2{margin-top:34px;font-size:21px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #d7e0e4;padding:9px;text-align:left}th{background:#edf3f5}.box{background:#edf5f6;border-left:4px solid #176983;padding:16px}img{max-width:100%}code{background:#eef1f2;padding:2px 5px}</style>
<h1>RefCloth：冻结颜色后的固定表面区域精简</h1><p>2026-11-01 · 独立实验分支 support-20261101 · 原论文及主结果未改写</p>
<div class="box">'''+html.escape(conclusion)+'''</div>
<h2>实验回答的范围</h2><p>保持同一静态共享桌布和原来的操控任务，只对已有 PNG 做区域恢复：保留区域使用原纹理颜色，删除区域恢复成底布。掩码对两路相机、所有状态和任务相同，不做图像平面贴块，不重新优化颜色、特征或策略。</p>
<p>4×4 表面网格；4 个选区任务 × 2 个时间点，共 8 个选区状态；更换初始状态后在 8 个任务 × 2 个时间点上验证，共 16 个状态，其中 4 个任务未参与选区。16 个状态不是 16 个独立任务，两个时间点也不是独立轨迹。</p>
<p>源模型为冻结的 OpenVLA-OFT。目标策略未参与选区，当前表格也不是目标模型闭环结果。GPU 6、7 并行；未使用 GPU 4。</p>
<h2>定量结果</h2><p>动作保真误差以“完整攻击相对底布的源动作变化能量”为分母，0 代表复现完整纹理动作；底布对照为 1。采用前 5 步的 6 个归一化位姿维度，平移和旋转等权，夹爪另报。动作幅度保持率不是任务失败率。</p>
<table><tr><th>保留面积</th><th>贪心动作保真误差 ↓</th><th>随机 5 掩码中位数 ↓</th><th>可见面积选区误差 ↓</th><th>贪心动作能量保持</th><th>贪心方向余弦</th></tr>'''+rows+'''</table>
<p><img src="figures/source_fidelity.png"></p><p><img src="figures/frozen_textures.png"></p>
<h2>预设门槛</h2><p>保留 50% 或 25% 面积时，汇总保真误差 ≤0.10，至少 75% 状态的误差 ≤0.25，动作能量保持 ≥80%，平均方向余弦 ≥0.90；误差需比随机区域中位数低至少 20%，且不比可见面积基线差超过 5%。原生重复前向及纹理还原要求精确一致。所有掩码在打开源侧验证状态前冻结。</p>
<table><tr><th>面积</th><th>总门槛</th><th>未满足项（原始键）</th></tr>'''+gate_rows+'''</table>
<h2>区域稳定性与混杂</h2><p>选区与验证的单块删除排序 Spearman = '''+str(ranking['fit_validation_spearman'])+'''；选区动作影响与相机可见面积排序相关 = '''+str(ranking['fit_action_vs_visibility_spearman'])+'''。删除一块后的动作变化不等于该区域对最终任务失败的因果贡献。区域之间存在非加性交互，所以贪心过程每次删除后重新计算，而不是把单块分数相加。</p>
<p><img src="figures/surface_regions.png"></p><p>所有二值对照使用相同网格、同样的硬边界；导出 PNG 的 RMS、TV、有效扰动像素、边界长度见 texture_manifest.json。每块的真实原生颜色标记可见率见状态文件，不能把 UV 面积当成相机像素面积；不同块的可见率相加在插值边缘会有重叠。尚未额外匹配不同掩码的扰动能量或精确连通边界。</p>
<h2>数学解释与创新性边界</h2><p>令 d=W(a*−a0)，e=W(aw−a*)。若单个状态满足 ||e||²≤τ||d||²，三角不等式给出 (1−√τ)²≤||d+e||²/||d||²≤(1+√τ)²；τ&lt;1 时方向余弦至少 √(1−τ)。汇总误差只能约束拼接后的动作向量，不能替代逐状态检查。这是标准几何事实，不是新的泛化定理，也不保证闭环任务成功率。</p>
<p>面积约束和响应保留已有 <a href="https://arxiv.org/abs/1910.08485">Extremal Perturbations</a>；视觉区域对动作的干预已有 <a href="https://proceedings.mlr.press/v306/zhang26gz.html">Embodied Interpretability</a>；物理攻击蒸馏已有 <a href="https://arxiv.org/abs/2501.02232">Distillation-Enhanced Physical Adversarial Attacks</a>。因此仅有掩码、贪心和“保住旧纹理”的设计不足以构成新贡献。潜在价值需要来自固定场景表面区域在源侧确定后仍能跨任务、跨策略保持攻击的实验证据。</p>
<h2>接口修正与复现材料</h2><p>首次冒烟时，同一物理状态在 mj_step 后与恢复后重新 forward 的派生运动学不一致，被像素断言拦截。现统一在缓存基准画面前执行 forward，并刷新 proprioception。第二次冒烟及正式采集的底布/完整纹理重复动作差异均为 0；所有原生查询执行 A→B→A 像素检查。第一次失败日志保留，未当作算法结果。</p>
<p>代码见 code/；逐次贪心查询、冻结候选、选区/验证状态及结果见 results/；protocol.json 是预先固定的规则。原始服务器路径：<code>/data/shaozhen.liu/code/tex3d/experiments/refcloth_support/20261101</code>。没有改写论文主表，没有把原方法的攻击率复制到精简版本。</p></html>'''
    (delivery/'区域精简探索报告.html').write_text(page,encoding='utf-8')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('root',type=Path)
    ap.add_argument('delivery',type=Path)
    a=ap.parse_args();main(a.root,a.delivery)
