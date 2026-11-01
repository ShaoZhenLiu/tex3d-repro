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
    max_influence=max(max(ranking['fit_scores']),max(ranking['validation_scores']))
    for ax,key,title in zip(axes,['fit_scores','validation_scores','visibility'],
                           ['Fit deletion influence','Held-out deletion influence','Native camera exposure']):
        a=np.array(ranking[key]).reshape(4,4)
        im=ax.imshow(a,cmap='viridis',vmin=0,vmax=(max_influence if key!='visibility' else None))
        fig.colorbar(im,ax=ax,fraction=.045,pad=.03)
        ax.set_title(title);ax.set_xticks([]);ax.set_yticks([])
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
    pilot_section=''
    pilot_file=delivery/'results'/'pilot_summary.json'
    if pilot_file.exists():
        pilot=json.loads(pilot_file.read_text(encoding='utf-8'))
        pooled=pilot['suites']['pooled_fixed_cases']
        conclusion=('工程方案可运行，但本轮不支持作为主创新点：50%/25% 区域未通过源侧保真门槛；75% 虽在源侧较保真，'
                    '目标闭环中完整纹理使机器人失败 {0}/12 次，动作选区为 {1}/12 次，可见面积选区为 {2}/12 次。'
                    '动作选区在 2 对回合中变弱，没有变强的配对。小样本不能给出精确总体降幅，也不能据此声称性能维持。').format(
                        pooled['full']['attack_failures_of_robot'],pooled['greedy_12']['attack_failures_of_robot'],
                        pooled['visibility_12']['attack_failures_of_robot'])
        fig,axes=plt.subplots(1,2,figsize=(10.6,4.0))
        condition_order=['plain','full','greedy_12','visibility_12','random101_12','random102_12']
        labels=['Plain','Full','Action 75%','Visible 75%','Random101','Random102']
        colors=['#b9c2c7','#b45147','#176983','#c88725','#748773','#9aa68d']
        for ax,suite,title in zip(axes,['libero_spatial','libero_object'],['Spatial','Object']):
            vals=[pilot['suites'][suite][c]['attack_failures_of_robot'] for c in condition_order]
            bars=ax.bar(range(6),vals,color=colors,width=.68)
            for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+.09,str(v),ha='center',fontsize=10)
            ax.set_xticks(range(6));ax.set_xticklabels(labels,rotation=28,ha='right')
            ax.set_ylim(0,6.3);ax.set_yticks(range(7));ax.set_title(title+' (6 paired cases)');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        axes[0].set_ylabel('Robot failures (attack outcomes), out of 6')
        fig.suptitle('Frozen surface masks: exploratory native closed-loop target test',fontsize=11)
        fig.text(.5,-.04,'72 valid episodes total. Three tasks per suite; no target-driven mask revision. Counts do not establish equivalence.',ha='center',fontsize=8)
        fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(figdir/('target_pilot.'+ext),dpi=180)
        plt.close(fig)
        prows=''
        for suite in ['libero_spatial','libero_object']:
            outcomes=pilot['suites'][suite]
            for condition,r in outcomes.items():
                prows+='<tr><td>%s</td><td>%s</td><td>%d/%d</td><td>%+.1f pp</td><td>%d</td><td>%d</td></tr>'%(
                    suite,condition,r['attack_failures_of_robot'],r['n'],100*r['paired_attack_rate_difference_vs_full'],
                    r['degraded_pairs'],r['improved_pairs'])
        pilot_section='''<h2>额外的 75% 区域目标闭环试验</h2><p>这一部分在看过源侧结果后另行登记：50%/25% 的原门槛未通过，而 75% 满足保真和对照要求，故增加探索性试验。它没有被重标成“通过原门槛”。所有掩码在查询任何目标策略前冻结；此后没有修改掩码、任务、噪声或候选条件。</p>
<p>π0.5，Spatial/Object 各 3 个任务 × 2 个初始状态 × 6 种纹理，共 72 回合；10 步去噪、5 步重规划、400 步上限。包括源侧保真优于动作选区的 random102。Object 使用生产质心铺布，不使用 fixed-floor 变体。按 suite/task/initial/query-index 显式传入同一噪声，并核对重复推理与上传。</p>
<table><tr><th>套件</th><th>纹理</th><th>机器人失败/回合数</th><th>相对完整纹理的攻击率差</th><th>变弱的配对</th><th>变强的配对</th></tr>'''+prows+'''</table>
<p><img src="figures/target_pilot.png"></p><p>机器人失败才计作表中的攻击结果；异常、纹理上传错误不计作攻击成功。每套件只有 6 对回合，3 个任务，不能证明统计非劣效，也不能代表完整 suite。pilot_summary.json 给出配对差值及独立回合假设下的保守 Hoeffding 区间；若 6 对均未出现变弱，也不能排除较大的真实退化概率。原论文主表没有被替换。</p>
<p><strong>该试验不支持“源侧动作保真选区可靠地保持目标攻击”的主张。</strong>源侧动作选区不优于目标闭环中的可见面积选区，也没有优于所有固定随机对照。可见面积选区的合并失败次数虽与完整纹理相同，Spatial 中仍有 1 对变弱、1 对变强，不能解释为逐回合无损。50% 和 25% 版本未做目标闭环；源动作变化下降不能据此当作它们闭环攻击失效的证据。</p>'''
    geometry_section=''
    geo_file=delivery/'results'/'geometry'/'geometry.json'
    if geo_file.exists():
        geo=json.loads(geo_file.read_text(encoding='utf-8'))['records'];grows=''
        for suite in ['libero_spatial','libero_object']:
            for condition in ['full','greedy_12','visibility_12','random101_12','random102_12']:
                a=np.mean([r['native_visible_fraction'] for r in geo if r['suite']==suite and r['candidate']==condition],axis=0)
                grows+='<tr><td>%s</td><td>%s</td><td>%.2f%%</td><td>%.2f%%</td></tr>'%(suite,condition,100*a[0],100*a[1])
        geometry_section='''<h2>原生可见面积核对</h2><p>在目标测试同一组初始状态上，以原生表面颜色标记实测被保留区域的可见像素比例。只测等待 10 步后的起始画面，不能替代整回合暴露统计；此处未加载或查询目标策略。GPU 0 仅执行这项轻量渲染。</p><table><tr><th>套件</th><th>纹理区域</th><th>主相机</th><th>腕相机</th></tr>'''+grows+'''</table><p>UV 面积减少 25% 并不表示两路输入的扰动面积都减少 25%。纹理 RMS、边界以及曝光仍是需要控制的混杂因素，不能把有限的比较全部归因于选区规则。</p>'''
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
'''+pilot_section+geometry_section+'''<h2>数学解释与创新性边界</h2><p>令 d=W(a*−a0)，e=W(aw−a*)。若单个状态满足 ||e||²≤τ||d||²，三角不等式给出 (1−√τ)²≤||d+e||²/||d||²≤(1+√τ)²；τ&lt;1 时方向余弦至少 √(1−τ)。汇总误差只能约束拼接后的动作向量，不能替代逐状态检查。这是标准几何事实，不是新的泛化定理，也不保证闭环任务成功率。</p>
<p><strong>75% 面积时，逐次重算的贪心方法与一次性单块排序导出的是同一掩码。</strong>不能声称额外的贪心迭代在该候选上提供了独立收益。较小面积的一次性排序在部分源侧指标上还优于贪心，详见原始 summary.json。五个随机掩码中，random102 的源侧保真误差为 0.0287，优于 greedy75 的 0.0427；“优于随机中位数”并不等于“优于所有随机区域”。</p>
<p>面积约束和响应保留已有 <a href="https://arxiv.org/abs/1910.08485">Extremal Perturbations</a>；视觉区域对动作的干预已有 <a href="https://proceedings.mlr.press/v306/zhang26gz.html">Embodied Interpretability</a>；物理攻击蒸馏已有 <a href="https://arxiv.org/abs/2501.02232">Distillation-Enhanced Physical Adversarial Attacks</a>。因此仅有掩码、贪心和“保住旧纹理”的设计不足以构成新贡献。潜在价值需要来自固定场景表面区域在源侧确定后仍能跨任务、跨策略保持攻击的实验证据。</p>
<h2>接口修正与复现材料</h2><p>首次冒烟时，同一物理状态在 mj_step 后与恢复后重新 forward 的派生运动学不一致，被像素断言拦截。现统一在缓存基准画面前执行 forward，并刷新 proprioception。第二次冒烟及正式源侧采集的底布/完整纹理重复动作差异均为 0；源侧固定状态查询和几何检查执行 A→B→A 像素检查。目标闭环则逐回合检查原生上传及首个查询的同噪声重复推理，最大动作差异为 0。第一次失败日志保留，未当作算法结果。</p>
<p>代码见 code/；逐次贪心查询、冻结候选、选区/验证状态及结果见 results/；protocol.json 是预先固定的规则。原始服务器路径：<code>/data/shaozhen.liu/code/tex3d/experiments/refcloth_support/20261101</code>。没有改写论文主表，没有把原方法的攻击率复制到精简版本。</p></html>'''
    (delivery/'区域精简探索报告.html').write_text(page,encoding='utf-8')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('root',type=Path)
    ap.add_argument('delivery',type=Path)
    a=ap.parse_args();main(a.root,a.delivery)
