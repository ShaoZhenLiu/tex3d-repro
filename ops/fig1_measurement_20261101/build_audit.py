# -*- coding: utf-8 -*-
"""Summarize the actual-input measurements and prepare an unapplied paper patch."""
import argparse
import csv
import difflib
import hashlib
import html
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(v,p):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def main(root,paper):
    data={t:load(root/'results'/t/'measurement.json') for t in ['pi0','pi05']}
    same=all(a['model_specs']==b['model_specs'] and a['simulator_state_sha256']==b['simulator_state_sha256']
             for a,b in zip(data['pi0']['records'],data['pi05']['records']))
    assert same and all(len(x['records'])==6 and x['completed'] and not x['actions_computed'] for x in data.values())
    stats={};flat=[]
    for target,x in data.items():
        stats[target]={}
        for stage,cameras in [('native',['main','wrist']),('model_input',['base_0_rgb','left_wrist_0_rgb'])]:
            stats[target][stage]={}
            for camera in cameras:
                rows=[r[stage][camera]['thresholds']['10'] for r in x['records']]
                stats[target][stage][camera]={key:dict(median=float(np.median([r[key] for r in rows])),
                    min=float(min(r[key] for r in rows)),max=float(max(r[key] for r in rows)))
                    for key in ['marker_fraction','plain_hidden_fraction','ref_plain_fraction','hidden_missed_marker_fraction']}
                stats[target][stage][camera]['ref_plain_rms_rgb255']=float(np.median([r[stage][camera]['rms_rgb255']['ref_plain'] for r in x['records']]))
                for r in x['records']:
                    flat.append(dict(target=target,task=r['task'],initial=r['initial'],stage=stage,camera=camera,
                                     **r[stage][camera]['thresholds']['10']))
    summary=dict(unique_object_states=6,policy_configurations=['pi0','pi05'],
        same_model_image_tensors_for_both_policies=same,threshold_rgb255=10,statistics=stats,
        no_actions_computed=True,no_target_gradients=True,no_closed_loop_rollouts=True,
        all_hidden_marker_checks_exact=all(r['hidden_marker_maxdiff']==0 for x in data.values() for r in x['records']),
        all_aba_checks_exact=all(r['aba_exact'] for x in data.values() for r in x['records']),
        policy_input_path='native 256 image -> vertical+horizontal flip -> raw dictionary -> LiberoInputs -> bilinear ResizeImages(224,224) -> Observation.from_dict uint8 to float32 [-1,1] and NCHW -> actual model _preprocess_observation(train=False)',
        eval_random_crop=False,square_images_require_no_letterbox_padding=True,
        inactive_camera='right_wrist_0_rgb is dummy padding with false mask; excluded from percentages',
        cannot_replace='60 Object full-rollout episode medians from old Figure1; these are 6 paused starting states',
        paper_findings=['Figure1a is native Plain-hidden contrast, not all three measurements',
            'Figure1b is source renderer feature offset, not RefCloth-Plain image change',
            'Old Object instrumentation omitted floor_mode=True used by formal run_eval',
            'Recent geometry audit lacks hidden and stops at raw 256 inputs',
            'Two local plot scripts overwrite motivation_exposure with incompatible 32-state/240-rollout data'])
    old=load(root/'paper_evidence/object_path_production_previous.json')
    deviations=[]
    for new in data['pi05']['records']:
        previous=next(r for r in old['records'] if (r['task'],r['initial'])==(new['task'],new['initial']))
        for camera in ['base_0_rgb','left_wrist_0_rgb']:
            deviations.append(abs(new['model_input'][camera]['rms_rgb255']['ref_plain']-
                                  previous['input_diffs']['AA']['/image/'+camera]['rms']))
    summary['max_rms_discrepancy_vs_prior_production_input_report']=max(deviations)
    dump(summary,root/'summary.json')
    with (root/'same_state_measurements.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    figdir=root/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.bbox':'tight'})
    fig,axes=plt.subplots(1,2,figsize=(10,4.2))
    metrics=[('marker_fraction','Two-color marker','#627b9e'),('plain_hidden_fraction','Plain-hidden','#c48a39'),('ref_plain_fraction','RefCloth-Plain','#246f75')]
    for ax,cam,title in zip(axes,['base_0_rgb','left_wrist_0_rgb'],['Main camera','Active wrist camera']):
        for j,(metric,label,color) in enumerate(metrics):
            values=np.array([r['model_input'][cam]['thresholds']['10'][metric] for r in data['pi05']['records']])*100
            xpos=j
            ax.scatter(np.full(6,xpos)+np.linspace(-.13,.13,6),values,s=24,color=color)
            ax.plot([xpos-.22,xpos+.22],[np.median(values)]*2,color=color,linewidth=2)
        ax.set_xticks(range(3));ax.set_xticklabels([x[1] for x in metrics],rotation=14)
        ax.set_title(title+' at actual model input');ax.set_ylim(0,65);ax.grid(axis='y',alpha=.18)
    axes[0].set_ylabel('Input-pixel support (%)')
    fig.suptitle('Same Object states; all supports use >10 RGB levels after real preprocessing',fontsize=11)
    fig.text(.5,-.03,'6 unique starting states. Pi0 and Pi0.5 produce identical image tensors; no action inference. Lines are medians.',ha='center',fontsize=8)
    fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(figdir/('same_state_support.'+ext),dpi=180)
    plt.close(fig)
    example=root/'results/pi05/object_t0_i10'
    archive=np.load(str(example/'final_model_rgb255.npz'))
    fig,axes=plt.subplots(2,6,figsize=(13.4,4.6))
    for row,(cam,label) in enumerate([('base_0_rgb','Main'),('left_wrist_0_rgb','Wrist')]):
        images={c:archive[c+'__'+cam] for c in ['plain','hidden','refcloth','marker_a','marker_b']}
        show=[np.rint(images[c]).clip(0,255).astype(np.uint8) for c in ['plain','hidden','refcloth']]
        show+=[np.max(np.abs(images[a]-images[b]),axis=-1)>10.0001 for a,b in [('marker_a','marker_b'),('plain','hidden'),('refcloth','plain')]]
        for col,img in enumerate(show):
            axes[row,col].imshow(img,cmap='gray' if col>=3 else None,vmin=0 if col>=3 else None,vmax=1 if col>=3 else None)
            axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if row==0:axes[row,col].set_title(['Plain','Cloth hidden','RefCloth','Marker support','Plain-hidden','Ref-Plain'][col],fontsize=10)
        axes[row,0].set_ylabel(label)
    fig.suptitle('Object task0 / initial10: actual 224 x 224 input images and matched support masks',fontsize=11)
    fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(figdir/('same_state_example.'+ext),dpi=170)
    plt.close(fig)
    # Reviewable minimal correction. No file in the original paper is modified.
    proposal=root/'proposed_changes';(proposal/'figures').mkdir(parents=True,exist_ok=True)
    replacements={}
    intro=(paper/'sec/1_intro.tex').read_text(encoding='utf-8')
    new_caption=r'\caption{\textbf{Empirical motivation.} (a) Native main-view Plain--hidden RGB-difference support over 180 complete Plain rollouts on Spatial, Goal, and Long, before policy preprocessing. Points and bars summarize episode medians (median and interquartile range; 30 episodes per target--suite). This contrast support is distinct from geometric visibility. (b) Across 32 fixed source states, the unchanged cloth induces an 8.58--25.74\% relative feature offset between rendering paths. A separate fixed-state Object input audit distinguishes visibility, contrast, and texture changes (\cref{supp:motivation}).}'
    intro_new='\n'.join(new_caption if line.startswith('\\caption{\\textbf{Empirical motivation.}') else line for line in intro.splitlines())+'\n'
    old_par=next(line for line in intro.splitlines() if line.startswith('\\Cref{fig:motivation} grounds'))
    suffix=old_par[old_par.index('Second,'):]
    new_par=(r'\Cref{fig:motivation} grounds this setting in two observations. First, 180 Plain rollouts on Spatial, Goal, and Long yield median native main-view Plain--hidden RGB-difference support of 17.9--23.7\%. This operational contrast measure is computed before policy preprocessing and is not a geometric visibility mask. The Object exposure traces are excluded from this rollout summary because their cloth-placement protocol differs from formal deployment; a same-state input audit is reported in \cref{supp:motivation}. '+suffix)
    intro_new=intro_new.replace(old_par,new_par);replacements['sec/1_intro.tex']=intro_new
    supp=(paper/'sec/X_suppl.tex').read_text(encoding='utf-8')
    start=supp.index('\\paragraph{Full-rollout background exposure.}');end=supp.index('\\paragraph{Unchanged-cloth reference offset.}',start)
    section=supp[start:end]
    section=section.replace('We record 240 Plain rollouts: two checkpoint configurations, four suites,','We retain 180 Plain rollouts: two checkpoint configurations, three suites (Spatial, Goal, and Long),')
    section=section.replace('Full-rollout cloth exposure.','Native Plain--hidden contrast support.')
    section='\n'.join(line for line in section.splitlines() if '& Object &' not in line)+'\n'
    old_conclusion=next(line for line in section.splitlines() if line.startswith('Main-view support exceeds'))
    section=section.replace(old_conclusion,'Main-view contrast support exceeds 1\\% at every recorded query in these three suites. The 60 archived Object exposure rollouts used a different cloth-placement dispatch and are excluded; their old 1.45--1.47\\% main-view and zero median wrist values do not estimate exposure under formal Object deployment.')
    st=stats['pi05']['model_input'];main=st['base_0_rgb'];wrist=st['left_wrist_0_rgb']
    extra=(r'\paragraph{Same-state Object input audit.} We pause six Object states (tasks $\{0,4,8\}$, initial states 10 and 11, after ten initialization actions), use the production centroid floor placement, and render the same state with two contrasting cloth colors, Plain, cloth hidden, and RefCloth. Each RGB observation follows the actual policy input transform and model preprocessing before differences are measured. At a common threshold of ten RGB levels, median marker / Plain--hidden / RefCloth--Plain supports are '
        +'%.2f / %.2f / %.2f\\%% in the main view and %.2f / %.2f / %.2f\\%% in the active wrist view. '%tuple(100*stt[k]['median'] for stt in [main,wrist] for k in ['marker_fraction','plain_hidden_fraction','ref_plain_fraction'])
        +r'The two policy configurations produce identical image tensors for these fixed states. Marker-defined support is not semantic segmentation; contrast support can omit visible cloth pixels. These six starting states do not replace full-rollout statistics, and no target actions or gradients are computed. This audit does not establish heightened action sensitivity.'+'\n\n')
    supp_new=supp[:start]+section+extra+supp[end:];replacements['sec/X_suppl.tex']=supp_new
    legacy=(paper/'scripts/plot_motivation.py').read_text(encoding='utf-8')
    legacy_new=legacy.replace("a=pd.read_csv(R/'data/motivation_mask_coverage.csv')\n",'')
    legacy_new=legacy_new.replace(" (a,'mask_fraction','(a) Background remains in view','Mask area (%)',(0,32),'motivation_exposure'),\n",'')
    legacy_new=legacy_new.replace('Rebuild the two independent motivation panels from archived measurements.','Rebuild the 32-state reference panel only. Exposure is owned by plot_stage_results.py.')
    legacy_new=legacy_new.replace("print('Two motivation panels use 32 archived states; font:',font)","print('Reference panel only; does not overwrite the rollout exposure figure; font:',font)")
    replacements['scripts/plot_motivation.py']=legacy_new
    stage=(paper/'scripts/plot_stage_results.py').read_text(encoding='utf-8')
    stage_new=stage.replace("s=pd.read_csv(data/'exposure_summary_verified.csv')","s=pd.read_csv(data/'exposure_summary_verified.csv')\ns=s[s.suite!='object'].copy()  # excluded until aligned Object rollout measurement")
    stage_new=stage_new.replace("suites=['spatial','object','goal','long'];xs=np.arange(4)","suites=['spatial','goal','long'];xs=np.arange(3)")
    stage_new=stage_new.replace("['Spatial','Object','Goal','Long']","['Spatial','Goal','Long']")
    stage_new=stage_new.replace('Cloth mask coverage (%)','Plain-hidden RGB support (%)').replace('(a) Exposure across 240 Plain rollouts','(a) Native RGB support across 180 Plain rollouts')
    replacements['scripts/plot_stage_results.py']=stage_new
    patch=[]
    for rel,text in replacements.items():
        dest=proposal/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text,encoding='utf-8')
        original=(paper/rel).read_text(encoding='utf-8')
        patch+=difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile='a/'+rel,tofile='b/'+rel)
    (proposal/'figure1_correction.patch').write_bytes(''.join(patch).encode('utf-8'))
    # Corrected exposure panel uses only the existing, retained 180 rollouts.
    with (paper/'data/four_stages/exposure_summary_verified.csv').open(encoding='utf-8') as f:old_rows=list(csv.DictReader(f))
    fig,ax=plt.subplots(figsize=(3.35,1.55));suites=['spatial','goal','long'];xpos=np.arange(3)
    for target,offset,marker,label in [('pi0',-.08,'o',r'$\pi_0$'),('pi05',.08,'s',r'$\pi_{0.5}$')]:
        selected=[next(r for r in old_rows if r['target']==target and r['suite']==s) for s in suites]
        y=np.array([float(r['agentview_median']) for r in selected])*100
        err=np.array([y-np.array([float(r['agentview_p25']) for r in selected])*100,np.array([float(r['agentview_p75']) for r in selected])*100-y])
        ax.errorbar(xpos+offset,y,yerr=err,fmt=marker,ms=4,capsize=2,label=label)
    ax.set_xticks(xpos);ax.set_xticklabels(['Spatial','Goal','Long']);ax.set_ylim(0,30);ax.set_ylabel('RGB support (%)',fontsize=8)
    ax.set_title('(a) Native RGB support: 180 Plain rollouts',fontsize=8);ax.legend(frameon=False,fontsize=7,ncol=2);fig.tight_layout(pad=.2)
    for ext in ['pdf','png']:fig.savefig(proposal/'figures'/('motivation_exposure.'+ext),dpi=220)
    plt.close(fig)
    # Plain-language report, separate from the unchanged paper.
    table=''
    for stage,cams in [('native',['main','wrist']),('model_input',['base_0_rgb','left_wrist_0_rgb'])]:
        for camera in cams:
            r=stats['pi05'][stage][camera]
            table+='<tr><td>%s</td><td>%s</td>'%(stage,camera)+''.join('<td>%.2f%%</td>'%(100*r[k]['median']) for k in ['marker_fraction','plain_hidden_fraction','ref_plain_fraction','hidden_missed_marker_fraction'])+'</tr>'
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>图一测量口径核对</title><style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.8 "Microsoft YaHei",sans-serif;color:#26383e}h1{font-size:27px}h2{font-size:21px;margin-top:32px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #d7e0e4;padding:9px;text-align:left}th{background:#edf3f5}img{max-width:100%}.note{border-left:4px solid #26727a;background:#edf5f5;padding:16px}</style>
<h1>图一与近期汇报：测量口径核对</h1><p>2026-11-01 · 仅测量校准，无新攻击训练、无目标动作或梯度。</p>
<p class="note"><strong>确实没有对齐。</strong>图一(a)是旧仪表化完整轨迹的 Plain／hidden 差分支持；图一(b)是源模型跨渲染路径的特征偏移；近期报告主要测颜色标记及 RefCloth／Plain 图像变化。原图中的 Object 约 1.5%／腕相机 0 数据还使用了与正式部署不同的铺布分支，不能继续支撑当前部署“低暴露但强攻击”的解释。</p>
<h2>分别测了什么</h2><table><tr><th>材料</th><th>对象与比较</th><th>处理域与统计</th></tr><tr><td>当前图一(a)</td><td>Plain − cloth-hidden，阈值10</td><td>256×256 原生主相机；240条完整轨迹，回合中位数再取中位数；非最终模型输入</td></tr><tr><td>当前图一(b)</td><td>同一素色布：可微渲染与原生渲染的源特征差</td><td>32个 Spatial 源状态；特征偏移，不是 RefCloth−Plain 像素差</td></tr><tr><td>近期目标几何检查 geometry_audit.py</td><td>颜色标记可见率与 RefCloth−Plain</td><td>起始状态；native/raw 256图像；没有 hidden 对照，也没有完整的最终目标模型输入覆盖率。更早的生产路径报告已有224图像RMS，本次补齐三种覆盖率。</td></tr><tr><td>本次补测</td><td>两色标记、Plain−hidden、RefCloth−Plain</td><td>相同暂停状态、双相机，逐条件经过真实输入处理后比较；6个固定场景状态，两种策略配置</td></tr></table>
<h2>旧 Object 数值的问题</h2><p>归档仪表化代码 setup_episode 调用 align_cloth_to_table(env)，遗漏 floor_mode=True；正式 run_eval 对 Object 则显式走 floor_mode=True。无 table_top 时，旧默认分支可能把布对齐到射线找到的物体高台。例如原先同一 t0/i10 被放到约 (−0.6,0,0.371)，正式部署则为约 (−0.053,−0.030,0.005)。这首先是协议不一致，不能只归因于 RGB 阈值。旧字段 target-name 身份修正没有检查或修复这个问题。</p>
<p>当前 PDF 已直接核对：第一页确实显示 240 轨迹图；第二页及附录仍保留约1.5%和零腕相机解释。此次发现定位于暴露仪表化；正式攻击评测代码有 Object 铺布分支，本核查没有据此推翻主成功率表。</p>
<h2>同状态补测结果</h2><p>Object任务0/4/8，初态10/11，等待10步后暂停，使用正式质心贴地铺布。以下是6个状态的中位数，所有支持都采用同一“最大RGB通道差 &gt;10”口径；标记支持是颜色标记定义的几何代理，不是语义分割真值。两个策略得到逐像素相同的输入图像，不能将其当作12个独立状态。</p>
<table><tr><th>阶段</th><th>相机</th><th>标记支持</th><th>Plain−hidden</th><th>RefCloth−Plain</th><th>差分漏掉的标记区域</th></tr>'''+table+'''</table>
<p><img src="figures/same_state_support.png"></p><p><img src="figures/same_state_example.png"></p>
<p>最终输入中，Plain／hidden 阈值漏掉约19.8%的主相机标记区域、11.0%的腕相机标记区域，确实不等同于几何可见性。但这批真实输入中的 RefCloth 变化覆盖约11.9%／54.0%，并不支持先假定“真实输入几乎不变”。本轮没有计算动作，因此不增加视觉敏感性或因果重要性的结论。</p>
<h2>真实输入路径</h2><p>native 256渲染 → 与生产一致的上下及左右翻转 → build_raw_inputs → LiberoInputs → 双线性 resize_with_pad 到224 → Observation.from_dict 转换为 NCHW、float32、[−1,1] → 实际模型 _preprocess_observation(train=False)，随后本应进入 embed_prefix。本次在该输入位置停止，没有调用编码器、扩散采样或动作头。正方形图像不需额外补边；推理路径没有随机裁剪。两路真实相机均启用，第三路 right_wrist 是 mask=False 的填充图，不参与统计。</p>
<p>隐藏有效性已用两种颜色在 hidden 条件下图像完全相同核验；各条件切换均通过 A→B→A 原生像素还原检查，物理状态、指令和 proprioception 一致。阈值1/5/10/20的全部结果及输入张量哈希保留在原始 JSON 中。新 RefCloth／Plain 输入RMS与此前生产路径报告的最大差为 '''+str(max(deviations))+'''，说明该部分近期报告可复现。</p>
<h2>还发现一个出图入口冲突</h2><p>scripts/plot_motivation.py 使用32个源状态的旧 mask_fraction，却写入 motivation_exposure.pdf；scripts/plot_stage_results.py 则把240轨迹结果写入同名文件。前者再运行会覆盖当前图一(a)，造成图和240轨迹图注不一致。修订草案让前者只生成参考特征子图，暴露图由后者统一生成。</p>
<h2>最小修订建议与已准备的草案</h2><p>先撤下原 Object 约1.5%／腕相机0对应的当前部署解释。若暂不重采完整轨迹，图一(a)可仅保留其他三套件已有的180条轨迹，并明确其为原生RGB差分支持；图一(b)保持独立的32状态特征问题。新的Object同状态三测量放入附录，清楚标注6状态，不能将其直接替换成60条Object完整轨迹统计。</p>
<p>proposed_changes/ 提供未应用的 TEX/Python 修改、统一diff和一个180轨迹的替代子图。原论文目录、当前主PDF与图一均未覆盖。若要恢复完整四套件暴露图，仍需只重采对齐后的60条Object Plain轨迹，统一记录模型实际输入；不需要重训纹理或重新跑攻击条件。</p></html>'''
    (root/'图一测量核对报告.html').write_text(page,encoding='utf-8')
    dump(dict(paper_files_unchanged=True,proposal_applied=False,source_paper=str(paper),
        source_file_hashes={str(p.relative_to(paper)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [paper/'main.pdf',paper/'sec/1_intro.tex',paper/'sec/X_suppl.tex',paper/'figures/motivation_exposure.pdf']},
        interpretation='Correct the old Object visibility explanation; do not replace rollout statistics with six starting states; do not infer tiny input changes or heightened action sensitivity.'),root/'decisions.json')
    print(json.dumps(dict(same_policy_image_tensors=same,model_statistics=stats['pi05']['model_input'],previous_rms_maxdiff=max(deviations))))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('paper',type=Path)
    a=ap.parse_args();main(a.root,a.paper)
