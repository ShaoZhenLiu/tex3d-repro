"""Chinese, self-contained evidence report for the completed Object investigation."""
import csv
import html
import json
from pathlib import Path
import numpy as np


ROOT=Path(__file__).resolve().parents[1]


def pct(x):return '—' if x is None else '%.2f%%'%(100*x)
def stat(x):return '%s [%s, %s]'%(pct(x['median']),pct(x['p25']),pct(x['p75']))
def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+str(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table>'


def main():
    s=json.loads((ROOT/'summary.json').read_text(encoding='utf-8'))
    checks=json.loads((ROOT/'final_integrity_checks.json').read_text(encoding='utf-8'))
    archive=json.loads((ROOT/'evidence/archived_object_counts.json').read_text())
    episodes=list(csv.DictReader((ROOT/'episode_outcomes.csv').open(encoding='utf-8-sig')))
    a=s['attack'];parts=[]
    parts.append('<h1>Object 到底是什么情况：完整轨迹与成对攻击复测</h1>')
    parts.append('<p class="meta">2026-11-02 · 固定原始 RefCloth · 两种策略 × 10 个 Object 任务 × 3 个初态 × Plain / RefCloth = 120 回合</p>')
    parts.append('<div class="finding"><b>核心判断：</b>旧图一中 Object 的约 1.5% 主相机暴露和零腕部暴露，不能代表正式铺布条件。位置对照已复现其来源；本次使用正式贴地位置、实际缓存相机帧和真实模型输入重测完整轨迹。攻击是否仍有效，按下面独立的成对结果判断。</div>')
    rows=[]
    for t in ['pi0','pi05']:
        x=a[t];rows.append([t,'%d / %d'%(x['plain_success'],x['n']),'%d / %d'%(x['ref_success'],x['n']),
            '%.1f pp'%(x['net_attack_gain']*100),x['induced_failures'],x['recovered_failures'],'%.3g'%x['mcnemar_p_holm']])
    parts.append('<h2>原纹理的净攻击效果</h2>')
    parts.append(table(['策略','Plain 机器人成功','RefCloth 机器人成功','净攻击增益 G','Plain 成功→Ref 失败','Plain 失败→Ref 成功','精确 McNemar p（Holm）'],rows))
    parts.append('<p>G = Plain 成功率 − RefCloth 成功率。自然失败没有全部归到攻击上；表内还单列攻击诱发失败和恢复成功的回合。两条件使用相同初态、相同逐次查询噪声；纹理、10 步去噪、每 5 动作重规划、10 步初始化与 400 动作上限均固定。</p>')
    parts.append('<img src="figures/paired_attack_results.png" alt="成对攻击结果">')
    parts.append('<h2>策略真正看到的三种变化</h2>')
    rows=[]
    for t in ['pi0','pi05']:
        for view,label in [('base_0_rgb','主相机'),('left_wrist_0_rgb','腕部相机')]:
            z=s['exposure'][t]['model_input'][view]
            rows.append([t,label]+[stat(z['t10__'+m]) for m in ['marker_fraction','plain_hidden_fraction','ref_plain_fraction']])
    parts.append(table(['策略','真实模型输入 224×224','双色标记可见范围','|Plain−hidden|','|RefCloth−Plain|'],rows))
    parts.append('<p>原生 256×256 图像先走策略相同的双轴翻转与官方双线性缩放路径，再经过实际模型的预处理，最后在 224×224 RGB 等价尺度上作差。阈值为 max<sub>c</sub>|ΔI| &gt; 10（0–255）。先取每条轨迹全部策略查询的中位数，再对 30 条轨迹取中位数和四分位区间；方括号不是置信区间。未启用的第三路占位相机不计入。</p>')
    parts.append('<p>两个模型共享相同的 30 个任务—初态键；此后各自闭环动作形成不同轨迹。所有统计分策略报告，没有把两个模型合并成 60 个独立物理初态。</p>')
    parts.append('<img src="figures/model_input_three_measures.png" alt="真实模型输入三种测量">')
    rows=[]
    for t in ['pi0','pi05']:
        for view,label in [('base_0_rgb','主相机'),('left_wrist_0_rgb','腕部相机')]:
            z=s['exposure'][t]['model_input'][view]
            rows.append([t,label]+[stat(z['t10__'+m+'__query_fraction_gt_1pct']) for m in ['marker_fraction','plain_hidden_fraction','ref_plain_fraction']])
    parts.append(table(['策略','相机','标记范围 >1% 的查询比例','Plain−hidden >1% 的查询比例','Ref−Plain >1% 的查询比例'],rows))
    parts.append('<p>上表描述整个回合内有多少查询仍有可见变化；与前表的单帧面积占比是两个不同的量，同样先按回合计算，再对 30 回合汇总。</p>')
    rows=[]
    for t in ['pi0','pi05']:
        for view,label in [('base_0_rgb','主相机'),('left_wrist_0_rgb','腕部相机')]:
            z=s['exposure'][t]['model_input'][view]
            rms=z['ref_plain_rms_rgb255']
            rows.append([t,label,stat(z['t10__hidden_missed_marker_fraction']),
                '%.3f [%.3f, %.3f]'%(rms['median'],rms['p25'],rms['p75']),
                '%.4f'%(rms['median']/127.5)])
    parts.append(table(['策略','相机','旧颜色差分漏掉的标记范围','Ref−Plain RMS（RGB 0–255）','同一 RMS（模型 −1…1 尺度）'],rows))
    parts.append('<p>双色标记是布面可见性的渲染代理，包含遮挡和成像影响；它不是语义分割真值。Plain−hidden 依赖底色、地板纹理、光照和阈值，不能直接叫作几何面积。阈值可能漏掉一部分可见布面，但不能据此断言真实输入几乎没变。</p>')
    parts.append('<h2>旧低暴露值为什么出现</h2>')
    place=s['placement'];rows=[]
    for view,label in [('main','主相机'),('wrist','腕部相机')]:
        for mode,title in [('legacy_ray','旧定位路径'),('production_floor','正式贴地路径')]:
            z=place['native'][view][mode]
            rows.append([label,title]+[stat(z[m]) for m in ['marker_fraction','plain_hidden_fraction','ref_plain_fraction']])
    parts.append(table(['原生相机','30 个相同动态初态','双色标记','Plain−hidden','RefCloth−Plain'],rows))
    parts.append('<p>此处 LIBERO-Object 为地面操作场景，正式评测使用贴地分支。旧仪表程序缺少 Object 的 floor_mode=True。30 个初态中，旧路径有 %d 个被射线网格移到高平台，%d 个留在偏离操作区的默认贴地位置；正式路径全部位于物体附近、离地 0.005 米。这里只改变布的模型位置，没有执行策略动作，也没有把初态统计冒充完整轨迹。</p>'%(place['old_ray_platform_count'],place['old_static_floor_count']))
    parts.append('<img src="figures/geometry_initial_state_comparison.png" alt="同状态位置对照">')
    parts.append('<p>图示固定为 task 0 / init 10；热图采用同一色标。正式攻击评测原先已有 Object 贴地分支，因此旧暴露统计的问题不会自动推翻主表攻击结果。</p>')
    parts.append('<h2>与旧图一完全相同口径的原生差分统计</h2>')
    old={'pi0':.01469,'pi05':.01454};rows=[]
    for t in ['pi0','pi05']:
        for view,label in [('main','主相机'),('wrist','腕部相机')]:
            z=s['exposure'][t]['native'][view]
            rows.append([t,label,pct(old[t] if view=='main' else 0),stat(z['t10__plain_hidden_fraction'])])
    parts.append(table(['策略','相机','旧图一 Object 中位数','本次正式铺布下完整轨迹'],rows))
    parts.append('<p>本次两目标各 30 条 Plain 完整轨迹，初态编号仍为 10–12。与旧图的主相机原生差分定义和按回合聚合方式相同，但修正了 Object 铺布位置，并按实际曝光时刻测量。其他套件未在本轮重跑，不能把本轮模型输入数据推广成所有套件的新统计。</p>')
    parts.append('<img src="figures/trajectory_support_curves.png" alt="完整轨迹支持范围曲线">')
    parts.append('<h2>严格的同状态校验</h2>')
    counts=s['counts']
    parts.append('<p>本次 %d 次 Plain 查询均检查了原图 A–B–A 还原与完整积分状态不变。相机重新渲染与实际缓存图逐像素相同；两路相机曝光状态相同；60 组 Plain/Ref 首次查询的初始仿真状态、proprio 和消耗的噪声一致，且临时渲染的 Ref 输入与真正 Ref 回合首帧一致。两目标各有完整轨迹的仪表开关对照，动作与输入哈希逐次相等。</p>'%counts['plain_queries'])
    parts.append('<p>开发时第一次校验发现：查询时直接重渲染与策略缓存图最大相差 51 个灰度级。该尝试在校验门被拒绝，没有计入正式回合。当前版本在相机真正渲染时复制原生 MjData，再用副本渲染反事实颜色；不推进、不 forward 运行中的物理状态。实际缓存延迟中位数为 %.1f 毫秒。</p>'%(counts['camera_delay_seconds']['median']*1000))
    parts.append('<p>原型失败记录保留在 raw/diagnostics/failed_v1。额外比较确认，相机状态复制钩子本身也没有改变原始生产轨迹。RMS、支持范围与动作差异是描述量，本轮不据此宣称新的视觉高敏感性机制。</p>')
    parts.append('<h2>任务分解与历史结果的关系</h2>')
    rows=[]
    for t in range(10):
        row=[t]
        for target in ['pi0','pi05']:
            for c in ['plain','refcloth']:
                chosen=[e for e in episodes if e['target']==target and int(e['task'])==t and e['condition']==c]
                row.append('%d / 3'%sum(int(e['robot_success']) for e in chosen))
        rows.append(row)
    parts.append(table(['Object 任务','π0 Plain','π0 RefCloth','π0.5 Plain','π0.5 RefCloth'],rows))
    rows=[]
    for target in ['pi0','pi05']:
        for method,c,title in [('F0','P','Plain'),('MR','A','RefCloth')]:
            z=next(x for x in archive if x['target']==target and x['method']==method and x['condition']==c)
            current=a[target]['plain_success' if c=='P' else 'ref_success']
            rows.append([target,title,'%d / %d'%(z['full_success'],z['full_n']),
                '%d / %d'%(z['subset_success'],z['subset_n']),'%d / 30'%current])
    parts.append(table(['策略','条件','旧正式主表','旧记录同 30 个初态','本次成对噪声复测'],rows))
    parts.append('<p>旧主表每条件 100 回合，本轮每条件 30 回合；本轮显式对齐每次查询噪声，旧评测随机性没有这一约束。新结果用于检验相同原纹理在正确条件下是否仍有净攻击效果，不当作旧 100 回合数据的逐回合复现或替代。</p>')
    parts.append('<h2>预先指定的双相机视频</h2><p>固定展示两个目标的 task 0 与 4、init 10、Plain 与 RefCloth；没有按成败挑选。每帧对应一次策略查询，左右为主相机与腕部相机，4 fps 对应每 5 动作查询一次。</p><div class="videos">')
    for target in ['pi0','pi05']:
        for t in [0,4]:
            for c in ['plain','refcloth']:
                key='%s_t%d_i10_%s'%(target,t,c)
                path=ROOT/'raw'/('pi0_main' if target=='pi0' else 'pi05_main')/'videos'/(key+'.mp4')
                assert path.exists(),path
                parts.append('<div><p>'+key+'</p><video controls preload="metadata" src="'+path.relative_to(ROOT).as_posix()+'"></video></div>')
    parts.append('</div>')
    parts.append('<p>固定案例中的定性观察：π0.5 的 task 0 指令要求取 alphabet soup，RefCloth 回合却将绿色瓶子放入篮子；task 4 要求取 ketchup，RefCloth 回合却将牛奶盒放入篮子。对应 Plain 回合完成了目标任务。π0 的 task 0 两条件都成功，task 4 两条件都失败，后者不计入攻击诱发失败。这两段 π0.5 视频提示存在目标物体选择混淆，但没有对全部失败回合进行类型标注，不能据此报告整体混淆比例或断言内部视觉机制。</p>')
    parts.append('<img src="figures/pi0_video_contact_sheet.png" alt="π0 预先指定视频帧"><img src="figures/pi05_video_contact_sheet.png" alt="π0.5 预先指定视频帧">')
    parts.append('<h2>论文应如何处理</h2>')
    parts.append('<p>应修正图一 Object 暴露点、补充材料对应表格和“小可见范围仍有显著攻击”的 Object 解释。应把“几何可见范围”“Plain 与背景的对比差分”“真正注入纹理导致的输入差分”分开命名。若采用本轮三种测量作新图，清楚写明是 60 条 Object 完整轨迹、实际模型输入，不能沿用旧 240 条全套件统计的图注。</p>')
    parts.append('<p>本轮直接检验的是原任务中的静态场景纹理攻击：固定原纹理在正式铺布下的净效果。它没有增加新算法，也没有把任务改成新的攻击设定。主论文原文件已核对哈希，未被本轮改写。</p>')
    parts.append('<p class="caveat">局限：每目标 30 个初态，仅每任务 3 个；支持范围先按回合聚合；物理世界攻击效果没有在本轮验证；标记是几何代理；没有新增 Original 条件回合；不能把没有显著差异解释成效果等价。</p>')
    parts.append('<h2>可复查文件</h2>')
    links=[('summary.json','完整数值摘要'),('episode_outcomes.csv','120 回合成败记录'),('plain_episode_measurements.csv','按回合聚合的全部测量'),
           ('plain_query_measurements.csv','逐查询三种测量与阈值敏感性'),('same_state_first_query_checks.csv','60 组同状态检查与动作差异'),
           ('code/protocol.json','预先固定的测量与评测协议'),('final_integrity_checks.json','最终完整性检查'),('artifact_manifest.json','文件校验清单')]
    parts.append('<ul>'+''.join('<li><a href="'+p+'">'+title+'</a></li>' for p,title in links)+'</ul>')
    parts.append('<p>代码分支：<a href="https://github.com/ShaoZhenLiu/tex3d-repro/tree/object-investigation-20261102">object-investigation-20261102</a>。代码在修改前已与 GitHub 同步；运行使用 4090 的 GPU0、3、6，未使用 GPU4。</p>')
    style='''body{font-family:"Microsoft YaHei",Arial,sans-serif;max-width:1160px;margin:32px auto;padding:0 22px;color:#203040;line-height:1.75;background:#fff}h1{font-size:29px}h2{margin-top:36px;font-size:21px;border-bottom:1px solid #d5dce2;padding-bottom:8px}p{margin:12px 0}.meta{color:#657585;font-size:14px}.finding{background:#edf3f7;border-left:5px solid #526d82;padding:18px}table{border-collapse:collapse;width:100%;margin:20px 0;font-size:13px}th,td{text-align:left;border-bottom:1px solid #d9e0e5;padding:9px 7px}th{background:#edf2f6}img{max-width:100%;display:block;margin:22px auto}.caveat{background:#fbf5e9;padding:16px}.videos{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}.videos video{width:100%}a{color:#165f91}@media(max-width:700px){.videos{grid-template-columns:1fr}table{font-size:11px}body{padding:0 12px}}'''
    page='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Object 完整测试报告</title><style>'+style+'</style></head><body>'+''.join(parts)+'</body></html>'
    (ROOT/'Object完整测试报告.html').write_text(page,encoding='utf-8')
    print('REPORT_WRITTEN',len(page))


if __name__=='__main__':main()
