先打开“Object完整测试报告.html”。报告中的所有面积均明确区分：
1. 30 个相同动态初态的位置对照；
2. 两目标各 30 条 Plain 完整轨迹；
3. 原生 256×256 图像与真实 224×224 模型输入；
4. 双色标记、Plain-hidden 对比差分、RefCloth-Plain 实际变化。

summary.json 是最终数值。episode_outcomes.csv 是 120 回合成败。
plain_episode_measurements.csv 是按回合聚合结果；plain_query_measurements.csv 是逐查询记录。
same_state_first_query_checks.csv 记录 60 组条件配对的输入、状态校验及动作差异。
raw/ 保存服务器原始 JSON、首帧五条件图像、预定视频和诊断日志。
evidence/ 保留原论文数据、环境版本与中间检查证据；其中 monitor 文件是进度快照，不用于最终结论。
figures/ 中的 PDF 可独立用于论文排版；PNG 可直接查看。

本轮只测试固定原纹理，不重新训练；所有目标推理均不使用梯度。
两模型共享 30 个任务—初态键，分别闭环运行。不能把两个模型合并成 60 个独立物理初态。
运行使用 4090 服务器 GPU0、3、6。09 同步代码；本轮未使用其被其他任务占用的 GPU。
GitHub 分支：ShaoZhenLiu/tex3d-repro / object-investigation-20261102。
修改前，两台服务器均已 fetch/ff-only 同步到 fig1-measurement-20261101 的 856264fa。
原论文文件保留，final_integrity_checks.json 记录了其哈希核对结果。

复算最终统计（本机现有 Python 环境）：
python code/build_results.py --root E:/Linux/cvpr2027/RefCloth_Object_20261102
python code/final_checks.py
python code/write_report.py

服务器实验入口、资源环境和显式噪声协议见 code/controller.py、code/run_object.py、code/protocol.json。
controller.py 先等待 pi05 冒烟通过，再运行 pi0 冒烟及正式任务；首次 pi05 冒烟需按 run_object.py --mode smoke 启动。
请勿直接在原结果目录改协议后混合续跑。新实验应使用新的输出目录。
最终压缩包旁的 archive_receipt.json 提供 ZIP 的 SHA256；artifact_manifest.json 提供各文件校验。
