# Tex3D 复现环境说明（4090-48G，2026-09-24）

## 位置
- 代码: /data/shaozhen.liu/code/tex3d （git 仓库, upstream vla-attack/tex3d）
- 依赖源码: /data/shaozhen.liu/code/tex3d-deps/{libero, libero-pi, openvla, openvla-oft, nvdiffrast, nvdiffrast-v040, taming-transformers}
- venv: /data/shaozhen.liu/envs/{tex3d-openvla, tex3d-oft, tex3d-pi0, tex3d-tools}
- 模型: /data/shaozhen.liu/model/（RLinf-Pi0-SFT 下载中; openvla 分套件待下载）
- libero assets: tex3d-deps/libero/libero/libero/assets（完整）
- tex3d/libero-eval -> tex3d-deps/libero 软链

## 环境对照（Docker → 主机 venv）
| Docker 镜像 | venv | 关键版本 |
|---|---|---|
| tex3d-openvla (py3.10.5) | envs/tex3d-openvla | torch2.2.0+cu121, transformers4.38.2, mujoco3.4.0 |
| tex3d-oft (py3.10.5) | envs/tex3d-oft | torch2.2.0+cu121, transformers4.40.1 |
| tex3d-pi0 (py3.11.14) | envs/tex3d-pi0 | torch2.6.0+cu124, transformers4.53.2(已覆盖 openpi 补丁), mujoco3.8.1, robosuite1.4.1, nvdiffrast v0.4.0 |

## 主机与 Docker 的差异（重要）
1. 渲染: 无 OSMesa → 必须 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl，选卡用
   CUDA_VISIBLE_DEVICES=N MUJOCO_EGL_DEVICE_ID=N（两个都要,只设 CUDA 会 EGL 枚举失败）
2. LIBERO weights_only 补丁已 sed; bddl 黑碗补丁文件仓库缺失未打（attack 目标描述可能差一句语言指令）
3. libero editable 安装在 uv venv 下 PEP-660 映射失效 → 已用 tex3d_libero_path.pth 修复
4. protobuf 升到 7.x 修复 runtime_version; π0 venv torch 被 requirements 拉高后已回退 2.6.0+cu124 并重编 nvdiffrast

## VQGAN（π0 对抗训练用）
ommer-lab 原链接 404 → 用官方 vq-f16.zip (792MB, ommer-lab.com/files/latent-diffusion/vq-f16.zip)
- ckpt: taming-transformers/checkpoints/vqgan_imagenet_f16_16384.ckpt (从 zip 提取 model.ckpt)
- config: taming-transformers/configs/vqgan_imagenet_f16_16384.yaml (来自 CompVis/latent-diffusion vq-f16.yaml)
- 代码用 VQModel load_state_dict(strict=False), 只用 encoder, 架构兼容已确认

## 运行示例（cd /data/shaozhen.liu/code/tex3d）
OFT clean eval:
  CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl   PYTHONPATH=/data/shaozhen.liu/code/tex3d:/data/shaozhen.liu/code/tex3d/openvla/experiments/robot   /data/shaozhen.liu/envs/tex3d-oft/bin/python openvla-oft/experiments/robot/libero/attack_oft.py     --pretrained_checkpoint /data/shaozhen.liu/model/openvla-7b-oft-finetuned-libero-spatial     --task_suite_name libero_spatial --task_id 0 --enable_attack False
π0:
  ATTACK_GPU_ID=1 pi/run_attack_pi.sh --run_mode clean_test --object_name alphabet_soup --eval_task_id 0 --eval_num_trials 2 --replan_steps 5

## 代理
ssh -R 7901:127.0.0.1:7890（7899/7900 被死会话占用）；脚本用 http://127.0.0.1:7901

## 更新（2026-09-24 下午）
- 原版 OpenVLA 分套件权重来自 ModelScope（国内直连,服务器不用代理即可下）:
  - spatial: superpeach/openvla-7b-finetuned-libero-spatial
  - object:  superpeach/openvla-7b-finetuned-libero-object （仓库名 typo finetuend 是对的）
  - 已把 auto_map 改为本地限定 + 补齐 configuration/modeling/processing_prismatic.py → 完全离线可加载（HF_HUB_OFFLINE=1）
  - 验证: AutoModelForVision2Seq 加载 7.54B params bf16 GPU OK
- HF 下载注意: 禁 Xet 后端（HF_HUB_DISABLE_XET=1）, 否则经代理会 CAS 解码失败
- RLinf-Pi0 与 VQGAN(vq-f16.zip) 走 7901 隧道下载中

## VQGAN 验证记录（2026-09-24 下午）
- vq-f16.zip 提取的 model.ckpt (902MB) -> checkpoints/vqgan_imagenet_f16_16384.ckpt
- config 来自 latent-diffusion models/first_stage_models/vq-f16/config.yaml（不是 configs/ 下!）
- taming-transformers 需两补丁: torch._six (data/utils.py) + torch.load weights_only=False
- attack_pi0.py:855 已打 weights_only=False 补丁
- 验证: 392 keys 全载入, encoder 前向 (1,8,16,16) OK

## 全部就绪（2026-09-24 12:50）
- RLinf-Pi0 SFT 下载完成(6.5GB): pi0 policy 经 openpi create_trained_policy 加载验证 3.50B GPU OK
  - 注意: ckpt 目录需 assets/physical-intelligence/libero/norm_stats.json 布局(已补)
  - π0 venv 需 pytest(openpi gemma_pytorch 模块级 import)
- 三条攻击线全部可跑:
  OpenVLA: tex3d-openvla venv + model/openvla-7b-finetuned-libero-{spatial,object}
  OFT:     tex3d-oft venv + model/openvla-7b-oft-finetuned-libero-*(服务器原有全套)
  pi0:     tex3d-pi0 venv + model/RLinf-Pi0-...-SFT + taming VQGAN(已验证)
- 隧道已关闭; 后续装小包用 tuna 镜像, 大模型用 ModelScope 直连

## OFT clean baseline 实测（2026-09-24 21:40）
- 命令模板见 experiments/logs/oft_clean_spatial_t0_n50.log 同名启动方式
- 结果: libero_spatial task0 (black bowl) 50/50 success, 符合论文 clean 水平
- 速度: 18.3 s/episode (50ep 任务约 15.5min + 模型加载 1.5min)
- 资源: 16.0GB 显存/进程, GPU 利用率 46% -> 单卡可并行 2-3 个 eval 进程
- 启动过程中修复: protobuf 全家桶需对齐 6.31.1(wandb 0.28.0 + tf-metadata 新版 + bddl 3.6.0),
  LIBERO 隐藏依赖需手装(robosuite 1.4.1 / bddl 3.6.0 / easydict / cloudpickle / termcolor),
  attack 脚本 MUJOCO_GL=osmesa 硬编码已改为尊重环境变量,
  OFT venv 中 prismatic 需指向 openvla-oft fork(已改 openvla finder),
  attack_oft.py 需要 ./LIBERO-master 软链(已建)

## Clean baseline 全量结果（2026-09-25）
| 线 | 套件 | 失败率 | 论文 clean |
|---|---|---|---|
| OpenVLA | spatial | 20.8% | 24.1% 吻合 |
| OpenVLA | object | 36.6% | - |
| OFT | spatial | 9.6% | 4.7% |
| OFT | object | 18.0% | - |
| pi0 | object | 29.4% (353/500) | 4.6% |
结果文件: experiments/logs/job_*.log + experiments/pi0_attacks/clean_test_*/evaluation_result.txt

## 攻击试点关键教训
- attack_oft.py 的 enable_attack 默认是 False（README 写错为 True），必须显式 --enable_attack True
- pi/run_attack_pi.sh 引用的 attack_pi.py 实际文件名是 attack_pi0.py（已软链）
- 攻击训练实测: 4.4s/iter（4090-48G，论文 7.2s），5000iter 约 6.1h；显存 35.6GB，92% 利用率 → 一卡一进程
- pkill -f 含目标字符串的命令行会自匹配（server.md 教训再现），批量启动用独立脚本文件
- GPU 占用纪律: 0/3/4 有他人任务，我们用 2/5/6/7
