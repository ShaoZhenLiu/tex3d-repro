# tex3d-repro 同步手册（41 / 09 / Windows 三端）

> 仓库：github.com/ShaoZhenLiu/tex3d-repro。2026-10-31 经 GitHub 实测为公开仓库，用户明确确认“保持公开，允许继续同步”。旧技能中的私有约定已被本次用户授权覆盖。
> 拓扑：GitHub 为中心；2026-10-31 已实测 41 与 09 均可直连 fetch/push。下面的 09 中转命令保留作网络不通时的备用路径。

## 分支与 remote 约定
- 工作分支 `lab`（推送为远端 `main`）；上游原始代码在 `origin/main`，勿混。
- 09 的 remote：`origin`=上游、`lab`=GitHub 私有仓
- 41 的 remote：`origin`=上游、`lab09`=ssh://shaozhen.liu@172.18.33.9/data/shaozhen.liu/code/tex3d（09 中转）

## 09（直连 GitHub）
```bash
cd /data/shaozhen.liu/code/tex3d
git add -A && git commit -m "..."
git push lab lab:main      # 推
git pull lab main          # 拉
```

## 41（经 09 中转）
```bash
cd /data/shaozhen.liu/code/tex3d
# 推：先推到 09 的中转分支，再在 09 上合入并推 GitHub
git push lab09 lab:lab-inbox
#   然后 09 上执行：
#   git fetch lab09 +lab-inbox:lab-inbox && git merge lab-inbox && git push lab lab:main
# 拉：
git fetch lab09 && git merge lab09/main   # 09 先 git pull lab main
```

## Windows 本机
```bash
gh repo clone ShaoZhenLiu/tex3d-repro   # gh 已登录 ShaoZhenLiu
```

## 注意
- 不要把 `git push --force` 用在本仓库；历史即实验记录。
- 模型/噪声 .pt/视频不入 git（.gitignore 已排除），大件仍走 rsync。
- 提交前 `git status` 检查，勿提交 experiments/ 产物。

## 2026-10-31 创新探索分支

修改前两台服务器均对齐 GitHub residual-explore 的 e94de0b。新工作位于 innovation-20261031，保持主分支和论文历史结果不变。两端同步使用 `git fetch lab` 与 `git merge --ff-only lab/innovation-20261031`。不 force-push；实验原始 .pt/模型/视频仍不入库。4090 GPU4 禁用，用户允许在显存足够时共享 GPU0 与其他空闲卡。
