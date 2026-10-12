# tex3d-repro 同步手册（41 / 09 / Windows 三端）

> 仓库：github.com/ShaoZhenLiu/tex3d-repro（**私有**，未发表研究内容，勿公开）
> 拓扑：GitHub 为中心；09 直连 GitHub（枢纽）；41 不可直连 GitHub，经 09 中转。

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
