# 私有工作区

除本说明外，`drafts` 默认被 Git 忽略，不进入公开站点。研究提纲、项目任务、写作笔记和文章草稿在这里保留；文章完成核对后再移入 `docs/articles`，设置 `status: published` 并声明真实 `prerequisites`。

本仓库在 GitHub 上是公开的；不要通过强制添加文件把私有草稿提交到仓库，也不要把求职记录、单位内部资料、凭据或外部备份放在这里。

## 全年计划

本地 `year-plan.json` 是私有全年计划，结构为：

```json
{
  "startDate": "YYYY-MM-DD",
  "timeZone": "Asia/Shanghai",
  "weeks": [
    {
      "topic": "本周主题",
      "days": [
        { "tasks": [{ "title": "当天任务", "minutes": 60 }] }
      ]
    }
  ]
}
```

`weeks` 按连续七天组织。构建只选择当天，不向前端输出其他日期的任务。若需线上显示，在 GitHub 仓库 Secret 中把完整 JSON 配置为 `STUDY_PLAN_JSON`，由每日工作流重新构建；不要提交该 JSON。未配置时文章页不显示 Todo，不影响其他页面。
