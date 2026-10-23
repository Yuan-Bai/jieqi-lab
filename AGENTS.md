# Jieqi Lab agent guidance

- 默认使用简体中文；先说明结果，再说明必要细节。
- 本目录是 Jieqi Lab 的公开源码工作区。以本目录为工作根目录；不要自动复制其他工作区的
  Git 历史、模型、训练数据或平台适配。
- 做最小、可验证的修改；不自动提交、推送、重写 Git 历史、发布或删除用户数据。
- 源码以 Python 规则引擎为准。保留规则版本、隐藏信息隔离、训练检查点及来源链的语义。
- `jieqi.py` / `ismcts.py`：规则和搜索；`azero.py`：训练；`search_engine.py`：通用引擎适配；
  `web_server.py` / `web/`：本地网页；`gui.py`：Tkinter 界面。
- Python 测试：`python -m pytest -q`。前端检查：`node --check web/app.js`；
  `node --test web/preferences.test.cjs web/history.test.cjs web/view-state.test.cjs`。
- 先运行最相关测试；可选原生扩展缺失导致的跳过不算通过。Windows 沙箱的硬链接/原子写入
  失败需与真实回归区分，不要为此削弱保存机制。
- 不自动启动长时间训练、模型评估或性能采样；普通单元测试可以运行。
- `data/`、模型、训练数据、缓存和编译产物不进入源码提交。
- 发布与验证流程见 `docs/public-release-preparation.md`；安全报告按 `SECURITY.md` 处理。
