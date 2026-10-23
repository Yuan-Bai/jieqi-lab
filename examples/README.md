# 合成棋谱

`synthetic-branches.json` 是当前规则下的人工构造示例，可用网页的“导入”或 GUI 加载。
它展示红方翻子、黑方吃掉暗子和第二手的替代分支，不是完整实战、训练样本或棋力评估。
初始页面定位于主线第二手；回到第一手可选择另一分支。

该棋谱允许展示全部分支的私有揭示，来源只有项目规则与生成器，没有真实玩家、平台内容、
模型、身份字段或本机路径。不要将它误认为任意真实棋谱脱敏后的样例。

```powershell
# 默认打印 JSON；写文件时要求目标不存在，避免覆盖已有棋谱。
python tools/generate_example_record.py --out synthetic-copy.json
python -m pytest -q tests/test_public_examples.py
```

修改示例应先修改 [生成器](../tools/generate_example_record.py)，再生成新文件与候选比较。
源码内的 JSON 由测试校验与生成器一致，并通过实际 Web 导入验证隐私过滤。
