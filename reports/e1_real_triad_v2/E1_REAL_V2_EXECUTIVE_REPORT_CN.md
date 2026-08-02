# E1 REAL TRIAD v2 总报告

## 结论摘要
- 最终决策：`E1_REAL_V2_STOP_CONTEXT_CAPACITY`
- A Gate：`PASS_REUSED_FROZEN_A`；B Gate：`STOP_CONTEXT_CAPACITY`；C Gate：`NOT_RUN_BECAUSE_B_NOT_FORMAL`；Gold Gate：`NOT_RUN_FORMAL_GOLD_V5_REQUIRED`
- B 真实目标回答比例：`1.0000`
- API 费用：`0.0000` 元

## 核心分析
本轮执行采用 `E1-REAL-TRIAD-v2.0` 协议，正式候选池仅保留 Qwen/DeepSeek 的真实目标模型回答；审计显示 source_derived 行数为 0，真实目标回答比例为 1.0000。

B 层容量预筛的四个 stratum 计数为：stable positive=26，stable negative=3891，critical positive=33，hard negative=6。与正式需求 420/420/180/180 相比，正例容量仍是主要瓶颈。

由于当前容量审计尚未形成 formal Gold v5 双评审/裁决面板，本轮没有继续训练 q-only、y-only、q+y 和 wrong-q，也没有进入 C 层迁移。这样处理会牺牲“好看”的指标，但避免再次出现模板面板、确定性标签或模式泄漏导致的不可用结果。

最终决策为 `E1_REAL_V2_STOP_CONTEXT_CAPACITY`；该结果可以作为后续补采样和 API 投入的准入依据，但不能作为论文主表中的 STRONG PASS 结果。

## A 层自然发生率
|口径|正例数|样本数|发生率|Wilson_low|Wilson_high|
|---|---|---|---|---|---|
|lower|12|3080|0.0039|0.0022|0.0068|
|central|26|3080|0.0084|0.0058|0.0123|
|upper|66|3080|0.0214|0.0169|0.0272|

## B 层容量与主表
|stratum|现有候选数|正式需求|缺口|满足正式门槛|
|---|---|---|---|---|
|context_stable_positive|26|420|394|False|
|context_stable_negative|3891|420|0|True|
|context_critical_positive|33|180|147|False|
|context_hard_negative|6|180|174|False|

## C 层迁移
|metric|value|reason|
|---|---|---|
|status|未运行|B 层未形成 formal Gold v5 面板，按协议不得迁移 C。|
