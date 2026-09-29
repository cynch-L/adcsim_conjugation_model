# Trastuzumab Cys-ADC 偶联模型优化计划 v3

> 基于完整批判的路线图，面向 demo 但尽量严谨。每个模块标注优先级（P0=必须/P1=重要/P2=可选）、改动量、预期收益。
> 偏离方向时以本表校准，不堆砌不硬套。

---

## 改动总览

| 模块 | 改动 | 优先级 | 改动量 | 收益 |
|---|---|---|---|---|
| ① 二硫键解析 | 加 strain 分类 + 还原难易度排序 | P1 | 小 | 中 |
| ② MD 模拟 | v4 已修复（还原态+CA锁+5ns），不再改 | ✅ 已收口 | — | — |
| ③ SG SASA | 加 open fraction/persistence，输入已修正 | P1 | 小 | 中 |
| ④ SG 位移 | 加有效位移占比（径向投影 vs 蛋白中心） | P2 | 小 | 低 |
| ⑤ pKa | 还原态 PROPKA 重跑 | P1 | 中 | 中-高 |
| ⑥ 偶联指数 | 权重重构（+立体位阻 +还原难度） | P1 | 中 | 高 |
| ⑦ DAR 分布 | 加还原步骤 + 位点消耗 | P0 | 中 | 高 |
| ⑧ 旁观者效应 | PDE 一维模型（Roadmap 阶段二） | P2 | 大 | 低（demo） |
| ⑨ 验证/校准 | 物理合理性检查 + 文献对标 | P1 | 小 | 高 |
| ⑩ 文档/配置化 | 面试报告 + 参数 YAML | P0 | 小 | 高 |

---

## 详细执行

### ① 二硫键解析 — P1 小改动

**为什么要改**：当前二硫键解析只输出 SG 键长 + Cα-Cα 距离，没有分类和排序。strain 信息直接影响还原步骤的速率差异。

**具体做法**：
```
每对链间 SS：
  - type: hinge_HC-HC / HC-LC_interface
  - strain_score = Cα-Cα距离 / 引用值(5.5Å) — 1.0
  - reducibility_class: easy(>0.1) / normal(-0.1~0.1) / hard(< -0.1)
  - burial_depth = SG 到蛋白表面最小距离（Å）

输出：ss_reduction_profile.json（4 对链间 SS 的还原难度排序）
```

**不改的**：二面角 strain、局部氢键网络——区分度太低，demo 不划算。

**产出文件**：`ss_reduction_profile.json`

---

### ② MD 模拟 — ✅ 已收口

**v4 已修复全部致命问题**：
- ✅ 还原态（`mod.delete([target_bond])` + CYS 命名）
- ✅ CA 约束 Fab+Fc（k=5 kcal/mol/Å²）
- ✅ prod 5ns（解决了 1.5ns 的收敛问题）
- ✅ elem 字段保存（离线 SASA 不再依赖外部 PDB）

**不改的**：
- 保持隐式溶剂 OBC2（QC 证排序正确，显式水是验证不是替代）
- 保持 4 位点（不再跑全 8 个）
- 不加 metadynamics（过度工程）

**验证点**：v4 跑完后，检查 4 位点的 split-half 收敛性（前后段 SASA 偏差 <15% 才算通过）。

---

### ③ SG 轨迹 SASA — P1 小改动

**为什么要改**：当前只有 mean/std/min/max，缺少时间维度的统计量。

**新增指标**（基于 v4 npz 后处理）：
```
persistence_score: SASA > 15Å² 的最长连续帧数 / 总帧数
burst_magnitude: 瞬时最大 SASA 值（>50Å² 的事件计数）
effective_exposure: mean × open_fraction（消除"脉冲式开关"噪音）
```

**为什么不改**：v4 的输入已经是还原态轨迹，方法正确（Shrake-Rupley 720 点），不需要重构。

**产出文件**：`md_sg_sasa_v4.json`（替换 all8.json）

---

### ④ SG 位移 — P2 小改动

**为什么要改**：当前位移是等方向幅度，"摆向溶剂"和"摆进蛋白"等价。

**新增指标**：
```
effective_displacement_ratio = 
   径向方向投影的正位移 / 总位移幅度（0~1）
   0 = 完全摆向蛋白内部（无效）
   1 = 完全摆向溶剂（有效）
```

**为什么不优先做**：SG 位移在 v2 偶联指数里只占 15% 权重，0→1 的变化拉不开排名。

---

### ⑤ pKa — P1 中等改动

**为什么要改**：当前用文献值 8.5 统一兜底，区分度接近零。

**具体做法**：
```
1. 用 OpenMM mod.delete([target_bond]) + mod.addHydrogens(pH=7.0)
   生成 8 个 Cys 全还原态的 PDB（一次跑，不是每个位点独立）

2. 喂给 PROPKA3，读每个 Cys 的预测 pKa

3. 预期：埋藏位点（C/D-Cys214）pKa > 9.0，暴露位点（A/B-Cys223）pKa ~7.5-8.0
   → 区分度从 0.1 提升到 1.5-2.0

4. 如果 PROPKA 还是全 99.99%（识别为成键态）→ 用几何距离法：
   - 每个 SG 周围 5Å 内带电残基数 × 0.3 修正
   - 埋藏位点 +1.0，暴露位点 -0.5
```

**为什么不跑全套**：
- MD 平均 pKa（每 100 帧一次 PROPKA）太贵，demo 不需要
- 几何修正法已有 QC 数据（类别 3 的 5Å 环境）

**产出文件**：`reduced_pka.json`（8 位点还原态 pKa）

---

### ⑥ 偶联指数 v3 — P1 中等改动

**为什么要改**：v2 的权重是主观的，且漏掉了立体位阻和还原难度。

**v3 公式**：
```
index = 0.25 × SASA_norm
      + 0.20 × open_fraction_norm
      + 0.20 × f_thiolate_norm (基于还原态 pKa)
      + 0.15 × steric_score (新增)
      + 0.10 × reduction_ease (新增，从 ① 继承)
      + 0.10 × effective_displacement_norm (从 ④ 继承，如果做了)
```

**新增特征计算**：
```
steric_score:
  在 SG 周围 8Å 球面上生成 200 个采样点
  检测这些点与蛋白重原子的碰撞
  无碰撞比例 = steric_score（0~1）
  → 模拟 payload(半径8Å)能否无碰撞靠近 SG

reduction_ease:
  从 ss_reduction_profile.json 读取
  strain_score > 0.1 → reduction_ease = 0.8
  strain_score -0.1~0.1 → reduction_ease = 0.5
  strain_score < -0.1 → reduction_ease = 0.3
```

**不做实验校准**（没有实验数据）——在文档里标明"权重为物理推断，未经实验拟合"。

**产出文件**：`site_conjugation_index_v3.json`

---

### ⑦ DAR 分布 v3 — P0 中等改动

**为什么要改**：当前 v2 直接假设 viable 位点 100% 可用（跳过还原），物理不完整。

**核心改动**：一步 → 两步

```
步骤1：还原（TCEP, 1h）
  每对链间 SS 独立：
    p_reduce = 1 - exp(-k_red × [TCEP] × 60)
    k_red = 0.5 × (1 + strain_score)  # M⁻¹·min⁻¹
    默认 [TCEP] = 5 mM, 还原 1h
    
  结果：确定哪些位点变成游离 SG

步骤2：偶联（pH 7.4, 2h）
  可用位点集 = 步骤1 中被还原的位点 ∩ viable 位点 (SASA>15)
  每站偶联概率：
    k_conj = 10.0 × f_thiolate × (SASA_norm)  
    p_conj = 1 - exp(-k_conj × [payload]_eff × 120)
    [payload]_eff 随已偶联位点数递减（竞争消耗）
    payload:Ab 初始投料比 = 6:1

枚举：4 对二硫键 → 2^4 = 16 种还原组合 × 每组合枚举偶联
```

**参数合理性检查**：
- k_red = 0.5 M⁻¹·min⁻¹ → 暴露 SS 在 5mM TCEP 下 1h 还原 ~78%（合理）
- k_conj = 10 M⁻¹·min⁻¹ → 暴露 SG 在 6× 投料 2h 下偶联 ~85%（合理）
- 两个参数都可以在文档里引用文献支撑（不拟合）

**预期结果**：
- mean DAR 从 v2 的 2.9 降到 ~2.0-2.5（还原步骤过滤掉了一些）
- DAR0 从 1.2% 升到 ~5%（部分二硫键不被还原）

**产出文件**：`dar_v3_twostep.json`

---

### ⑧ 旁观者效应 — P2 大改动

**Roadmap 阶段二，demo 暂不优先做。** 理由是：
- DAR 均一性是面试展示的主线，旁观者效应是加分项
- PDE 模型需要的载荷参数（logP、膜通透性）要额外查文献
- 计算量不大但信息收集耗时间

**如果主线做完有余力**：按 Roadmap 阶段二的代码骨架跑一遍 MMAE vs MMAF vs DM1 的旁观者排序。

---

### ⑨ 验证 / 校准 — P1 小改动

**物理合理性检查**（全部自动化）：
```
✓ 所有 SASA ≥ 0
✓ 所有 pKa 在 0-14 范围
✓ mean DAR < 8（最多 8 个链间 Cys）
✓ sum(site_occupancy) ≈ mean DAR
✓ 铰链位点排名 > HC-LC 界面位点排名
✓ 还原步骤中，strain 越大的 SS 对还原概率越高
```

**文献对标**（不拟合，只对照）：
```
T-DM1 (Kadcyla): 实验 mean DAR ≈ 3.5
我们的预测：mean DAR ≈ 2.0-2.5 (偏低的原因：隐式溶剂 SASA 偏低)
偏差来源：记入文档
```

**不做的事**：
- 不拟合权重（没有实验数据）
- 不盲测（需要新抗体 PDB + 新 MD）

---

### ⑩ 文档 / 配置化 — P0 小改动

**面试报告更新**：
- ADC_Final_Report.md → ADC_Final_Report_v3.md
- 加入 v3 改动说明 + 模型局限性（专题小节）
- 结果对比表（v2 vs v3，解释为什么 v3 更可信）

**参数 YAML**（避免后续硬编码）：
```yaml
# herceptin_config.yaml
antibody:
  name: Trastuzumab
  pdb: B220_235_disulfide_repaired.pdb
  chains: [A, B, C, D]

interchain_ss:
  - [A223, C214, HC-LC_interface]
  - [B223, D214, HC-LC_interface]
  - [A229, B229, hinge]
  - [A232, B232, hinge]

reaction:
  reducer: TCEP
  tcep_conc_mM: 5.0
  reduction_time_h: 1.0
  payload_ratio: 6
  conjugation_time_h: 2.0
  pH: 7.4
  temperature_K: 300

md:
  solvent: implicit_OBC2  # validated: rank correct
  equilibration_ns: 0.5
  production_ns: 5.0
  ca_restraint: [1-210, 250+]
```

---

## 执行顺序

| 序号 | 模块 | 优先级 | 预计耗时 | 依赖 |
|---|---|---|---|---|
| 1 | ⑩ 文档/配置化 | P0 | 30 min | 无 |
| 2 | ⑤ pKa 还原态重算 | P1 | 30 min | 无 |
| 3 | ① SS reduction profile | P1 | 15 min | 已有 QC 数据 |
| 4 | ⑥ v3 偶联指数 | P1 | 30 min | ①⑤ |
| 5 | ⑦ v3 DAR 分布 | P0 | 30 min | ①⑥ |
| 6 | ③ SG SASA 新指标 | P1 | 15 min | v4 npz 到位后 |
| 7 | ⑨ 验证检查 | P1 | 15 min | ⑥⑦ |
| 8 | ④ SG 有效位移 | P2 | 15 min | v4 npz |
| 9 | ⑧ 旁观者效应 | P2 | 2-3h | （后续） |

---

## 偏离校准清单

当想加东西时，先问自己：

1. **是否在主线（位点筛选 → 偶联指数 → DAR 分布）上？** 不在 → 砍
2. **已有数据够不够支撑？** 不够 → 先用文献值兜底，不在 demo 阶段补实验
3. **区分度够不够？** 不够 → 即使理论上正确也砍（如 pKa MD 平均）
4. **面试能讲清楚吗？** 不能 → 简化或用 v2 版本
5. **是否过度工程？** 比如 metadynamics、盲测、DOE → 砍

**本计划里的 P1 全部做，P2 看余力。偏离时先降优先级不砍模块。**