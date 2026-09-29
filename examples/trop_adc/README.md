# adcsim — Trastuzumab Cys-ADC 偶联建模管线

用计算方法预筛抗体偶联位点、预测 DAR 分布，减少湿实验穷举试错。

---

## 1. 逻辑树（从结构到结论）

```
输入：Trastuzumab 结构 B220_235（4 条链，8 个链间半胱氨酸）
  │
  ├─ [QC] 结构零信任校验（26 项：二硫键几何、Cα–Cα、二面角、总 SASA）
  │
  ├─ [位点可及性] MD（还原态，4 位点 × 5 ns，隐式溶剂，CA 约束）
  │     └─> 动态 SG SASA、open fraction、前后半段收敛性
  │
  ├─ [化学活性] PROPKA（部分还原态：只断 4 对链间 SS，SG 拉开到 4 Å）
  │     └─> 各位点 pKa -> Henderson-Hasselbalch -> f(硫醇负离子)
  │
  ├─ [还原选择性] 二硫键可还原性 profile
  │     └─> 成键态静态可及性 + 几何应变 + 文献先验（IgG1：铰链 > HL）
  │
  ├─ [综合] 偶联指数 v3
  │     0.30·SASA + 0.20·open + 0.18·还原易度 + 0.22·f(SH⁻) + 0.10·位移
  │
  └─ [DAR] 两步模型 v4
        Step 1  还原：p_red(pair) = 1 − exp(−Λ·ease·TCEP)      ← 整对事件
        Step 2  偶联：p_c = 1 − exp(−k₂·[P]·t)，封顶到工艺收率  ← 共价动力学
        └─> DAR 分布、even/odd、工艺窗口
```

---

## 2. 一键复现

```bash
cd examples/trop_adc/kaggle_md_8site
python run_all.py            # 全部（含 MD 后处理，若 npz 已下载）
python run_all.py --no-md    # 只跑本地部分
```

依赖：`numpy scipy matplotlib openmm` + `propka`（pKa 步骤需要）

MD 本身在 GPU 服务器上跑，不在本地管线内：

```bash
scp -P <port> run_md_v4.py B220_235_disulfide_repaired.pdb root@<host>:/root/autodl-tmp/
cd /root/autodl-tmp && nohup <python> -u run_md_v4.py > md_v4.log 2>&1 &
```

---

## 3. 关键文件

| 文件 | 作用 |
|---|---|
| `run_md_v4.py` | MD：还原态 4 位点 × 5 ns，断 SS + 加氢后重找 SG + CA 约束 |
| `process_md_v4.py` | MD 后处理：动态 SASA、收敛性、C214 裁决（**已就绪，等 npz**） |
| `ss_reduction_profile.py` | 4 对链间二硫键可还原性 |
| `make_interchain_reduced.py` | 生成部分还原态 PDB（PROPKA 用） |
| `conjugation_index_v3.py` | 偶联指数 v3 |
| `dar_v4_covalent.py` | DAR v4（共价动力学，复现偶数偏好） |
| `process_window_v4.py` | 工艺窗口：偶联收率 / TCEP 敏感性 |
| `make_summary_figure.py` | 面试一页图 |

---

## 4. 核心结论

1. **静态结构会误导**。A-Cys223 静态 SASA 4.4 Å²（看着埋着）→ 动态 44.8 Å²（实际最暴露）；
   反过来 A-Cys229 静态 54 → 动态只剩 10（假阳性）。这是整个项目的立足点。
2. **成键态二硫键普遍埋藏**（1IGT 晶体交叉验证最高也只有 30 Å²），
   所以"静态可及性"这个 proxy 区分度天生不足，必须用动态值。
3. **偶数 DAR 偏好不是拟合出来的**，是"偶联接近定量"的数学必然（见下节）。

---

## 5. DAR v3 → v4：为什么之前复现不出偶数峰

v3 用 **Langmuir 平衡吸附** 描述一个 **共价不可逆反应**。Langmuir 让 p_conj
随位点指数摊在 0.45–0.92，每对二硫键的"单臂泄漏" 2p(1−p) 高达 0.1–0.5，
叠加起来正好把偶数性抹平。

正确的物理分解：

- **位点指数 → 只决定 payload 能否触及**（几何门控）
- **反应条件 → 决定触及后是否反应**（共价二级动力学，与位点无关）

马来酰亚胺-巯基：conversion = 1 − exp(−k₂·[P]·t)。
k₂=300 M⁻¹s⁻¹、[P]=60 µM、t=2 h 时指数 ~130，理想溶液转化率 ≈ 1；
蛋白表面微环境封顶到实测工艺收率 0.92–0.98（本模型取 0.96）。

| 修复步骤 | even/odd |
|---|---|
| v3 现状 | 1.03 |
| 去同对惩罚 | 1.05（无效，被 C214 死亡掩盖） |
| 去空间门控 | 1.37 |
| **Langmuir → 共价动力学** | **6.05** ← 决定性 |
| 加还原选择性 | 6.03（非关键） |

v4 主结果（TCEP 2.75 eq）：mean 4.00、mode 4、even/odd 6.05，
分布 2:21% / 4:32% / 6:22%，奇数合计 14% —— HIC 图谱形态。

---

## 6. 已知局限（务必诚实说明）

1. **结构来自 Boltz 预测**，不是实验晶体。1IGT 交叉验证只能确认量级，不能确认细节。
2. **MD 只有 5 ns × 单个位点**，隐式溶剂、CA 约束锁住 Fab/Fc，
   低估了铰链还原后 Fab 的相对运动，因此可能低估 C214 / 229 的可及性。
   这直接导致"情形 A（even/odd 1.09）与实验矛盾"——是该局限的定量体现。
3. **还原易度 ease 混合了计算 proxy 与文献先验**（0.4 / 0.6），不是纯第一性原理。
4. **PROPKA 对 82% 埋藏的巯基给出 pKa 20+（物理荒谬）**，已 clamp 到 12 并留痕。
5. **同对双臂偶联未惩罚（PEN=1.0）**，依据是 DAR8 可制备；若实际有位阻需下调。
6. **隐式溶剂 GB 是 O(N²)**，2 万粒子在 4090 上实测仅 ~15 ns/day；
   想要更长轨迹需换显式溶剂 + PME，或做截断近似。
7. **还原易度与动态可及性不一致（已知待修）**：权重敏感性扫描显示，
   B-Cys229 / A-Cys232 在被随机权重抬高时会登顶，原因是"还原易度"分量给
   229/232 系打了高分（文献先验：铰链优先还原），但 A-Cys229 的动态
   open fraction 只有 0.175。物理上还原剂够不着埋着的二硫键，
   所以下一步应改为 `ease_effective = ease_prior × f(动态可及性)`。

---

## 7. 目录

```
examples/trop_adc/
├── data/                    结构与参考数据
├── kaggle_md_8site/        全部脚本（见上表）
└── results_hinge/          全部输出（json / png / md）
```
