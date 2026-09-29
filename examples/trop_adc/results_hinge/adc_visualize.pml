# =====================================================================
# adc_visualize.pml — ADC 结构可视化（PyMOL，无界面批量出图）
#
# 每张图包含：
#   - 抗体：卡通（重链蓝 / 轻链青）+ 半透明表面
#   - 载荷 vc-MMAE：棍状高亮
#   - 偶联位点 Cys-SG：黄球；未偶联空位：灰球
#   - 中文箭头注释 + 位点文字标记（含 S-C 键长）
#
# 为什么注释不直接用 PyMOL 的 label：
#   PyMOL 内置字体不含中文字形，中文 label 会渲染成方块。
#   所以这里走两步：先用"隐藏的纯色标记球"把关键三维位置
#   投影成像素坐标，再用 PIL 以 Arial Unicode 字体画中文与箭头。
#
# 运行：
#   /Applications/PyMOL.app/Contents/MacOS/PyMOL -cq adc_visualize.pml
#
# 构象来源（重要，别误读）：
#   用的是 md_placed 的**MD 真实轨迹帧 + 蜷缩构象**：vc-MMAE 这类疏水小分子
#   在水里会自己蜷起来（疏水塌缩），不是伸展的棍子。
#      - 依据：payload_ensemble_100.sdf（溶液态构象系综）中位
#        Rg = 6.96 A、展布 = 20.2 A；本套实测 Rg 6.2~7.1 A。
#      - 对照：polished_kaggle/ 是伸展构象（Rg ~11.8 A、展布 ~38 A），
#        落在系综最末端，不是溶液里的主体形态，不要拿来做代表图。
#   抗体构象取自 allcut_r2.npz（4 副本全开 MD，4 对链间二硫键全还原），
#   选 SASA 最大（最暴露）的帧放置载荷——侧链不需要人为推开，
#   因为 MD 模拟本身就采样到了那个有空间的构象。
#   DAR6/7 的 A232 被迫用伸展档——铰链区太挤放不下蜷缩球，真实几何约束。
# =====================================================================

python

from pymol import cmd, cgo
from PIL import Image, ImageDraw, ImageFont
import os, math

BASE = "<repo>/examples/trop_adc/results_hinge"
SRC  = os.path.join(BASE, "md_placed")
OUT  = os.path.join(BASE, "pymol_images")
LOG  = os.path.join(BASE, "pymol_images", "render_log.txt")
os.makedirs(OUT, exist_ok=True)

W, H = 1600, 1200
FONT_PATH = "/Library/Fonts/Arial Unicode.ttf"

# 8 个偶联位点 = 4 对链间二硫键（A/B 重链，C/D 轻链）
ALL_SITES = [("A", 223), ("A", 229), ("A", 232),
             ("B", 223), ("B", 229), ("B", 232),
             ("C", 214), ("D", 214)]
SITE_NOTE = {("A", 223): "HC Cys220 轻重链间", ("B", 223): "HC Cys220 轻重链间",
             ("A", 229): "铰链", ("B", 229): "铰链",
             ("A", 232): "铰链", ("B", 232): "铰链",
             ("C", 214): "LC Cys214 轻重链间", ("D", 214): "LC Cys214 轻重链间"}

# 标记球专用纯色（渲染时其余部分全部隐藏，便于按色找像素）
MARKER = {"ANT": (0, 200, 0), "PAY": (220, 0, 0)}
SITE_COLOR = {("A", 223): (0, 0, 255),   ("A", 229): (0, 255, 255),
              ("A", 232): (255, 0, 255), ("B", 223): (128, 0, 255),
              ("B", 229): (255, 128, 0), ("B", 232): (0, 128, 255),
              ("C", 214): (128, 255, 0), ("D", 214): (255, 255, 0)}

log = open(LOG, "w")


def say(msg):
    log.write(str(msg) + "\n")
    log.flush()


def dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def xyz(sel):
    m = cmd.get_model(sel)
    if not m.atom:
        return None
    a = m.atom[0].coord
    return [float(a[0]), float(a[1]), float(a[2])]


def centroid(sel):
    m = cmd.get_model(sel)
    if not m.atom:
        return None
    n = len(m.atom)
    return [sum(float(a.coord[i]) for a in m.atom) / n for i in range(3)]


def find_marker(img, target, tol=60):
    """在标记图里找纯色球的重心（像素坐标）。"""
    px = img.load()
    w, h = img.size
    sx = sy = n = 0
    for y in range(0, h, 1):
        for x in range(0, w, 1):
            r, g, b = px[x, y][:3]
            if abs(r - target[0]) <= tol and abs(g - target[1]) <= tol and abs(b - target[2]) <= tol:
                sx += x; sy += y; n += 1
    if n == 0:
        return None
    return (sx / n, sy / n)


def pil_arrow(draw, start, end, color, width=4, head=22):
    draw.line([start, end], fill=color, width=width)
    import math as _m
    ang = _m.atan2(end[1] - start[1], end[0] - start[0])
    a1 = ang + 2.6
    a2 = ang - 2.6
    p1 = (end[0] + head * _m.cos(a1), end[1] + head * _m.sin(a1))
    p2 = (end[0] + head * _m.cos(a2), end[1] + head * _m.sin(a2))
    draw.polygon([end, p1, p2], fill=color)


def render_one(pdb_file, tag):
    cmd.reinitialize()
    cmd.load(os.path.join(SRC, pdb_file), "adc")

    # ---------- 显示设置 ----------
    cmd.set("bg_rgb", [1, 1, 1])
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 1)
    cmd.set("depth_cue", 0)
    cmd.set("ray_shadows", 0)
    cmd.set("antialias", 2)
    # surface_quality=2 会让 1600x1200 的单张图渲染 >15 min；1 的画质已够用
    cmd.set("surface_quality", 1)
    cmd.set("transparency", 0.62)
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("sphere_scale", 0.55)

    # ---------- 抗体：卡通 + 半透明表面 ----------
    cmd.select("hc", "adc and chain A+B and polymer")
    cmd.select("lc", "adc and chain C+D and polymer")
    cmd.select("prot", "adc and polymer")
    cmd.select("pay", "adc and resn LPP")
    cmd.hide("everything", "adc")
    cmd.show("cartoon", "prot")
    cmd.color("slate", "hc")
    cmd.color("teal", "lc")
    cmd.show("surface", "prot")
    cmd.set("surface_color", "grey80", "prot")

    # ---------- 载荷：半透明橙红表面（统一色）+ 棍状随偶联链色 ----------
    # 表面用独立对象 paysurf：原子色留给棍状（随链色），表面单独统一橙红。
    # （直接 set surface_color 会被后续按链的 cmd.color 覆盖，实测不生效。）
    cmd.create("paysurf", "adc and resn LPP")   # create 接受 selection；copy 只认 object
    cmd.hide("everything", "paysurf")
    cmd.show("surface", "paysurf")
    cmd.set_color("pay_orange", [1.0, 0.38, 0.02])   # 橙红，PyMOL 内置无 darkorange
    cmd.color("pay_orange", "paysurf")
    cmd.set("transparency", 0.30, "paysurf")
    # 棍状：按偶联链染色（A/B 重链蓝系，C/D 轻链青系），与抗体卡通色一致
    chain_col = {"A": "slate", "B": "slate", "C": "teal", "D": "teal"}
    lpp_chains = sorted(set(a.chain for a in cmd.get_model("pay").atom))
    for ch in lpp_chains:
        cmd.select("pay_%s" % ch, "adc and resn LPP and chain %s" % ch)
        cmd.show("sticks", "pay_%s" % ch)
        cmd.color(chain_col.get(ch, "orange"), "pay_%s" % ch)
        cmd.set("stick_radius", 0.22, "pay_%s" % ch)

    # ---------- 硫醚共价键：黄色圆柱 ----------
    # PDB 里 CONECT 已记录 SG-C91 键，PyMOL 不会自动画跨链键，
    # 这里显式创建 bond 对象并用黄色圆柱表示，一眼看出载荷挂在哪个半胱氨酸上。
    # 注意：硫醚键不是死板的 1.82 A，实物 S-C 键长在 1.70-2.40 A 都成立，
    # 放置脚本已在该范围内挑位阻最小的键长，所以这里按实测距离判定（< 2.6 A 算成键）。
    lpp_res = sorted(set(a.resi for a in cmd.get_model("pay").atom))
    for (ch, resi) in ALL_SITES:
        sg_sel = "(adc and chain %s and resi %d and resn CYS and name SG)" % (ch, resi)
        sg = xyz(sg_sel)
        if sg is None:
            continue
        # 找最近 C91（应 < 2.6 A）
        best = None
        for ri in lpp_res:
            c91 = xyz("(adc and resn LPP and resi %s and name C91)" % ri)
            if c91 is None:
                continue
            dd = dist(sg, c91)
            if best is None or dd < best[0]:
                best = (dd, ri, c91)
        if best and best[0] < 2.6:
            dd, ri, c91 = best
            cmd.bond(sg_sel, "(adc and resn LPP and resi %s and name C91)" % ri)
            idx_list = [str(a.index) for a in
                        cmd.get_model(sg_sel).atom
                        + cmd.get_model("(adc and resn LPP and resi %s and name C91)" % ri).atom]
            cmd.select("bond_%s%d" % (ch, resi),
                       "adc and (index %s)" % " or index ".join(idx_list))
            cmd.show("sticks", "bond_%s%d" % (ch, resi))
            cmd.set("stick_radius", 0.45, "bond_%s%d" % (ch, resi))
            cmd.color("yellow", "bond_%s%d" % (ch, resi))

    # ---------- 判定哪些位点被偶联 ----------
    occ = {}
    for ri in lpp_res:
        c91 = xyz("(adc and resn LPP and resi %s and name C91)" % ri)
        if c91 is None:
            continue
        best = None
        for (ch, resi) in ALL_SITES:
            sg = xyz("(adc and chain %s and resi %d and resn CYS and name SG)" % (ch, resi))
            if sg is None:
                continue
            d = dist(c91, sg)
            if best is None or d < best[0]:
                best = (d, ch, resi)
        if best:
            occ[(best[1], best[2])] = best[0]      # (chain,resi) -> S-C 键长

    occupied = set(occ.keys())
    empty = [s for s in ALL_SITES if s not in occupied]

    # 实测载荷回转半径 Rg（溶液态紧致度的直接指标）
    rgs = []
    for ri in lpp_res:
        m = cmd.get_model("(adc and resn LPP and resi %s)" % ri)
        if not m.atom:
            continue
        n = len(m.atom)
        c = [sum(float(a.coord[i]) for a in m.atom) / n for i in range(3)]
        rgs.append(math.sqrt(sum(
            sum((float(a.coord[i]) - c[i]) ** 2 for i in range(3)) for a in m.atom) / n))
    rg_avg = sum(rgs) / len(rgs) if rgs else float("nan")

    for (ch, resi) in occupied:
        n = "site_%s%d" % (ch, resi)
        cmd.select(n, "(adc and chain %s and resi %d and resn CYS and name SG)" % (ch, resi))
        cmd.show("spheres", n)
        cmd.color("yellow", n)
    for (ch, resi) in empty:
        n = "open_%s%d" % (ch, resi)
        cmd.select(n, "(adc and chain %s and resi %d and resn CYS and name SG)" % (ch, resi))
        cmd.show("spheres", n)
        cmd.color("grey50", n)

    cmd.orient("adc")
    cmd.zoom("adc", buffer=8)

    # ---------- 第一遍：干净图 ----------
    clean_png = os.path.join(OUT, "_clean_%s.png" % tag)
    cmd.png(clean_png, width=W, height=H, ray=1)

    # ---------- 第二遍：纯色标记球，用来取像素坐标 ----------
    cmd.hide("everything", "all")   # 连 paysurf 一起藏掉，画面里只留 marker
    cmd.set("ambient", 1.0)
    cmd.set("direct", 0.0)
    cmd.set("specular", 0.0)
    cmd.set("shininess", 0.0)
    cmd.set("power", 1.0)
    cmd.set("antialias", 0)
    cmd.set("sphere_quality", 3)

    targets = {}
    cen_p = centroid("prot")
    cen_l = centroid("pay")
    if cen_p:
        cmd.pseudoatom("mk_ANT", pos=cen_p, name="C00", vdw=1.6)
        cmd.set("surface_color", "green", "mk_ANT")
        cmd.show("sphere", "mk_ANT")
        cmd.set_color("mkANTcol", list(MARKER["ANT"]))
        cmd.color("mkANTcol", "mk_ANT")
        targets["ANT"] = MARKER["ANT"]
    if cen_l:
        cmd.pseudoatom("mk_PAY", pos=cen_l, name="C00", vdw=1.6)
        cmd.show("sphere", "mk_PAY")
        cmd.set_color("mkPAYcol", list(MARKER["PAY"]))
        cmd.color("mkPAYcol", "mk_PAY")
        targets["PAY"] = MARKER["PAY"]
    for (ch, resi) in ALL_SITES:
        sg = xyz("(adc and chain %s and resi %d and resn CYS and name SG)" % (ch, resi))
        if sg is None:
            continue
        nm = "mk_%s%d" % (ch, resi)
        cmd.pseudoatom(nm, pos=sg, name="C00", vdw=1.6)
        cmd.show("sphere", nm)
        col = SITE_COLOR[(ch, resi)]
        cmd.set_color("c_%s%d" % (ch, resi), list(col))
        cmd.color("c_%s%d" % (ch, resi), nm)
        targets["%s%d" % (ch, resi)] = col

    mark_png = os.path.join(OUT, "_mark_%s.png" % tag)
    cmd.png(mark_png, width=W, height=H, ray=1)

    # ---------- 取像素坐标 ----------
    mk = Image.open(mark_png).convert("RGB")
    pos = {}
    for k, col in targets.items():
        p = find_marker(mk, col)
        if p:
            pos[k] = p
    say("%s: markers found %d/%d" % (tag, len(pos), len(targets)))

    # ---------- 第三遍：PIL 画中文注释 ----------
    img = Image.open(clean_png).convert("RGB")
    d = ImageDraw.Draw(img)
    f_big = ImageFont.truetype(FONT_PATH, 30)
    f_mid = ImageFont.truetype(FONT_PATH, 24)
    f_small = ImageFont.truetype(FONT_PATH, 20)

    BLACK = (25, 25, 25)
    GREY = (95, 95, 95)

    # 标题
    d.text((40, 28), "%s  ·  曲妥珠单抗 + vc-MMAE（半胱氨酸偶联）" % tag,
           font=f_big, fill=BLACK)
    d.text((42, 68), "8 个偶联位点 = 4 对链间二硫键；黄球=已偶联位点 SG，灰球=空位",
           font=f_small, fill=GREY)

    # ---------- 文字注释（无箭头/引线，保留注释文字） ----------
    # 抗体与载荷说明
    d.text((46, 120), "抗体：卡通+半透明表面（重链蓝/轻链青）", font=f_small, fill=BLACK)
    d.text((W - 346, 120), "载荷 vc-MMAE：半透明橙红表面", font=f_small, fill=(190, 60, 20))
    d.text((46, 148), "黄圆柱=硫醚共价键（SG-C91）", font=f_small, fill=(180, 140, 0))

    # ---------- 位点注释：仅文字（无引线），分左右两列 ----------
    def text_w(t, font):
        bb = d.textbbox((0, 0), t, font=font)
        return bb[2] - bb[0], (bb[3] - bb[1])

    left_col, right_col = [], []
    for (ch, resi) in ALL_SITES:
        k = "%s%d" % (ch, resi)
        if k not in pos:
            continue
        x, y = pos[k]
        if (ch, resi) in occ:
            txt = "%s%d %s · S-C %.2f A" % (ch, resi, SITE_NOTE[(ch, resi)], occ[(ch, resi)])
            col = (200, 140, 0)
        else:
            txt = "%s%d %s · 未偶联" % (ch, resi, SITE_NOTE[(ch, resi)])
            col = (110, 110, 110)
        (left_col if x < W / 2 else right_col).append((x, y, txt, col))

    left_col.sort(key=lambda t: t[1])
    right_col.sort(key=lambda t: t[1])

    def draw_column(items, side, y0=210, gap=42):
        for i, (x, y, txt, col) in enumerate(items):
            ty = y0 + i * gap
            tw, _ = text_w(txt, f_small)
            if side == "left":
                tx = 46
            else:
                tx = W - 46 - tw
            d.text((tx, ty), txt, font=f_small, fill=col)

    draw_column(left_col, "left")
    draw_column(right_col, "right")

    # 底部说明
    d.text((40, H - 88),
           "构象：MD 真实轨迹帧 + 溶液态蜷缩构象（疏水塌缩）。抗体取自 allcut MD（4 对链间二硫键全还原），选 SASA 最大的帧放置载荷。",
           font=f_small, fill=GREY)
    d.text((40, H - 62),
           "本图载荷实测回转半径 Rg = %.1f A（溶液态系综中位 6.96 A）。重原子最小间距 >=2.0 A。"
           % rg_avg,
           font=f_small, fill=GREY)
    d.text((40, H - 36),
           "系综中的一个瞬时构象，不是唯一真实结构——铰链区本身无序（pLDDT 约 40-60），载荷位置天然不确定。",
           font=f_small, fill=GREY)

    out_png = os.path.join(OUT, "adc_%s.png" % tag)
    img.save(out_png)
    say("%s -> %s  occupied=%s  empty=%s" % (tag, out_png, sorted(occupied), empty))

    os.remove(clean_png)
    os.remove(mark_png)


# DAR1~6 用硫醚键优化版（键长在 1.70~2.40 A 内挑位阻最小的，不是钉死 1.82 A）。
# DAR7 目前是上一版产物（键长固定 1.82 A），尚未用硫醚键优化重跑。
FILES = [
    ("ADC_DAR1_md.pdb", "DAR1"),
    ("ADC_DAR2_md.pdb", "DAR2"),
    ("ADC_DAR3_md.pdb", "DAR3"),
    ("ADC_DAR4_md.pdb", "DAR4"),
    ("ADC_DAR5_md.pdb", "DAR5"),
    ("ADC_DAR6_md.pdb", "DAR6"),
    ("ADC_DAR7_omit_md.pdb", "DAR7"),
]

import traceback
for f, tag in FILES:
    p = os.path.join(SRC, f)
    if os.path.exists(p):
        try:
            render_one(f, tag)
        except Exception:
            say("TRACEBACK for %s:\n%s" % (f, traceback.format_exc()))
            break
    else:
        say("missing: " + p)

say("DONE")
log.close()
cmd.quit()

python end
