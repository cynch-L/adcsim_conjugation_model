#!/usr/bin/env python3
"""Render the README ADC figures (overview + conjugation close-up).

Run with the Python bundled inside PyMOL.app:

    /Applications/PyMOL.app/Contents/MacOS/python render_readme_figures.py [DAR1|DAR4|DAR7]

Structures come from results_hinge/polished_collapsed/ (the solution-like
collapsed ensemble we agreed to use for representative figures). The
conjugated sites listed in FIGURES were read out of the coordinates
themselves, so labels cannot drift from the structures.

Two views per figure:
  overview - whole antibody, rotated so the conjugation cluster faces the
             viewer (fixes the round-1 problem of payloads hidden behind
             the molecule);
  closeup  - the conjugation cluster alone, with per-site labels, the
             translucent payload surface and the SG-C91 bond distances.
annotate_figure.py pastes them side by side with a legend.
"""
import json
import os
import subprocess
import sys

import pymol

pymol.finish_launching(["pymol", "-cq"])

from pymol import cmd  # noqa: E402  (needs the launched interpreter)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results_hinge")
COLLAPSED = os.path.join(RESULTS, "polished_collapsed")
OUTDIR = os.path.join(RESULTS, "pymol_images")

PYMOL_PYTHON = sys.executable
ANNOTATE = os.path.join(HERE, "annotate_figure.py")

HEAVY_A, HEAVY_B = "0x3C6E9F", "0x7FA9CE"
LIGHT_C, LIGHT_D = "0x2E9F8F", "0x6FC7B6"
PAYLOAD_RGB = (0.878, 0.482, 0.224)   # orange
LINKER_RGB = (0.85, 0.68, 0.13)       # amber, conjugated cysteine

# (figure tag, structure, conjugated sites)
FIGURES = [
    ("DAR1", "ADC_DAR1_collapsed.pdb",
     [("A", 232)]),
    ("DAR4", "ADC_DAR4_collapsed.pdb",
     [("A", 232), ("A", 229), ("A", 223), ("D", 214)]),
    ("DAR7", "ADC_DAR7_omit_B232_collapsed.pdb",
     [("A", 232), ("A", 229), ("A", 223),
      ("B", 229), ("B", 223),
      ("C", 214), ("D", 214)]),
]

WIDTH, HEIGHT, DPI = 2000, 1500, 220
LEGEND = [
    ((0.235, 0.431, 0.624), "Heavy chain A"),
    ((0.498, 0.663, 0.808), "Heavy chain B"),
    ((0.180, 0.624, 0.561), "Light chain C"),
    ((0.435, 0.780, 0.714), "Light chain D"),
    (PAYLOAD_RGB, "vc-MMAE payload"),
    (LINKER_RGB, "Conjugated cysteine"),
]


def style_common():
    cmd.bg_color("white")
    cmd.set("opaque_background", "on")
    cmd.set("ray_trace_mode", 0)
    cmd.set("ray_shadow", 0)
    cmd.set("specular", 0.30)
    cmd.set("antialias", 2)
    cmd.set("cartoon_transparency", 0.30)
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("cartoon_smooth_loops", 1)


def style_payload(closeup):
    """payload as ball-and-stick; spheres scaled so the hinge crowd stays readable"""
    cmd.set_color("payload", list(PAYLOAD_RGB))
    cmd.set_color("linker", list(LINKER_RGB))
    cmd.show("sticks", "resn LPP")
    cmd.color("payload", "resn LPP")
    cmd.show("spheres", "resn LPP")
    if closeup:
        cmd.set("stick_radius", 0.30, "resn LPP")
        cmd.set("sphere_scale", 0.28, "resn LPP")
    else:
        cmd.set("stick_radius", 0.40, "resn LPP")
        cmd.set("sphere_scale", 0.20, "resn LPP")


def highlight_conjugated(sites, closeup):
    conj = " or ".join("(chain %s and resi %d and resn CYS)" % (c, i) for c, i in sites)
    cmd.show("sticks", conj)
    cmd.color("linker", conj)
    cmd.set("stick_radius", 0.34 if closeup else 0.46, conj)


def render(tag, pdb, sites):
    path = os.path.join(COLLAPSED, pdb)
    if not os.path.exists(path):
        print("missing:", path, file=sys.stderr)
        return False

    conj = " or ".join("(chain %s and resi %d and resn CYS)" % (c, i) for c, i in sites)

    cmd.reinitialize()
    cmd.load(path, "adc")
    cmd.remove("solvent")
    style_common()
    for ch, colour in (("A", HEAVY_A), ("B", HEAVY_B), ("C", LIGHT_C), ("D", LIGHT_D)):
        cmd.color(colour, "chain %s and polymer" % ch)
    # reinitialize wipes named selections -> define focus after each reload
    cmd.select("focus", "resn LPP or (%s)" % conj)

    # rotate the conjugation cluster towards the viewer, then frame
    cmd.orient("focus")

    os.makedirs(OUTDIR, exist_ok=True)
    stems = []

    # ---- overview: the whole antibody, cluster facing the camera -------
    style_payload(closeup=False)
    highlight_conjugated(sites, closeup=False)
    cmd.zoom("polymer", buffer=18)
    raw = os.path.join(OUTDIR, "_raw_%s_overview.png" % tag)
    cmd.png(raw, width=WIDTH, height=HEIGHT, dpi=DPI, ray=1)
    stems.append(raw)

    # ---- close-up: the conjugation cluster with labels and distances ---
    cmd.reinitialize()
    cmd.load(path, "adc")
    cmd.remove("solvent")
    style_common()
    cmd.set("cartoon_transparency", 0.45)
    for ch, colour in (("A", HEAVY_A), ("B", HEAVY_B), ("C", LIGHT_C), ("D", LIGHT_D)):
        cmd.color(colour, "chain %s and polymer" % ch)
    style_payload(closeup=True)
    highlight_conjugated(sites, closeup=True)
    cmd.select("focus", "resn LPP or (%s)" % conj)

    cmd.orient("focus")
    cmd.zoom("focus", buffer=15)

    # 5+ sites: full "A-Cys232" labels overlap; fall back to numbered tags
    # plus a site key drawn under the panels by annotate_figure.py.
    numbered = len(sites) > 5
    if numbered:
        # wider spread for single digits so 7 tags never touch each other
        offsets = [(7, 7, 7), (7, -7, -7), (-7, 7, 7), (-7, -7, -7),
                   (0, 9, -7), (0, -9, 7), (9, 0, -7), (-9, 0, 7)]
    else:
        offsets = [(4, 4, 4), (4, -4, -4), (-4, 4, 4), (-4, -4, -4),
                   (0, 5, -4), (0, -5, 4), (5, 0, 4), (-5, 0, -4)]
    cmd.set("label_color", "black")
    cmd.set("label_outline_color", "white")
    cmd.set("label_font_id", 7)
    if numbered:
        cmd.set("label_size", 20)
    else:
        cmd.set("label_size", 24)
    site_key = []
    for idx, ((chain, resi), off) in enumerate(zip(sites, offsets)):
        obj = "site_%s%d" % (chain, resi)
        cmd.select(obj, "chain %s and resi %d and name SG" % (chain, resi))
        if numbered:
            cmd.label(obj, '"%d"' % (idx + 1))
            site_key.append("%d = %s-Cys%d" % (idx + 1, chain, resi))
        else:
            cmd.label(obj, '"%s-Cys%d"' % (chain, resi))
        cmd.set("label_position", "(%g,%g,%g)" % off, obj)

    raw = os.path.join(OUTDIR, "_raw_%s_closeup.png" % tag)
    cmd.png(raw, width=WIDTH, height=HEIGHT, dpi=DPI, ray=1)
    stems.append(raw)

    final = os.path.join(OUTDIR, "adc_%s.png" % tag)
    subprocess.run(
        [PYMOL_PYTHON, ANNOTATE,
         "--tag", tag,
         "--output", final,
         "--overview", stems[0],
         "--closeup", stems[1],
         "--sites", json.dumps(sites),
         "--legend", json.dumps(LEGEND),
         "--site_key", json.dumps(site_key)],
        check=True)
    print("wrote", final)
    return True


def main(only=None):
    done = []
    for tag, pdb, sites in FIGURES:
        if only and tag != only:
            continue
        if render(tag, pdb, sites):
            done.append(tag)
    print("done:", ", ".join(done) or "nothing")


if __name__ == "__main__":
    main(sys.argv[1].upper() if len(sys.argv) > 1 else None)
