"""Python twin of the dataviz skill's validate_palette.js (no node on this machine).

Ports the computable checks verbatim: OKLCH lightness band, chroma floor, CVD
separation under Machado-Oliveira-Fernandes severity 1.0, the normal-vision floor,
and WCAG contrast vs the chart surface. Thresholds copied from the JS source.
"""

from __future__ import annotations

import itertools
import math
import sys

BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0
NORMAL_FLOOR = 15.0
CONTRAST_MIN = 3.0
DEFAULT_SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}

MACHADO = {
    "protan": [[0.152286, 1.052583, -0.204868],
               [0.114503, 0.786281, 0.099216],
               [-0.003882, -0.048116, 1.051998]],
    "deutan": [[0.367322, 0.860646, -0.227968],
               [0.280085, 0.672501, 0.047413],
               [-0.011820, 0.042940, 0.968881]],
    "tritan": [[1.255528, -0.076749, -0.178779],
               [-0.078411, 0.930809, 0.147602],
               [0.004733, 0.691367, 0.303900]],
}


def hex2srgb(h: str):
    h = h.strip().lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def s2lin(c): return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
def lin(h): return [s2lin(c) for c in hex2srgb(h)]


def rel_lum(h):
    r, g, b = lin(h)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    hi, lo = sorted([rel_lum(a), rel_lum(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def oklab_from_lin(rgb):
    r, g, b = rgb
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return [0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s]


def oklch(h):
    L, a, b = oklab_from_lin(lin(h))
    return L, math.hypot(a, b)


def simulate(h, kind):
    r, g, b = lin(h)
    M = MACHADO[kind]
    return [min(1.0, max(0.0, M[i][0] * r + M[i][1] * g + M[i][2] * b)) for i in range(3)]


def delta_e(h1, h2, kind=None):
    a = oklab_from_lin(simulate(h1, kind) if kind else lin(h1))
    b = oklab_from_lin(simulate(h2, kind) if kind else lin(h2))
    return 100 * math.dist(a, b)


def validate(palette, mode="light", surface=None, pairs="adjacent"):
    surface = surface or DEFAULT_SURFACE[mode]
    lo, hi = BAND[mode]
    ok = True
    print(f"\nPalette ({mode}, surface {surface}, pairs {pairs}): {len(palette)} slots")

    off = [(c, round(oklch(c)[0], 3)) for c in palette if not (lo <= oklch(c)[0] <= hi)]
    ok &= not off
    print(f"  [{'PASS' if not off else 'FAIL'}] Lightness band        "
          f"{'all inside L %.2f-%.2f' % (lo, hi) if not off else 'outside: %s' % off}")

    lowc = [(c, round(oklch(c)[1], 3)) for c in palette if oklch(c)[1] < CHROMA_FLOOR]
    ok &= not lowc
    print(f"  [{'PASS' if not lowc else 'FAIL'}] Chroma floor          "
          f"{'all >= %.2f' % CHROMA_FLOOR if not lowc else 'below floor: %s' % lowc}")

    n = len(palette)
    pl = (list(itertools.combinations(range(n), 2)) if pairs == "all"
          else [(i, i + 1) for i in range(n - 1)])

    worst = min(((delta_e(palette[i], palette[j], k), k, palette[i], palette[j])
                 for k in ("protan", "deutan") for i, j in pl), key=lambda t: t[0])
    tri = min(delta_e(palette[i], palette[j], "tritan") for i, j in pl)
    state = "PASS" if worst[0] >= CVD_TARGET else "WARN" if worst[0] >= CVD_FLOOR else "FAIL"
    ok &= state != "FAIL"
    print(f"  [{state}] CVD separation        worst {worst[2]}<->{worst[3]} "
          f"dE {worst[0]:.1f} ({worst[1]}) · tritan {tri:.1f}")

    nworst = min(((delta_e(palette[i], palette[j]), palette[i], palette[j]) for i, j in pl),
                 key=lambda t: t[0])
    nstate = "PASS" if nworst[0] >= NORMAL_FLOOR else "FAIL"
    ok &= nstate == "PASS"
    print(f"  [{nstate}] Normal-vision floor   worst {nworst[1]}<->{nworst[2]} "
          f"dE {nworst[0]:.1f} (floor {NORMAL_FLOOR})")

    low = [(c, round(contrast(c, surface), 2)) for c in palette
           if contrast(c, surface) < CONTRAST_MIN]
    print(f"  [{'PASS' if not low else 'WARN'}] Contrast vs surface   "
          f"{'all >= 3:1' if not low else 'below 3:1 (needs labels): %s' % low}")

    print(f"  -> {'ALL CHECKS PASS' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    pal = [c.strip() for c in sys.argv[1].split(",") if c.strip()]
    mode = sys.argv[2] if len(sys.argv) > 2 else "light"
    surf = sys.argv[3] if len(sys.argv) > 3 else None
    prs = sys.argv[4] if len(sys.argv) > 4 else "adjacent"
    sys.exit(0 if validate(pal, mode, surf, prs) else 1)
