"""
Draws the class diagram of the code (docs/figures/class_diagram.png).

Run from the repository root:

    python scripts/make_class_diagram.py

The boxes are the real classes of src/ and app/. How to read the arrows:

    hollow triangle   inherits from
    filled diamond    owns (the owner creates the part)
    open diamond      is given (the object is passed in, not created)
    dashed arrow      uses

The script checks that no text runs outside its box and prints a warning if one does.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle  # noqa: E402

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"]

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "figures", "class_diagram.png")
WIDTH_IN, HEIGHT_IN = 7.0, 4.15
W = 200.0
H = W * HEIGHT_IN / WIDTH_IN            # drawing units are square
UNIT_IN = WIDTH_IN / W                  # inches per drawing unit

INK, GREY = "#1f2933", "#52606d"
FILL_ABSTRACT, FILL_CLASS, FILL_OTHER = "#e3ebf6", "#ffffff", "#eef0f3"
FRAME_FILL, FRAME_EDGE = "#f8f9fb", "#b8c0cc"


def line_height(font_size):
    """Height of one line of text in drawing units."""
    return font_size * 1.3 / 72 / UNIT_IN


class Diagram:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(WIDTH_IN, HEIGHT_IN), dpi=300)
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(0, H)
        self.ax.axis("off")
        self.fig.subplots_adjust(left=0.003, right=0.997, top=0.997, bottom=0.003)
        self.texts = []   # (text artist, box rectangle, name) for the overflow check

    # ---- boxes -------------------------------------------------------------------------
    def frame(self, x, y, w, h, label):
        self.ax.add_patch(Rectangle((x, y), w, h, fc=FRAME_FILL, ec=FRAME_EDGE, lw=0.6, zorder=0))
        self.ax.text(x + 2, y + h - 1.6, label, fontsize=5.6, fontweight="bold", color=GREY, va="top", zorder=4)

    def box(self, x, y, w, h, title, body="", abstract=False, fill=None, fs=6.0, fs_body=4.9):
        fill = fill or (FILL_ABSTRACT if abstract else FILL_CLASS)
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.9",
                                         fc=fill, ec=INK, lw=0.6, zorder=3))
        title_h = line_height(fs) * (title.count("\n") + 1)
        body_h = line_height(fs_body) * (body.count("\n") + 1) if body else 0.0
        gap = 0.6 if body else 0.0
        top = y + h / 2 + (title_h + gap + body_h) / 2
        cx = x + w / 2
        t1 = self.ax.text(cx, top, title, ha="center", va="top", fontsize=fs, fontweight="bold", color=INK,
                          style="italic" if abstract else "normal", linespacing=1.15, zorder=4)
        self.texts.append((t1, (x, y, w, h), title.replace("\n", " ")))
        if body:
            t2 = self.ax.text(cx, top - title_h - gap, body, ha="center", va="top", fontsize=fs_body,
                              color=INK, linespacing=1.15, zorder=4)
            self.texts.append((t2, (x, y, w, h), title.replace("\n", " ") + " (body)"))
        return x, y, w, h

    # ---- lines -----------------------------------------------------------------------------
    def line(self, xs, ys, dashed=False):
        self.ax.plot(xs, ys, color=INK, lw=0.6, zorder=2, ls=(0, (3, 2)) if dashed else "-",
                     solid_capstyle="butt")

    def triangle(self, x, y):
        """Hollow triangle with its tip at (x, y), pointing up (the parent's bottom edge)."""
        self.ax.add_patch(Polygon([[x, y], [x - 1.2, y - 2.0], [x + 1.2, y - 2.0]], closed=True,
                                  fc="white", ec=INK, lw=0.6, zorder=3))

    def inherit(self, parent, children, bus_y, tip_x=None):
        """One triangle on the parent, a bus line, and a drop to each child's top edge."""
        px = parent[0] + parent[2] / 2 if tip_x is None else tip_x
        py = parent[1]
        self.triangle(px, py)
        self.line([px, px], [py - 2.0, bus_y])
        centres = [c[0] + c[2] / 2 for c in children]
        self.line([min(centres + [px]), max(centres + [px])], [bus_y, bus_y])
        for child, cx in zip(children, centres):
            self.line([cx, cx], [bus_y, child[1] + child[3]])

    def inherit_stack(self, parent, children, bus_x):
        """Children stacked under the parent, joined by a vertical bus on their left."""
        py = parent[1]
        self.triangle(bus_x, py)
        last_y = children[-1][1] + children[-1][3] / 2
        self.line([bus_x, bus_x], [py - 2.0, last_y])
        for child in children:
            cy = child[1] + child[3] / 2
            self.line([bus_x, child[0]], [cy, cy])

    def diamond(self, x, y, filled, horizontal=False):
        a, b = 1.3, 0.8
        pts = [[x - a, y], [x, y + b], [x + a, y], [x, y - b]] if horizontal else \
              [[x, y + a], [x + b, y], [x, y - a], [x - b, y]]
        self.ax.add_patch(Polygon(pts, closed=True, fc=INK if filled else "white", ec=INK, lw=0.6, zorder=3))

    def uses(self, start, end):
        self.ax.annotate("", xy=end, xytext=start, zorder=2,
                         arrowprops=dict(arrowstyle="->,head_length=0.35,head_width=0.18", lw=0.6, color=INK,
                                         linestyle=(0, (3, 2)), shrinkA=0, shrinkB=0))

    def note(self, x, y, text, **kw):
        kw.setdefault("fontsize", 4.8)
        kw.setdefault("color", INK)
        self.ax.text(x, y, text, va="center", zorder=5, **kw)

    # ---- check and save -------------------------------------------------------------------------
    def overflow_report(self):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        inverse = self.ax.transData.inverted()
        problems = []
        for artist, (x, y, w, h), name in self.texts:
            (x0, y0), (x1, y1) = inverse.transform(artist.get_window_extent(renderer).get_points())
            margin = 0.7
            if x0 < x + margin or x1 > x + w - margin or y0 < y + 0.3 or y1 > y + h - 0.3:
                problems.append(f"  text runs outside its box: {name}  "
                                f"(text {x1 - x0:.1f} x {y1 - y0:.1f}, box {w:.1f} x {h:.1f})")
        return problems

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.fig.savefig(path, dpi=300)


TOP_Y, FRAME_H = 32, 84          # where the three frames start (from the bottom) and how tall they are


def draw_stage1_models(d):
    d.frame(0, TOP_Y, 78, FRAME_H, "stage1.py")
    s1 = d.box(20, 93, 42, 16, "Stage1Model", "fit()  predict()\npredict_with_disagreement()", abstract=True)
    median = d.box(1, 70, 14, 14, "Median\nBaseline", fs=5.8)
    carry = d.box(16.5, 70, 17, 14, "CarryForward\nBaseline", fs=5.8)
    twoin = d.box(35, 70, 14, 14, "TwoInput\nLinear", fs=5.8)
    matrix = d.box(50.5, 70, 17, 14, "MatrixModel", abstract=True, fs=5.8)
    ridge = d.box(41, 47, 15, 12, "RidgeModel", fs=5.8)
    forest = d.box(58, 47, 19, 12, "RandomForest\nModel", fs=5.8)
    d.inherit(s1, [median, carry, twoin, matrix], bus_y=88)
    d.inherit(matrix, [ridge, forest], bus_y=64)


def draw_the_two_stages_together(d):
    d.frame(84, TOP_Y, 54, FRAME_H, "forecaster.py, stage2.py")
    d.box(88, 48, 46, 18, "TwoStageForecaster",
          "fit()  forecast()  is_kept()\ncalibrate_demo_cutoff()\nto_bundles()  from_files()")
    d.box(92, 76, 38, 12, "Stage2ErrorEstimator", "fit()  predict()")
    d.box(92, 97, 38, 10, "RandomForestRegressor", "(scikit-learn)", fill=FILL_OTHER)
    d.line([111, 111], [66, 76])
    d.diamond(111, 67.3, filled=True)
    d.line([111, 111], [88, 97])
    d.diamond(111, 89.3, filled=True)
    d.note(113, 71, "owns")
    d.note(113, 92.5, "owns")
    # Stage 1 forest is handed to the forecaster, not built by it
    d.line([88, 77], [55, 55])
    d.diamond(86.7, 55, filled=False, horizontal=True)
    d.note(82.5, 57.6, "is given", ha="center",
           bbox=dict(fc="white", ec="none", pad=0.3))


def draw_selectors(d):
    d.frame(144, TOP_Y, 56, FRAME_H, "selection.py")
    selector = d.box(155, 98, 34, 11, "Selector", "summarise()", abstract=True)
    score = d.box(145, 78, 31, 12, "ScoreSelector", "score()  keep_mask()", abstract=True)
    random_sel = d.box(178, 78, 21, 12, "RandomSelector", "200 random draws", fs=5.4, fs_body=4.6)
    d.inherit(selector, [score, random_sel], bus_y=93.5)
    names = ["LearnedStage2Selector", "TreeDisagreementSelector", "SimpleRuleSelector", "OracleSelector"]
    leaves = []
    for i, name in enumerate(names):
        top = 74.5 - i * 8.5
        leaves.append(d.box(153, top - 6.8, 45, 6.8, name, fs=5.4))
    d.inherit_stack(score, leaves, bus_x=149.5)


def draw_who_calls_them(d):
    d.box(0, 6, 50, 18, "app/main.py", "GET /health   POST /forecast\nDepends(get_service)",
          fill=FILL_OTHER, fs=5.8)
    d.box(56, 6, 50, 18, "ForecastService", "forecast(features)\nthe model loader is passed in", fs=5.8)
    d.box(112, 6, 56, 18, "evaluation.py",
          "evaluate_partition(df, forecaster, selectors)\nexact coverage, paired bootstrap",
          fill=FILL_OTHER, fs=5.8)
    d.box(173, 6, 27, 18, "scripts", "run_stage1.py\nrun_stage2.py\nrun_locked_test.py",
          fill=FILL_OTHER, fs=5.8)
    d.uses((50, 15), (56, 15))
    d.uses((173, 15), (168, 15))
    d.uses((96, 24), (96, 48))        # service -> forecaster
    d.uses((124, 24), (124, 48))      # evaluation -> forecaster
    d.uses((156, 24), (156, TOP_Y))   # evaluation -> selectors
    d.note(158, 28, "any Selector", ha="left")


def draw_legend(d):
    d.note(1, 2.2, "Hollow triangle: inherits from.   Filled diamond: owns (creates) the part.   "
                   "Open diamond: is given the object.   Dashed arrow: uses.   Italic name: abstract class.",
           ha="left", fontsize=4.7, color=GREY)


def build():
    d = Diagram()
    for draw in (draw_stage1_models, draw_the_two_stages_together, draw_selectors, draw_who_calls_them, draw_legend):
        draw(d)
    return d


def main():
    d = build()
    problems = d.overflow_report()
    d.save(OUT)
    print("saved", os.path.abspath(OUT))
    if problems:
        print("\n".join(problems))
        raise SystemExit(1)
    print("no text runs outside its box")


if __name__ == "__main__":
    main()
