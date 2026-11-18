"""The time graph's analysis-region overlay (`viewer.add_region_shapes`): draggable boundary lines
at shapes[0] / shapes[1], and excluded-region shading that can never take a drag.

Why the layer matters: the `#time` graph config sets `edits.shapePosition`, which makes EVERY shape
draggable — a shape's own `editable=False` is ignored — and a vrect drawn above the lines with an
edge exactly on a line stole the grab (it resized / slid the gray block instead of moving the
boundary). The fix is CSS (assets/viewer.css): no pointer events on the shading paths, and no x-axis
drag strips under the graph (their ↔ cursor right under a line's end panned the view instead)."""
from pathlib import Path
from plotly.subplots import make_subplots
import viewer

CSS = Path(viewer.__file__).parent / "assets" / "viewer.css"


def _fig():
    """Two stacked subplots WITH a trace each: `add_vrect(row="all")` skips empty subplots, exactly
    as in build_figures, where the lines + shading are added after the traces."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    for r in (1, 2):
        fig.add_scatter(x=[0.0, 15.0], y=[0.0, 1.0], row=r, col=1)
    return fig


def test_boundary_lines_are_shapes_0_and_1_and_draggable():
    fig = viewer.add_region_shapes(_fig(), 3.12, 13.12, 0.0, 15.0)
    s = fig.layout.shapes
    assert [(sh.type, sh.x0, sh.x1) for sh in s[:2]] == [("line", 3.12, 3.12), ("line", 13.12, 13.12)]
    assert all(sh.editable is True and sh.yref == "paper" and sh.y0 == 0 and sh.y1 == 1 for sh in s[:2])
    assert all(sh.layer in (None, "above") for sh in s[:2])      # the lines stay on top


def test_excluded_shading_bands():
    fig = viewer.add_region_shapes(_fig(), 3.12, 13.12, 0.0, 15.0)
    rects = [sh for sh in fig.layout.shapes[2:]]
    assert rects and all(sh.type == "rect" and sh.editable is False for sh in rects)
    spans = sorted({(sh.x0, sh.x1) for sh in rects})
    assert spans == [(0.0, 3.12), (13.12, 15.0)]                 # one band each side, every subplot


def test_css_keeps_the_pointer_off_the_shading_and_the_axis_strips():
    css = CSS.read_text()
    assert '#time .shapelayer path[style*="pointer-events: all"] { pointer-events: none !important; }' in css
    assert '#time .draglayer .ewdrag, #time .draglayer .wdrag, #time .draglayer .edrag { display: none !important; }' in css


def test_crop_draws_the_lines_only():
    fig = viewer.add_region_shapes(_fig(), 3.12, 13.12, 0.0, 15.0, crop=True)
    assert len(fig.layout.shapes) == 2


def test_no_band_when_the_region_touches_the_edge():
    fig = viewer.add_region_shapes(_fig(), 0.0, 13.12, 0.0, 15.0)
    assert sorted({(sh.x0, sh.x1) for sh in fig.layout.shapes[2:]}) == [(13.12, 15.0)]
