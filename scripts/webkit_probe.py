"""WebKit probe — open a viewer in a pywebview (WKWebView) window and run JavaScript in it.

The desktop app renders in WKWebView, which hit-tests SVG differently from Chrome (2026-11-18: a
path with no `stroke` is never hit, even under pointer-events:all — Plotly's shape hit-paths). The
browser pane is Chromium, so the only way to test the app's engine headlessly is this: a pywebview
window on a TEST server (port 8052 — never the user's 8050), `evaluate_js` from a thread, print, close.

    python scripts/webkit_probe.py '{"cell":["2026-11-12|c04"],"files":[".../raw/2026_11_12_0042.abf"]}' [url]

The JSON is written into localStorage as `last-session` so the page opens on real data. NOTE the UI
uses CSS zoom (body 0.9 at 1440×900): compute hit-test points from an element's own
getBoundingClientRect(), never from Plotly's _fullLayout pixel offsets, or elementFromPoint lands
elsewhere. Edit PROBE / DRAG below for a new question; AFTER reads the region boxes back.
"""
import json
import sys
import time

import webview

SESSION = sys.argv[1] if len(sys.argv) > 1 else ""
URL = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8052/"

PROBE = r"""
(function(){
  var gd = document.querySelector('#time .js-plotly-plot');
  function d(el){ return el ? (el.tagName + '.' + (el.getAttribute('class') || '') + (el.parentNode.getAttribute && el.parentNode.getAttribute('drag-helper') ? '@helper' : '')) : null; }
  var paths = Array.prototype.slice.call(gd.querySelectorAll('[drag-helper] path'));
  var hp = paths[2]; var hr = hp.getBoundingClientRect(); var cx = hr.left, cy = hr.top + hr.height / 2;
  var out = {withStroke: [d(document.elementFromPoint(cx, cy)), d(document.elementFromPoint(cx + 8, cy))]};
  paths.forEach(function(p){ p.style.setProperty('stroke', 'none', 'important'); p.style.setProperty('pointer-events', 'all', 'important'); });
  out.noStroke_peAll = [d(document.elementFromPoint(cx, cy)), d(document.elementFromPoint(cx + 8, cy))];
  paths.forEach(function(p){ p.style.setProperty('pointer-events', 'stroke', 'important'); });
  out.noStroke_peStroke = [d(document.elementFromPoint(cx, cy)), d(document.elementFromPoint(cx + 8, cy))];
  paths.forEach(function(p){ p.style.removeProperty('stroke'); p.style.removeProperty('pointer-events'); });
  out.restored = [d(document.elementFromPoint(cx, cy)), d(document.elementFromPoint(cx + 8, cy))];
  return JSON.stringify(out);
})()
"""

DRAG = r"""
(function(){
  var gd = document.querySelector('#time .js-plotly-plot'); var fl = gd._fullLayout, xa = fl.xaxis, ya = fl.yaxis, ya2 = fl.yaxis2, sh = fl.shapes;
  var hp = gd.querySelectorAll('[drag-helper] path')[2]; var hr = hp.getBoundingClientRect();
  var px = hr.left + hr.width / 2, y = hr.top + hr.height / 2;
  var target = document.elementFromPoint(px, y);
  function ev(type, t, cx, cy){ t.dispatchEvent(new MouseEvent(type, {bubbles: true, cancelable: true, clientX: cx, clientY: cy, screenX: cx, screenY: cy, button: 0, buttons: (type === 'mouseup' ? 0 : 1), view: window})); }
  ev('mousemove', target, px, y);
  ev('mousedown', target, px, y);
  for (var i = 1; i <= 6; i++) ev('mousemove', document, px + 10 * i, y);
  ev('mouseup', document, px + 60, y);
  return JSON.stringify({target: target && target.tagName, before: sh[0].x0});
})()
"""

AFTER = "JSON.stringify({start: document.getElementById('region-start').value, readout: document.getElementById('readout').innerText, x0: document.querySelector('#time .js-plotly-plot')._fullLayout.shapes[0].x0})"


def run(win):
    time.sleep(4)
    if SESSION:
        print("session:", win.evaluate_js("localStorage.setItem('last-session', %s); 'set'" % json.dumps(SESSION)))
        win.load_url(URL)
    n = 0
    for _ in range(90):
        time.sleep(1)
        try:
            n = win.evaluate_js("(function(){var g=document.querySelector('#time .js-plotly-plot'); return (g&&g._fullLayout&&g._fullLayout.shapes)?g._fullLayout.shapes.length:0})()")
        except Exception as e:
            n = 0
        if n and n >= 2:
            break
    print("shapes:", n)
    time.sleep(2)
    print("PROBE:", win.evaluate_js(PROBE))
    print("DRAG:", win.evaluate_js(DRAG))
    time.sleep(4)
    print("AFTER:", win.evaluate_js(AFTER))
    win.destroy()


if __name__ == "__main__":
    w = webview.create_window("wk test (closes itself)", URL, width=1440, height=900)
    webview.start(run, w)
    print("done")
