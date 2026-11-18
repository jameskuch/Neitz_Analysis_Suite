/* STRF map (Analysis View, lower-left #fft graph when a checkerboard is shown): clicks and the
 * double-click reach viewer.strf_pick through #strf-pick-sink (a hidden dcc.Input) rather than
 * Dash's clickData, because Plotly's event order defeats the obvious wiring:
 *   - a double-click in select drag-mode emits  click · deselect · click  — the second click would
 *     override the reset; and with nothing selected Dash's `selectedData = null` fires no callback;
 *   - the figure the callback returns makes Plotly re-emit an EMPTY `plotly_selected`, which looked
 *     like "clear the choice" (viewer.strf_checks_from_events now ignores empty selections).
 * So: a click is held for 350 ms (Plotly's double-click window) and written as "check|y|x|nonce"
 * unless a deselect lands first; a deselect writes "peak|nonce" and ignores the clicks around it.
 * Box selections still arrive as Dash selectedData. Re-hooked by a light poll (Dash may re-create
 * the graph div). */
(function () {
  var nonce = 0, pending = null, ignoreClicksUntil = 0;
  function setDashInput(id, value) {
    var el = document.getElementById(id);
    if (!el) return;
    var setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
    setter.call(el, value);
    el.dispatchEvent(new Event("input", { bubbles: true }));
  }
  function hook() {
    var host = document.getElementById("fft");
    var gd = host && host.querySelector(".js-plotly-plot");
    if (!gd || gd._strfPickHooked || typeof gd.on !== "function") return;
    gd._strfPickHooked = true;
    gd.on("plotly_click", function (d) {
      var p = d && d.points && d.points[0];
      if (!p || Date.now() < ignoreClicksUntil) return;
      if (pending) clearTimeout(pending);
      var y = Math.round(+p.y), x = Math.round(+p.x);
      pending = setTimeout(function () {
        pending = null;
        setDashInput("strf-pick-sink", "check|" + y + "|" + x + "|" + (nonce++));
      }, 350);
    });
    gd.on("plotly_deselect", function () {
      if (pending) { clearTimeout(pending); pending = null; }
      ignoreClicksUntil = Date.now() + 500;
      setDashInput("strf-pick-sink", "peak|" + (nonce++));
    });
  }
  setInterval(hook, 500);
})();
