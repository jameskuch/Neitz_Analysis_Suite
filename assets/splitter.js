// Draggable sidebar | graphs divider. The width is stored as a FRACTION of the window
// width (localStorage), so a layout saved on a big monitor still looks right on a small
// one. Dash auto-loads any .js in this assets/ folder.
(function () {
  var KEY = "neitz_sidebar_frac";
  function clamp(f) { return Math.max(0.10, Math.min(0.50, f)); }
  function apply(frac) {
    var sb = document.getElementById("sidebar");
    if (!sb) return;
    var pct = (frac * 100).toFixed(2) + "%";
    sb.style.flex = "0 0 " + pct;
    sb.style.maxWidth = pct;
  }
  function init() {
    var sb = document.getElementById("sidebar");
    var sp = document.getElementById("splitter");
    if (!sb || !sp) { return setTimeout(init, 250); }   // wait for Dash to render
    var saved = parseFloat(localStorage.getItem(KEY));
    if (!isNaN(saved)) apply(clamp(saved));
    var dragging = false;
    sp.addEventListener("mousedown", function (e) {
      dragging = true; e.preventDefault();
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
    });
    window.addEventListener("mousemove", function (e) {
      if (!dragging) return;
      var origin = sb.getBoundingClientRect().left;
      apply(clamp((e.clientX - origin) / window.innerWidth));
    });
    window.addEventListener("mouseup", function () {
      if (!dragging) return;
      dragging = false;
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      var frac = sb.getBoundingClientRect().width / window.innerWidth;
      localStorage.setItem(KEY, clamp(frac).toFixed(4));
      window.dispatchEvent(new Event("resize"));   // let the Plotly graphs refit
    });
  }
  if (document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", init);
  else init();
})();
