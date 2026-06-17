// Resizable Data-Explorer rail columns. Each column width is a CSS variable (--rc1/2/3) on
// #exp-rail, shared by the header, search row and every data row. Drag a handle to resize the
// column to its LEFT; the dead column absorbs the change. Widths persist in localStorage.
(function () {
  var KEY = "neitz_rail_cols";
  function rail() { return document.getElementById("exp-rail"); }
  function applySaved() {
    var r = rail(); if (!r) return;
    var saved = {};
    try { saved = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
    ["1", "2", "3"].forEach(function (c) {
      if (saved["rc" + c]) r.style.setProperty("--rc" + c, saved["rc" + c]);
    });
  }
  function init() {
    var r = rail();
    var handles = document.querySelectorAll(".rail-rz");
    if (!r || !handles.length) { return setTimeout(init, 300); }   // wait for Dash to render
    applySaved();
    handles.forEach(function (h) {
      if (h._wired) return; h._wired = true;
      h.addEventListener("mousedown", function (e) {
        e.preventDefault(); e.stopPropagation();
        var col = h.getAttribute("data-col");
        var startX = e.clientX;
        var startW = h.parentElement.getBoundingClientRect().width;
        document.body.style.cursor = "col-resize";
        document.body.style.userSelect = "none";
        function mm(ev) {
          // getBoundingClientRect & clientX are in RENDERED px; the CSS var is unzoomed px,
          // so divide by the page zoom (responsive scaling) to keep the drag 1:1 with the mouse.
          var zoom = parseFloat(getComputedStyle(document.body).zoom) || 1;
          var w = Math.max(20, (startW + (ev.clientX - startX)) / zoom);
          r.style.setProperty("--rc" + col, w + "px");
        }
        function mu() {
          document.removeEventListener("mousemove", mm);
          document.removeEventListener("mouseup", mu);
          document.body.style.cursor = "";
          document.body.style.userSelect = "";
          var saved = {};
          try { saved = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
          saved["rc" + col] = r.style.getPropertyValue("--rc" + col).trim();
          localStorage.setItem(KEY, JSON.stringify(saved));
        }
        document.addEventListener("mousemove", mm);
        document.addEventListener("mouseup", mu);
      });
    });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
