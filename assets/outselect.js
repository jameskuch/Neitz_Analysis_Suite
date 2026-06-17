// Rubber-band (mouse-drag) multi-select for the Data Explorer output figures.
// Drag a window over the thumbnails in #exp-outputs-grid and every figure the box touches gets its
// select checkbox checked (so "Delete selected" / the delete modal act on the whole group). A plain
// click still enlarges a figure; only an actual drag selects.
(function () {
  function setup() {
    var grid = document.getElementById('exp-outputs-grid');
    if (!grid || grid._dragSelSetup) return;
    grid._dragSelSetup = true;

    grid.addEventListener('mousedown', function (e) {
      if (e.button !== 0) return;                                   // left button only
      if (e.target.closest('.out-check') || e.target.closest('button')) return;  // let those work
      var downX = e.clientX, downY = e.clientY, lastX = downX, lastY = downY, dragging = false, box = null;

      function move(ev) {
        lastX = ev.clientX; lastY = ev.clientY;
        if (!dragging && Math.abs(lastX - downX) + Math.abs(lastY - downY) > 5) {
          dragging = true;
          box = document.createElement('div');
          box.style.cssText = 'position:fixed;z-index:5000;pointer-events:none;border:1px solid ' +
            '#6dd2aa;background:rgba(110,210,170,0.15);';
          document.body.appendChild(box);
        }
        if (dragging) {
          box.style.left = Math.min(lastX, downX) + 'px'; box.style.top = Math.min(lastY, downY) + 'px';
          box.style.width = Math.abs(lastX - downX) + 'px';
          box.style.height = Math.abs(lastY - downY) + 'px';
        }
      }
      function up() {
        document.removeEventListener('mousemove', move);
        document.removeEventListener('mouseup', up);
        if (dragging) {
          var r = {left: Math.min(downX, lastX), top: Math.min(downY, lastY),    // from coords, robust
                   right: Math.max(downX, lastX), bottom: Math.max(downY, lastY)};
          grid.querySelectorAll('.out-check').forEach(function (cl) {
            var tile = cl.parentElement;                            // the figure tile (position:relative)
            var t = tile.getBoundingClientRect();
            var hit = !(t.right < r.left || t.left > r.right || t.bottom < r.top || t.top > r.bottom);
            if (hit) { var inp = cl.querySelector('input'); if (inp && !inp.checked) inp.click(); }
          });
          if (box) box.remove();
          // swallow the click that fires right after the drag (so it doesn't enlarge a figure)
          var sup = function (ce) { ce.stopPropagation(); ce.preventDefault(); };
          grid.addEventListener('click', sup, true);
          setTimeout(function () { grid.removeEventListener('click', sup, true); }, 0);
        }
      }
      document.addEventListener('mousemove', move);
      document.addEventListener('mouseup', up);
    });
  }
  if (document.readyState !== 'loading') setup();
  document.addEventListener('DOMContentLoaded', setup);
  setInterval(setup, 600);           // the detail pane re-renders — re-attach to the new grid node
})();
