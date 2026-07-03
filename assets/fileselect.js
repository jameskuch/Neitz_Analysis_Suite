// Rubber-band (mouse-drag) TOGGLE-select for the Analysis-View files list (#file-box).
// Drag a box over the file rows and every checkbox the box touches is TOGGLED
// (selected↔deselected). A plain click still toggles a single row natively; only an actual
// drag (> a few px) selects a range. Mirrors assets/outselect.js (the output-figure selector).
(function () {
  function setup() {
    var box = document.getElementById('file-box');
    if (!box || box._dragToggleSetup) return;
    box._dragToggleSetup = true;

    box.addEventListener('mousedown', function (e) {
      if (e.button !== 0) return;                                    // left button only
      var downX = e.clientX, downY = e.clientY, lastX = downX, lastY = downY;
      var dragging = false, rubber = null;

      function move(ev) {
        lastX = ev.clientX; lastY = ev.clientY;
        if (!dragging && Math.abs(lastX - downX) + Math.abs(lastY - downY) > 5) {
          dragging = true;
          rubber = document.createElement('div');
          rubber.style.cssText = 'position:fixed;z-index:5000;pointer-events:none;border:1px solid ' +
            '#3367d6;background:rgba(80,130,240,0.15);';
          document.body.appendChild(rubber);
          e.preventDefault();
        }
        if (dragging) {
          rubber.style.left = Math.min(lastX, downX) + 'px';
          rubber.style.top = Math.min(lastY, downY) + 'px';
          rubber.style.width = Math.abs(lastX - downX) + 'px';
          rubber.style.height = Math.abs(lastY - downY) + 'px';
        }
      }
      function up() {
        document.removeEventListener('mousemove', move);
        document.removeEventListener('mouseup', up);
        if (dragging) {
          var r = {left: Math.min(downX, lastX), top: Math.min(downY, lastY),
                   right: Math.max(downX, lastX), bottom: Math.max(downY, lastY)};
          box.querySelectorAll('input[type="checkbox"]').forEach(function (inp) {
            var row = inp.closest('label') || inp.parentElement;
            var t = row.getBoundingClientRect();
            var hit = !(t.right < r.left || t.left > r.right || t.bottom < r.top || t.top > r.bottom);
            if (hit) inp.click();                                    // toggle each touched row
          });
          if (rubber) rubber.remove();
          // swallow the click that fires right after the drag (so it doesn't re-toggle one row)
          var sup = function (ce) { ce.stopPropagation(); ce.preventDefault(); };
          box.addEventListener('click', sup, true);
          setTimeout(function () { box.removeEventListener('click', sup, true); }, 0);
        }
      }
      document.addEventListener('mousemove', move);
      document.addEventListener('mouseup', up);
    });
  }
  if (document.readyState !== 'loading') setup();
  document.addEventListener('DOMContentLoaded', setup);
  setInterval(setup, 800);          // the checklist re-renders on cell change — re-attach
})();
