// True full-screen toggle for the Analysis Suite (browser Fullscreen API).
// The button (#fs-toggle) is created by Dash; we use event delegation so it works no matter when
// Dash (re)renders it. Also bind the "F" key as a shortcut (ignored while typing in a field).
// Fullscreen MUST be requested from a user gesture — a click / keypress both qualify.
(function () {
  function toggle() {
    if (document.fullscreenElement) {
      (document.exitFullscreen || document.webkitExitFullscreen || function () {}).call(document);
    } else {
      var el = document.documentElement;
      (el.requestFullscreen || el.webkitRequestFullscreen || function () {}).call(el);
    }
  }

  document.addEventListener('click', function (e) {
    if (e.target && e.target.closest && e.target.closest('#fs-toggle')) toggle();
  });

  document.addEventListener('keydown', function (e) {
    if (e.key !== 'f' && e.key !== 'F') return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;          // leave ⌘F / ctrl-F alone
    var t = e.target;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
    e.preventDefault();
    toggle();
  });

  // keep the button label in sync with the actual state
  function relabel() {
    var b = document.getElementById('fs-toggle');
    if (b) b.textContent = document.fullscreenElement ? '⛶ Exit full screen' : '⛶ Full screen';
  }
  document.addEventListener('fullscreenchange', relabel);
  document.addEventListener('webkitfullscreenchange', relabel);
  setInterval(relabel, 1500);
})();
