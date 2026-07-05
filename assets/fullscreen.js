// Full-screen toggle for the Analysis Suite. In a normal browser it uses the Fullscreen API; in the
// native app window (pywebview) that API is a no-op, so we toggle the NATIVE window fullscreen via
// the JS API bridge (window.pywebview.api.toggle_fullscreen, wired in neitz_app.py). The button
// (#fs-toggle) and the "F" key both trigger it; event delegation covers Dash re-renders. Fullscreen
// must be requested from a user gesture — a click / keypress both qualify.
(function () {
  function inApp() {
    return !!(window.pywebview && window.pywebview.api && window.pywebview.api.toggle_fullscreen);
  }
  var pywFs = false;                                   // tracked fullscreen state for the native window

  function toggle() {
    if (inApp()) {                                     // native app window (WKWebView / WebView2)
      window.pywebview.api.toggle_fullscreen();
      pywFs = !pywFs;
      relabel();
      return;
    }
    if (document.fullscreenElement) {                  // browser tab
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
    if (!b) return;
    var fs = inApp() ? pywFs : !!document.fullscreenElement;
    b.textContent = fs ? '⛶ Exit full screen' : '⛶ Full screen';
  }
  document.addEventListener('fullscreenchange', relabel);
  document.addEventListener('webkitfullscreenchange', relabel);
  setInterval(relabel, 1500);
})();
