// Ctrl-Z / Shift-Ctrl-Z (and Ctrl-Y) → program-state undo/redo.
// Writes "undo:N" / "redo:N" into the hidden #undo-key input (N increments so repeats re-fire the
// Dash callback). Uses the native value setter + input event so Dash/React sees the change.
// While focused in a text field we DON'T hijack ctrl-z — the browser's own text undo runs instead.
(function () {
  var n = 0;
  function fire(action) {
    var el = document.getElementById('undo-key');
    if (!el) return;
    n += 1;
    var setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
    setter.call(el, action + ':' + n);
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }

  document.addEventListener('keydown', function (e) {
    if (!(e.ctrlKey || e.metaKey)) return;
    var t = e.target;
    var inField = t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable);
    var z = (e.key === 'z' || e.key === 'Z');
    var y = (e.key === 'y' || e.key === 'Y');
    if (!z && !y) return;
    if (inField) return;                                   // leave native text undo alone in fields
    e.preventDefault();
    if (y) { fire('redo'); return; }                       // ctrl-Y = redo
    fire(e.shiftKey ? 'redo' : 'undo');                    // ctrl-Z = undo, shift-ctrl-Z = redo
  });
})();
