// When a "trial align" box is focused, write its file path into the hidden #align-focus input so a
// clientside callback can highlight that file's trace in both graphs. On blur (with no sibling align
// box taking focus) write "" to clear the highlight. The align boxes are Dash pattern-matching
// components, so their DOM id attribute is a JSON string carrying {"path": ..., "type": ...}.
(function () {
  function pathOf(el) {
    try { return JSON.parse(el.id).path; } catch (e) { return null; }
  }
  function write(v) {
    var el = document.getElementById('align-focus');
    if (!el) return;
    var setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
    setter.call(el, v);
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }
  function isAlignBox(el) {
    return el && el.matches && el.matches('#align-editor input');
  }

  document.addEventListener('focusin', function (e) {
    if (isAlignBox(e.target)) {
      var p = pathOf(e.target);
      if (p) write(p);
    }
  });
  document.addEventListener('focusout', function (e) {
    if (!isAlignBox(e.target)) return;
    // let a sibling box's focusin win first; only clear if focus truly left the align editor
    setTimeout(function () {
      if (!isAlignBox(document.activeElement)) write('');
    }, 60);
  });
})();
