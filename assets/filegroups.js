// Fold / unfold a block of the Analysis-View files list in GROUP mode (#file-box.grouped).
// A block header is a disabled checklist option whose label holds a .grp-title line; clicking that
// line hides (class .grp-hidden) every following file row up to the next header and flips the arrow.
// The rows stay in the checklist, so the header's own all / none buttons and `file.value` are
// unaffected; re-grouping (options rebuilt) starts every block unfolded again.
(function () {
  function isHeader(lab) {
    var inp = lab.querySelector('input[type="checkbox"]');
    return !!(inp && inp.disabled);
  }
  function onClick(e) {
    var box = document.getElementById('file-box');
    if (!box || !box.classList.contains('grouped')) return;
    var title = e.target.closest('.grp-title');
    if (!title || !box.contains(title)) return;
    e.preventDefault(); e.stopPropagation();
    var header = title.closest('label');
    if (!header) return;
    var fold = !header.classList.contains('grp-folded');
    header.classList.toggle('grp-folded', fold);
    var arrow = title.querySelector('.grp-arrow');
    if (arrow) arrow.textContent = fold ? '▸ ' : '▾ ';
    var row = header.nextElementSibling;
    while (row && !isHeader(row)) {
      row.classList.toggle('grp-hidden', fold);
      row = row.nextElementSibling;
    }
  }
  document.addEventListener('click', onClick, true);   // capture: runs before the label toggles anything
})();
