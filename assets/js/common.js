// Publication list: toggle the abstract / bibtex / code panels.
// Vanilla JS (no jQuery). Each trigger reveals the matching hidden panel
// found a couple of levels up in the entry markup (see _layouts/bib.html).
document.addEventListener('DOMContentLoaded', function () {
  var toggles = [
    { selector: 'a.abstract', up: 2, target: '.abstract.hidden' },
    { selector: 'a.code',     up: 3, target: '.code.hidden' },
    { selector: 'a.bibtex',   up: 2, target: '.bibtex.hidden' },
  ];

  toggles.forEach(function (t) {
    document.querySelectorAll(t.selector).forEach(function (el) {
      el.addEventListener('click', function () {
        var container = this;
        for (var i = 0; i < t.up && container; i++) {
          container = container.parentElement;
        }
        if (!container) return;
        var panel = container.querySelector(t.target);
        if (panel) panel.classList.toggle('open');
      });
    });
  });
});
