// Navbar search: wires the search input to simple-jekyll-search (vanilla, no jQuery).
document.addEventListener('DOMContentLoaded', function () {
  var input = document.getElementById('search-input');
  var results = document.getElementById('search-results');
  if (!input || !results || typeof SimpleJekyllSearch === 'undefined') return;

  SimpleJekyllSearch({
    searchInput: input,
    resultsContainer: results,
    json: input.dataset.searchJson,
    searchResultTemplate:
      '<li><a href="{url}"><span class="search-title">{title}</span>' +
      '<span class="search-cat">{category}</span></a></li>',
    noResultsText: '<li class="search-empty">No results found</li>',
    limit: 8,
    fuzzy: false,
  });

  var clear = function () { results.innerHTML = ''; };

  // Dismiss results when clicking outside the search box.
  document.addEventListener('click', function (e) {
    if (!e.target.closest('.search-box')) clear();
  });

  // Escape clears and blurs.
  input.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') { input.value = ''; clear(); input.blur(); }
  });
});
