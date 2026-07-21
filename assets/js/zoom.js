// Initialize medium-zoom (vanilla JS, no jQuery).
document.addEventListener('DOMContentLoaded', function () {
  window.medium_zoom = mediumZoom('[data-zoomable]', {
    margin: 100,
    background: getComputedStyle(document.documentElement)
        .getPropertyValue('--global-bg-color') + 'ee',  // + 'ee' for transparency.
  });
});
