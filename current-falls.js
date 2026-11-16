const fallVideo = document.getElementById('current-fall-video');
const fallDescriptions = {
  baseline: ['overnight/missed-catch-trip.mp4', 'The natural failed Go catch leads to a trip. The complete twelve-second baseline includes the landing.'],
  avoidance: ['overnight/missed-catch-avoidance.mp4', 'The same failed catch triggers a camera-based cancellation of the walking velocity command before the loose ball reaches the ankle. The robot stays upright; the catch still fails.']
};
for (const key of ['baseline', 'avoidance']) {
  document.getElementById('fall-'+key).addEventListener('click', () => {
    fallVideo.pause();
    fallVideo.src = fallDescriptions[key][0];
    fallVideo.setAttribute('aria-label', fallDescriptions[key][1]);
    fallVideo.load();
    document.getElementById('current-fall-caption').textContent = fallDescriptions[key][1];
    for (const other of ['baseline', 'avoidance']) document.getElementById('fall-'+other).setAttribute('aria-pressed', String(other === key));
  });
}
