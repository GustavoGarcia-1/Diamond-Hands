const latestVideo = document.getElementById('latest-video');
const latestHuman = document.getElementById('latest-human-player');
const latestExample = document.getElementById('latest-example');
const latestRoutes = ['go', 'out', 'slant'];
const latestClips = {
  go: {
    label: 'Selected successful walking Go trial, with a bobble before secure control',
    evidence: '#results', src: 'experiments/go-selected-success.mp4',
    caption: 'In Go trial 02, the robot recovers from a bobble and maintains clean control with both hands for 3.537 seconds through the finish. The current controller retains the ball in 4 of 50 trials (8%) under this simulation model. The receiver is walking, and hardware transfer remains unverified.',
    nominal: {label: 'Nominal walking Go trial that ends with hand mount support', evidence: '#results', src: 'experiments/go-comparison-nominal.mp4', caption: 'The nominal Go trial maintains clean control with both hands for 0.558 seconds before the ball contacts the left wrist mount. This excluded support causes the retained catch check to fail.'}
  },
  out: {
    label: 'Selected successful walking Out trial with upward facing hands', evidence: '#results', src: 'experiments/out-selected-success.mp4',
    caption: 'In Out trial 06, the ball settles for 428 milliseconds before the robot maintains clean control with both hands for 3.380 seconds through the finish. The current controller retains the ball in 9 of 50 trials (18%) under this simulation model. The receiver is walking, and hardware transfer remains unverified.',
    nominal: {label: 'Nominal walking Out trial that ends with wrist mount support', evidence: '#results', src: 'experiments/out-comparison-nominal.mp4', caption: 'The nominal Out trial recovers from a bobble and maintains clean control with both hands for 2.321 seconds before wrist mount support occurs. The hand posture criterion passes, but retained possession fails.'}
  },
  slant: {
    label: 'Full Slant play with lower, upward facing hands and a missed catch', evidence: '#results', src: 'experiments/slant-lower-palms.mp4',
    caption: 'The Slant trial shows lower hands with the palms facing upward, followed by 22 milliseconds of bilateral contact and a drop. The current controller meets the hand posture criterion in all 50 trials, but none retains the ball.'
  }
};
let latestRoute = 'go';
let latestView = 'robot';
let latestNominal = false;
let latestReferences = {};
function updateLatestExample() {
  latestExample.hidden = latestView === 'human' || !latestClips[latestRoute].nominal;
  latestExample.textContent = latestNominal ? 'Show selected successful trial' : 'Show nominal trial';
}
function loadLatestClip() {
  const route = latestClips[latestRoute];
  const clip = latestNominal && route.nominal ? route.nominal : route;
  latestVideo.pause();
  latestVideo.src = clip.src;
  latestVideo.setAttribute('aria-label', clip.label);
  latestVideo.load();
  document.getElementById('latest-caption').textContent = clip.caption;
  document.getElementById('latest-evidence').href = clip.evidence;
  updateLatestExample();
}
function updateLatestView() {
  const human = latestView === 'human';
  latestVideo.hidden = human;
  latestHuman.hidden = !human;
  document.getElementById('latest-human-note').hidden = !human;
  document.getElementById('latest-caption').hidden = human;
  document.getElementById('latest-robot').setAttribute('aria-pressed', String(!human));
  document.getElementById('latest-human').setAttribute('aria-pressed', String(human));
  latestHuman.src = 'about:blank';
  if (human) {
    latestVideo.pause();
    const reference = latestReferences[latestRoute];
    if (reference && /^[A-Za-z0-9_-]{11}$/.test(reference.videoId)) {
      latestHuman.src = `https://www.youtube-nocookie.com/embed/${reference.videoId}?rel=0`;
      document.getElementById('latest-human-source').href = reference.sourceUrl;
      document.getElementById('latest-human-source').textContent = reference.sourceLabel || 'Watch original';
      document.getElementById('latest-human-context').textContent = reference.note || 'Independent instructional reference; not a matched comparison.';
    }
  }
  updateLatestExample();
}
for (const route of latestRoutes) {
  document.getElementById('latest-'+route).addEventListener('click', () => {
    if (route === latestRoute) return;
    latestRoute = route;
    latestNominal = false;
    loadLatestClip();
    for (const name of latestRoutes) document.getElementById('latest-'+name).setAttribute('aria-pressed', String(name === route));
    updateLatestView();
  });
}
latestExample.addEventListener('click', () => { latestNominal = !latestNominal; loadLatestClip(); });
for (const view of ['robot', 'human']) document.getElementById('latest-'+view).addEventListener('click', () => {
  latestView = view;
  updateLatestView();
});
fetch('human-references.json').then(r => { if (!r.ok) throw Error(r.status); return r.json(); })
  .then(data => { latestReferences = data; document.getElementById('latest-human').disabled = false; if (latestView === 'human') updateLatestView(); })
  .catch(() => { document.getElementById('latest-human').disabled = true; });
