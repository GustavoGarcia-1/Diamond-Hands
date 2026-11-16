const latestVideo = document.getElementById('latest-video');
const latestHuman = document.getElementById('latest-human-player');
const latestExample = document.getElementById('latest-example');
const latestRoutes = ['go', 'out', 'slant'];
const latestClips = {
  "go": {
    "label": "Walking Go with persistent finger encoder feedback",
    "evidence": "#results",
    "src": "overnight/go-regulated.mp4",
    "caption": "In this selected Go play, the robot keeps clean two-hand control for 5.663 seconds through the finish. That single play is not the 34-of-50 result: on a separate fresh 50-throw confirmation near six metres, persistent finger feedback retained 34 versus 2 for the original controller. This is a walking receive, not a running catch. The separate ten yard probe is available below.",
    "alternateLabel": "Show ten yard probe",
    "primaryLabel": "Show six metre example",
    "nominal": {
      "label": "Separate walking Go probe near ten yards",
      "evidence": "#results",
      "src": "overnight/go-tenyard.mp4",
      "caption": "This separate ten yard probe makes first contact at 9.970 yards of surveyed downfield depth. Its longest clean two-hand interval is 1.645 seconds, and the ball is held again at the finish. Both controllers pass two of seven fixed conditions here. This is a fixed development fixture, not the 50-throw confirmation; the 68% result does not apply."
    }
  },
  "out": {
    "label": "Walking Out with a left turn and upward facing hands",
    "evidence": "#results",
    "src": "experiments/out-selected-success.mp4",
    "caption": "In this selected trial (Out 06), the robot turns left near five yards and keeps clean two-hand control for 3.380 seconds through the finish. The original controller retains 9 of 50 development throws; this play is one example, not the distribution. The outside leg travels forward, and this is a walking receive.",
    "alternateLabel": "Show nominal failure",
    "primaryLabel": "Show retained example",
    "nominal": {
      "label": "Nominal Out with excluded wrist mount support",
      "evidence": "#results",
      "src": "experiments/out-comparison-nominal.mp4",
      "caption": "The unperturbed (nominal) Out keeps clean two-hand control for 2.321 seconds before wrist-mount support occurs. The form check passes, but retained possession fails. This full play preserves the failure."
    }
  },
  "slant": {
    "label": "Retained walking Slant with learned arms and a scripted late closing limit",
    "evidence": "#results",
    "src": "overnight/slant-retained.mp4",
    "caption": "Learned arm adjustments plus a scripted limit on further finger closure produce 1.995 seconds of clean two-hand control through the finish in this selected development play. That example is not the fresh result: on fifty fresh matched throws, the learned controller retained 12 versus 19 for scripted arms with the same closing limit, so it was not promoted. This is a walking receive in simulation.",
    "alternateLabel": "Show PPO without closing limit",
    "primaryLabel": "Show retained Slant",
    "nominal": {
      "label": "PPO Slant without the added closing limit, ending against excluded mounts",
      "evidence": "#results",
      "src": "overnight/slant-ppo.mp4",
      "caption": "Without the closing limit, PPO alone gives 873 milliseconds of clean two-hand control, then loses legitimate endpoint support and rests against mounts and wrists. Adding the scripted closing limit changes that ending in the retained example; the arm checkpoint is identical."
    }
  }
};
let latestRoute = 'go';
let latestView = 'robot';
let latestNominal = false;
let latestReferences = {};
function updateLatestExample() {
  latestExample.hidden = latestView === 'human' || !latestClips[latestRoute].nominal;
  latestExample.textContent = latestNominal ? latestClips[latestRoute].primaryLabel : latestClips[latestRoute].alternateLabel;
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
