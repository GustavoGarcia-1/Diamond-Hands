const byId = id => document.getElementById(id);
const video = byId('video');
const field = byId('field');
const ctx = field.getContext('2d');
const buttons = byId('plays');
const humanPlayer = byId('human-player');
const robotTimes = new Map();
const resumeOnRobot = new Map();
let selected;
let humanReferences = {};
let view = 'robot';
let pendingSeek = 0;

function routeName(play) {
  const route = play.route.toLowerCase();
  return route.includes('slant') ? 'slant' : route.includes('out') ? 'out' : 'go';
}

function drawRoute(at) {
  if (!selected) return;
  const { waypoints, motion, firstTouch } = selected;
  const width = field.width;
  const height = field.height;
  const positions = [...waypoints, ...motion.map(([, x, y]) => [x, y]), ...(selected.launcher ? [selected.launcher] : [])];
  const xs = positions.map(([x]) => x);
  const ys = positions.map(([, y]) => y);
  const bounds = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const scale = Math.min((height - 100) / Math.max(.20, bounds[1] - bounds[0]),
                         (width - 148) / Math.max(.20, bounds[3] - bounds[2]));
  const center = [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2];
  // Downfield is up; positive route normal is the receiver's left.
  const point = ([x, y]) => [width / 2 - (y - center[1]) * scale, height / 2 - (x - center[0]) * scale];
  ctx.fillStyle = '#fafafa';
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = '#777';
  ctx.font = '13px Helvetica, Arial, sans-serif';
  ctx.fillText('↑ Downfield     ← Left sideline', 18, 25);
  ctx.lineWidth = 1;
  ctx.strokeStyle = '#eee';
  for (let x = 30; x < width; x += 70) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke(); }
  for (let y = 25; y < height; y += 65) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke(); }

  function line(points, color, dashes) {
    if (points.length < 2) return;
    ctx.beginPath();
    ctx.setLineDash(dashes);
    ctx.lineWidth = 5;
    ctx.strokeStyle = color;
    ctx.moveTo(...point(points[0]));
    points.slice(1).forEach(p => ctx.lineTo(...point(p)));
    ctx.stroke();
    ctx.setLineDash([]);
  }

  function dot(p, color, radius) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(...point(p), radius, 0, Math.PI * 2);
    ctx.fill();
  }

  line(waypoints, '#bbb', [14, 10]);
  line(motion.map(([, x, y]) => [x, y]), '#888', []);
  const shown = motion.filter(([t]) => t <= at).map(([, x, y]) => [x, y]);
  line(shown, '#111', []);
  if (waypoints.length) {
    dot(waypoints[0], '#555', 7);
    ctx.strokeStyle = '#555';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(...point(waypoints.at(-1)), 12, 0, Math.PI * 2);
    ctx.stroke();
  }
  if (firstTouch) dot(firstTouch, '#b56f35', 9);
  if (selected.launcher) dot(selected.launcher, '#4b7e9c', 10);
  if (shown.length) dot(shown.at(-1), '#111', 8);
  const facing = selected.facing?.filter(([t]) => t <= at).at(-1);
  if (facing && shown.length) {
    const [px, py] = point(shown.at(-1));
    const dx = -Math.sin(facing[1]);
    const dy = -Math.cos(facing[1]);
    ctx.strokeStyle = '#555';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(px + 27 * dx, py + 27 * dy);
    ctx.lineTo(px + 18 * dx + 6 * dy, py + 18 * dy - 6 * dx);
    ctx.moveTo(px + 27 * dx, py + 27 * dy);
    ctx.lineTo(px + 18 * dx - 6 * dy, py + 18 * dy + 6 * dx);
    ctx.stroke();
  }
}

function stopHuman() {
  humanPlayer.src = 'about:blank';
  humanPlayer.hidden = true;
}

function showView() {
  if (!selected) return;
  const human = view === 'human';
  byId('robot-view').setAttribute('aria-pressed', String(!human));
  byId('human-view').setAttribute('aria-pressed', String(human));
  video.hidden = human;
  byId('human-stage').hidden = !human;
  byId('jump').hidden = human;
  const source = byId('human-source');
  source.hidden = !human;
  if (!human) {
    stopHuman();
    byId('human-unavailable').hidden = true;
    return;
  }
  const reference = humanReferences[routeName(selected)];
  const available = reference && /^[A-Za-z0-9_-]{11}$/.test(reference.videoId);
  byId('human-unavailable').hidden = Boolean(available);
  if (!available) {
    stopHuman();
    source.hidden = true;
    return;
  }
  humanPlayer.title = `${routeName(selected)} route · ${reference.sourceLabel}`;
  humanPlayer.src = `https://www.youtube-nocookie.com/embed/${reference.videoId}?rel=0`;
  humanPlayer.hidden = false;
  source.href = reference.sourceUrl;
  source.textContent = reference.sourceLabel;
  byId('human-detail').textContent = `${reference.note} Human and robot videos are independent references, not synchronized benchmarks.`;
}

function setView(next) {
  if (next === view || !selected) return;
  if (next === 'human') {
    robotTimes.set(selected.route, video.currentTime || 0);
    resumeOnRobot.set(selected.route, !video.paused);
    video.pause();
  }
  view = next;
  showView();
  if (next === 'robot' && resumeOnRobot.get(selected.route)) {
    resumeOnRobot.set(selected.route, false);
    video.play().catch(() => {});
  }
}

function select(play) {
  if (selected?.route === play.route) return;
  if (selected) {
    robotTimes.set(selected.route, video.currentTime || 0);
    video.pause();
  }
  stopHuman();
  selected = play;
  for (const link of buttons.children) {
    if (link.dataset.route === play.route) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  }
  byId('speed').textContent = play.speed == null ? 'No pass' : `${play.speed.toFixed(2)} m/s`;
  byId('distance-label').textContent = play.airTravel === undefined ? 'Travel after touch' : 'Airborne pass travel';
  byId('travel').textContent = play.airTravel == null ? 'No pass' : `${play.airTravel.toFixed(2)} m`;
  byId('control').textContent = play.hold ?? 'Not recorded';
  byId('retention').textContent = play.retention ?? 'Not recorded';
  byId('delivery').textContent = play.delivery ?? '';
  byId('evidence').href = `plays/${play.route}.json`;
  byId('jump').disabled = play.release == null;
  byId('trial-notes').textContent = play.notes ?? '';
  byId('facing-legend').hidden = !play.facing;
  byId('route-definition').textContent = play.definition ?? '';
  const human = humanReferences[routeName(play)];
  byId('human-detail').textContent = `${human?.note ?? ''} Human and robot videos are independent references, not synchronized benchmarks.`;
  video.setAttribute('aria-label', `${play.label} · recorded robot trial`);
  video.poster = `plays/${play.route}.webp`;
  pendingSeek = robotTimes.get(play.route) ?? 0;
  video.src = `plays/${play.route}.mp4`;
  video.load();
  drawRoute(pendingSeek);
  showView();
}

byId('robot-view').addEventListener('click', () => setView('robot'));
byId('human-view').addEventListener('click', () => setView('human'));
byId('jump').addEventListener('click', () => {
  if (!selected || selected.release == null || video.readyState === 0) return;
  video.currentTime = Math.max(0, selected.release - .45);
  video.play().catch(() => {});
});
video.addEventListener('loadedmetadata', () => {
  if (pendingSeek > 0 && Number.isFinite(video.duration)) {
    video.currentTime = Math.min(pendingSeek, Math.max(0, video.duration - .01));
  }
  pendingSeek = 0;
});
video.addEventListener('timeupdate', () => {
  if (!selected) return;
  robotTimes.set(selected.route, video.currentTime);
  drawRoute(video.currentTime);
});

Promise.all([
  fetch('plays/index.json').then(response => {
    if (!response.ok) throw new Error('Failed to load trials');
    return response.json();
  }),
  fetch('human-references.json').then(response => {
    if (!response.ok) throw new Error('Failed to load human references');
    return response.json();
  }).catch(() => ({}))
]).then(([plays, references]) => {
  humanReferences = references;
  for (const play of plays) {
    const link = document.createElement('a');
    link.href = `#${play.route}`;
    link.dataset.route = play.route;
    const route = routeName(play);
    link.textContent = route[0].toUpperCase() + route.slice(1);
    link.addEventListener('click', event => {
      event.preventDefault();
      history.replaceState(null, '', link.hash);
      select(play);
    });
    buttons.appendChild(link);
  }
  select(plays.find(play => play.route === location.hash.slice(1)) ??
         plays.find(play => routeName(play) === 'out') ?? plays[0]);
  window.addEventListener('hashchange', () => {
    const play = plays.find(play => play.route === location.hash.slice(1));
    if (play) select(play);
  });
}).catch(() => {
  byId('load-error').hidden = false;
  byId('load-error').textContent = 'The trials could not load. Please reload.';
});
