'use strict';

const studyVideo = document.getElementById('video');
const studyResult = document.getElementById('result');
const studyCases = {
  baseline: { label: 'Baseline', source: 'plays/native-baseline.mp4', result: '1.457 s continuous bilateral control · later reacquired' },
  learned: { label: 'Learned', source: 'plays/native-learned.mp4', result: '3.229 s continuous bilateral control · held at finish' }
};

for (const [key, trial] of Object.entries(studyCases)) {
  document.getElementById(key).addEventListener('click', () => {
    studyVideo.pause();
    studyVideo.src = trial.source;
    studyVideo.setAttribute('aria-label', `${trial.label} lower hand catch trial`);
    studyVideo.load();
    studyResult.textContent = trial.result;
    for (const other of Object.keys(studyCases)) {
      document.getElementById(other).setAttribute('aria-pressed', String(other === key));
    }
  });
}

studyVideo.addEventListener('error', () => {
  studyResult.textContent = 'The video could not load. Reload the page or select the other controller.';
});
