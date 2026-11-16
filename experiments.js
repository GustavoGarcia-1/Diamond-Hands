const addRow = (id, values) => {
  const tr = document.createElement('tr');
  values.forEach(value => { const td = document.createElement('td'); td.textContent = value; tr.append(td); });
  document.getElementById(id).append(tr);
};
const number = x => Number(x).toFixed(3);
const held = x => x >= .9 ? 'Yes' : 'No';
fetch('experiments/results.json').then(r => { if (!r.ok) throw Error(r.status); return r.json(); }).then(d => {
  d.go.slice(0, 6).forEach(r => {
    const model = r.case.startsWith('original') ? '20 ms' : '2.5 ms';
    const delay = r.case.endsWith('30') ? '0.30 s' : r.case.endsWith('40') ? '0.40 s' : '0.50 s';
    addRow('go-timing', [model + ' / ' + delay, number(r.control_s), number(r.overlap_mm), r.final_hold ? 'Yes' : 'No']);
  });
  const labels = ['Lateral +2 cm', 'Lateral −2 cm', 'Vertical +2 cm', 'Vertical −2 cm'];
  d.throw_offsets.forEach((r, i) => addRow('offsets', [labels[i], number(r.hold), number(r.depth_mm), held(r.final)]));
  d.timesteps.forEach(r => addRow('timesteps', [number(Number(r.case.slice(3)) * 1000), number(r.hold), number(r.depth_mm), held(r.final)]));
  d.latency.forEach(r => addRow('latencies', [Math.round(Number(r.case.slice(8)) * 1000), number(r.hold), number(r.depth_mm), held(r.final)]));
  const names = {nominal:'Nominal',latency20:'20 ms delay',latency40:'40 ms delay',latency80:'80 ms delay',dropout10:'10% frame drops',survey:'5 mm / 0.25° survey error',combined:'Combined errors'};
  Object.entries(d.out_stress).forEach(([name, r]) => addRow('out-stress', [names[name] || name, number(r.zero.control_ms), number(r.model.control_ms), held(r.zero.endpoint) + ' / ' + held(r.model.endpoint)]));
}).catch(() => { document.getElementById('data-error').hidden = false; });
