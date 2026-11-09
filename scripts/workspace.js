/* The forest and historical evaluation data are exported from the Python research pipeline. */
let evidence;
const $ = id => document.getElementById(id);
const percent = value => value == null ? '—' : `${(100 * value).toFixed(1)}%`;
const primary = {PAY_0: 'pay0', PAY_2: 'pay2', LIMIT_BAL: 'limit', AGE: 'age', PAY_AMT1: 'payamt'};
const note = message => { $('status').textContent = message; };

function setMenu(open) {
  $('workspace-sidebar').hidden = !open;
  document.querySelector('.app-frame').classList.toggle('sidebar-collapsed', !open);
  $('menu-toggle').setAttribute('aria-expanded', String(open));
  $('menu-toggle').querySelector('span').textContent = open ? 'Hide menu' : 'Show menu';
  // Plotly observes container dimensions asynchronously after the sidebar changes width.
  requestAnimationFrame(() => {
    const chart = document.querySelector('.view-panel.active .js-plotly-plot');
    if (chart && window.Plotly) Plotly.Plots.resize(chart);
  });
}

function showTab(id, button) {
  document.querySelectorAll('.view-panel').forEach(el => el.classList.toggle('active', el.id === `view-${id}`));
  document.querySelectorAll('.nav-item').forEach(el => {
    const active = el === button;
    el.classList.toggle('active', active);
    el.setAttribute('aria-current', active ? 'page' : 'false');
  });
  if (!evidence || !window.Plotly) return;
  if (id === 'overview') updateOverview();
  if (id === 'capacity') updateCapacity();
  if (id === 'roc') updateROC();
}

function probability(values) {
  const x = evidence.features.map((name, i) => Math.fround((values[name] - evidence.mean[i]) / evidence.scale[i]));
  let total = 0;
  for (const tree of evidence.trees) {
    let node = 0;
    while (tree.left[node] !== -1) node = x[tree.feature[node]] <= tree.threshold[node] ? tree.left[node] : tree.right[node];
    total += tree.probability[node];
  }
  return total / evidence.trees.length;
}

function calcRisk() {
  if (!evidence) return;
  const values = {...evidence.defaults};
  for (const [feature, id] of Object.entries(primary)) values[feature] = Number($(`inp-${id}`).value);
  document.querySelectorAll('[data-feature]').forEach(el => { values[el.dataset.feature] = Number(el.value); });
  const p = probability(values);
  for (const id of ['pay0', 'pay2']) {
    const value = Number($(`inp-${id}`).value);
    $(`txt-${id}`).textContent = value < 0 ? `Code ${value}` : value === 0 ? 'Code 0' : `Delay ${value} month${value === 1 ? '' : 's'}`;
  }
  $('txt-limit').textContent = `NT$ ${values.LIMIT_BAL.toLocaleString()}`;
  $('txt-age').textContent = `${values.AGE} years`;
  $('txt-payamt').textContent = `NT$ ${values.PAY_AMT1.toLocaleString()}`;
  $('disp-pd').textContent = percent(p);
  $('disp-el').textContent = `NT$ ${Math.round(p * 0.45 * values.LIMIT_BAL).toLocaleString()}`;
  $('disp-badge').textContent = p < .10 ? 'Illustrative: lower risk' : p < .25 ? 'Illustrative: medium risk' : 'Illustrative: higher risk';
}

const chartLayout = (x, y) => ({
  margin: {l: 58, r: 18, t: 20, b: 76}, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
  font: {family: '-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif', color: '#313a43', size: 12},
  xaxis: {title: {text: x}, tickformat: '.0%', range: [0, 1], gridcolor: '#e9eeef', zeroline: false},
  yaxis: {title: {text: y}, tickformat: '.0%', range: [0, 1], gridcolor: '#e9eeef', zeroline: false},
  legend: {orientation: 'h', y: -0.24},
});
const config = {responsive: true, displaylogo: false};
function plot(id, x, y, point, xlabel, ylabel, name) {
  Plotly.react(id, [
    {x, y, mode: 'lines', name, line: {color: '#2864b5', width: 3}},
    {x: [0,1], y: [0,1], mode: 'lines', name: 'Random', line: {color: '#9aaac0', dash: 'dash', width: 2}},
    {x: [point[0]], y: [point[1]], mode: 'markers', name: 'Selected', marker: {size: 12, color: '#203d6a'}}
  ], chartLayout(xlabel, ylabel), config);
}

function capacityAt(pct) {
  const slots = Math.round(evidence.outcomes.length * pct / 100);
  const captured = evidence.outcomes.slice(0, slots).reduce((a, b) => a + b, 0);
  const total = evidence.outcomes.reduce((a, b) => a + b, 0);
  return {slots, captured, recall: captured / total, precision: captured / slots, lift: (captured / slots) / (total / evidence.outcomes.length)};
}
function renderCapacity(id, pct) {
  const x = [0], y = [0], n = evidence.outcomes.length;
  const total = evidence.outcomes.reduce((a,b) => a+b, 0);
  let captured = 0;
  for (let i = 0; i < n; i++) { captured += evidence.outcomes[i]; if ((i+1) % 30 === 0 || i === n-1) {x.push((i+1)/n); y.push(captured/total);} }
  const selected = capacityAt(pct);
  plot(id, x, y, [selected.slots/n, selected.recall], 'Accounts reviewed', 'Defaults covered', 'Model ranking');
}
function updateOverview() {
  const pct = Number($('overview-capacity').value), d = capacityAt(pct);
  $('overview-capacity-label').textContent = `${pct}%`;
  $('overview-summary').textContent = `${d.slots.toLocaleString()} accounts reviewed · ${d.captured.toLocaleString()} observed defaults · ${percent(d.recall)} of all defaults`;
  renderCapacity('plot-overview-curve', pct);
}
function updateCapacity() {
  const pct = Number($('inp-cap').value), d = capacityAt(pct);
  $('cap-pct-txt').textContent = `${pct}%`;
  $('cap-slots-txt').textContent = `${d.slots.toLocaleString()} accounts`;
  $('cap-defaults').textContent = d.captured.toLocaleString();
  $('cap-recall').textContent = percent(d.recall);
  $('cap-prec').textContent = percent(d.precision);
  $('cap-lift').textContent = `${d.lift.toFixed(2)}×`;
  renderCapacity('plot-capacity-interactive', pct);
}
function updateROC() {
  const threshold = Number($('inp-roc-thresh').value);
  let tp=0, fp=0, positives=0, negatives=0;
  evidence.scores.forEach((score, i) => {
    const actual = evidence.outcomes[i];
    if (actual) positives++; else negatives++;
    if (score >= threshold) { if (actual) tp++; else fp++; }
  });
  $('txt-roc-thresh').textContent = threshold.toFixed(2);
  $('roc-disp-rec').textContent = percent(tp/positives);
  $('roc-disp-fpr').textContent = percent(fp/negatives);
  $('roc-disp-prec').textContent = percent(tp+fp ? tp/(tp+fp) : null);
  const x=[0], y=[0]; let t=0, f=0;
  for(let i=0; i<evidence.scores.length; i++) {
    if(evidence.outcomes[i]) t++; else f++;
    if(i === evidence.scores.length-1 || evidence.scores[i] !== evidence.scores[i+1]) {x.push(f/negatives); y.push(t/positives);}
  }
  plot('plot-roc-interactive', x, y, [fp/negatives, tp/positives], 'False positive rate', 'True positive rate', 'Random forest');
}

async function init() {
  $('menu-toggle').addEventListener('click', () => setMenu($('workspace-sidebar').hidden));
  try {
    const response = await fetch('data/web_model.json');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    evidence = await response.json();
    if (!window.Plotly) throw new Error('Chart library unavailable');
    const advanced = $('advanced-fields');
    evidence.features.filter(name => !(name in primary)).forEach(name => {
      const label = document.createElement('label'); label.className = 'advanced-field';
      label.textContent = name.replace('BILL_AMT', 'Bill amount · month ').replace('PAY_AMT', 'Payment amount · month ').replace('PAY_', 'Repayment status · code ');
      const input = document.createElement('input'); input.type='number'; input.step='1'; input.value=evidence.defaults[name]; input.dataset.feature=name;
      input.addEventListener('input', () => { if (input.value !== '' && Number.isFinite(Number(input.value))) calcRisk(); });
      label.append(input); advanced.append(label);
    });
    Object.values(primary).forEach(id => $(`inp-${id}`).addEventListener('input', calcRisk));
    $('overview-capacity').addEventListener('input', () => {
      $('inp-cap').value = $('overview-capacity').value;
      updateOverview();
    });
    $('inp-cap').addEventListener('input', () => {
      $('overview-capacity').value = $('inp-cap').value;
      updateCapacity();
    });
    $('inp-roc-thresh').addEventListener('input', updateROC);
    calcRisk(); updateOverview();
    note('Historical research · synthetic inputs · not a lending decision');
  } catch (error) {
    note('Data could not be loaded. Open this page through a web server and refresh.');
    $('disp-pd').textContent = 'Unavailable';
    console.error(error);
  }
}
window.addEventListener('DOMContentLoaded', init);
