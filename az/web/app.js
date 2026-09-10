const $ = id => document.getElementById(id);
let state = null, token = '', config = {models:{}, recent:{}, ready:false}, selection = 'latest', color = 'black';
let pending = false, revision = 0, disconnected = false, lastMoveSeconds = null;
const letters = 'ABCDEFGHIJKLMNO';
const cells = [];
for (let i = 0; i < 225; i++) {
  const r = Math.floor(i / 15), c = i % 15, b = document.createElement('button');
  b.className = 'point' + (r === 0 ? ' top' : '') + (r === 14 ? ' bottom' : '') + (c === 0 ? ' left' : '') + (c === 14 ? ' right' : '') + ([48,56,112,168,176].includes(i) ? ' star' : '');
  b.innerHTML = `<span class="piece"></span>${r === 0 ? `<span class="coord col">${letters[c]}</span>` : ''}${c === 0 ? `<span class="coord row">${r+1}</span>` : ''}`;
  b.setAttribute('aria-label', `${letters[c]}${r+1}，空位`);
  b.addEventListener('click', () => {
    if (!state || pending || state.busy || state.error || state.winner !== null || state.player !== state.human) return;
    if (!state.legal.includes(i)) { showError('这里不能落子：已占用或不符合当前规则。'); return; }
    mutate('/api/move', {id: state.id, action: i});
  });
  cells.push(b); $('board').appendChild(b);
}
function showError(message) { $('error').textContent = message || ''; $('error').hidden = !message; }
function ruleName(rule) { return rule === 'renju' ? '连珠' : '自由五子棋'; }
async function api(path, data) {
  const init = data === undefined ? {} : {method:'POST', headers:{'Content-Type':'application/json','X-Renju-Token':token}, body:JSON.stringify(data)};
  // Without a timeout a half-open local socket would stall the polling loop.
  const response = await fetch(path, {...init, signal: AbortSignal.timeout(15000)});
  const result = await response.json();
  if (!response.ok) throw Error(result.error || '请求失败');
  return result;
}
function availability() {
  const rule = $('rule').value, has = !!checkpointFile(rule, selection);
  showCheckpoints(rule);
  if (!config.ready) $('availability').textContent = '正在读取模型列表…';
  else if (has) $('availability').textContent = selection === 'latest' ? '已找到检查点 · 可开始对弈'
    : selection === 'best' ? '已找到历史最佳 · 可开始对弈' : '已选定历史检查点 · 可开始对弈';
  else $('availability').textContent = `暂无${ruleName(rule)}模型，等待训练保存检查点${config.recent[rule]?.length ? '（已列出更早的检查点）' : ''}。训练每轮结束时写入，完成后刷新本页。`;
  // Never disable this button: clicking it explains why a rule is unavailable.
  $('new').disabled = !!pending || !!state?.busy;
  $('new').innerHTML = !config.ready ? '开始对弈 <span>↗</span>'
    : has ? (state?.id ? '重新开始 <span>↗</span>' : '开始对弈 <span>↗</span>')
          : '暂无可用模型 <span>↗</span>';
}
// A selection is usable when it is an alias the server resolves or a file it listed.
function checkpointFile(rule, wanted) {
  const entry = config.models[rule];
  if (!entry) return null;
  if (wanted === 'latest' || wanted === 'best') return entry[wanted];
  const listed = (config.recent[rule] || []).some(item => item.name === wanted) || entry.latest === wanted;
  return listed ? wanted : null;
}
let checkpointSignature = '';
function showCheckpoints(rule) {
  const entry = config.models[rule] || {};
  const options = [{value:'latest', label: entry.latest ? `最新检查点 · ${entry.latest}` : '最新检查点（暂无）'},
                   {value:'best', label: entry.best ? `历史最佳 · ${entry.best}` : '历史最佳（暂无）'}];
  for (const item of config.recent[rule] || []) {
    if (item.name !== entry.latest) options.push({value:item.name, label:`${item.name.replace('checkpoint-','#').replace('.pt','')} · ${item.mtime} · ${item.mb}MB`});
  }
  const signature = rule + '|' + options.map(o => o.value + o.label).join('|');
  const select = $('checkpoint');
  if (signature !== checkpointSignature) {
    // Rebuilding only on change keeps an open dropdown from closing under the pointer.
    checkpointSignature = signature;
    select.replaceChildren();
    for (const option of options) select.appendChild(new Option(option.label, option.value));
  }
  select.value = options.some(o => o.value === selection) ? selection : 'latest';
  if (select.value !== selection) {
    // The selected file was pruned by training; fall back to latest instead of failing later.
    selection = select.value;
  }
  const latest = entry.latest, shown = (config.recent[rule] || []).find(item => item.name === latest);
  $('checkpoint-detail').textContent = latest
    ? `最新：${latest}${shown ? ` · ${shown.mtime} · ${shown.mb}MB` : ''} · 共 ${(config.recent[rule] || []).length} 个可选`
    : '该规则下还没有可选检查点';
}
// The five stones that ended the game, so the result is verifiable on the board.
function winningLine(state) {
  if (state.winner === 0 || state.winner === null || !state.history.length) return new Set();
  const last = state.history.at(-1), color = state.winner;
  if (state.board[last] !== color) return new Set();
  for (const [dr, dc] of [[0,1],[1,0],[1,1],[1,-1]]) {
    const line = [last];
    for (const sign of [-1, 1]) {
      let r = Math.floor(last / 15) + sign * dr, c = last % 15 + sign * dc;
      while (r >= 0 && r < 15 && c >= 0 && c < 15 && state.board[r * 15 + c] === color) {
        line.push(r * 15 + c);
        r += sign * dr; c += sign * dc;
      }
    }
    if (line.length >= 5) return new Set(line);
  }
  return new Set();
}
function render() {
  if (!state) return;
  const s = state, busy = pending || s.busy, active = !!s.id;
  const moveNumbers = new Map(s.history.map((a, i) => [a, i+1]));
  const winners = winningLine(s);
  cells.forEach((b, i) => {
    const v = s.board[i], p = b.querySelector('.piece');
    p.className = 'piece' + (v ? (v === 1 ? ' black' : ' white') : '') + (s.history.at(-1) === i ? ' last' : '');
    p.classList.toggle('win', winners.has(i));
    p.textContent = v && $('numbers').checked ? moveNumbers.get(i) : '';
    const legal = active && !busy && !s.error && s.winner === null && s.player === s.human && s.legal.includes(i);
    b.classList.toggle('legal', legal);
    b.setAttribute('aria-disabled', String(!legal));
    b.setAttribute('aria-label', `${letters[i%15]}${Math.floor(i/15)+1}，${v === 1 ? '黑子' : v === -1 ? '白子' : legal ? '可落子' : '不可落子'}`);
  });
  // The icon must never contradict the text: when the game is over it shows the
  // winner's stone instead of the side that would move next.
  const dot = $('turn-dot'), finished = active && s.winner !== null;
  const dotColor = finished ? s.winner : s.player;
  dot.className = 'stone-icon' + (dotColor === 1 ? ' black' : dotColor === -1 ? ' white' : '')
                  + (finished ? ' result' : '');
  dot.hidden = finished && s.winner === 0;
  dot.setAttribute('aria-hidden', 'true');
  dot.title = finished ? (s.winner === 0 ? '和棋' : s.winner === 1 ? '黑方获胜' : '白方获胜')
                       : (s.player === 1 ? '黑方落子' : '白方落子');
  $('status').textContent = s.error ? '模型运行遇到问题' : !active ? '准备开始一局' : busy ? '模型思考中…' : s.winner !== null ? (s.winner === 0 ? '本局和棋' : s.winner === s.human ? '你赢了，这一局下得漂亮。' : '模型获胜，再来一局？') : '轮到你了 · ' + (s.human === 1 ? '黑方落子' : '白方落子');
  $('count').textContent = `第 ${s.history.length} 手`;
  $('hint').textContent = !active ? '选择右侧设置，开始与模型对弈' : busy ? '正在本地搜索，请稍候…' : s.model.rule === 'renju' ? '黑首子天元 · 黑方三三、四四、长连禁手' : '点击交叉点落子 · 连成五子或以上获胜';
  $('undo').disabled = !active || busy || !!s.error || !s.history.some((_,i) => (i%2 === 0 ? 1 : -1) === s.human);
  $('model-name').textContent = active ? `${ruleName(s.model.rule)} / ${s.model.checkpoint === 'best' ? '历史最佳' : '指定检查点'}` : '尚未载入';
  $('model-detail').textContent = s.model.file ? `${s.model.file} · ${s.model.simulations} 次搜索${s.model.step != null ? ' · 训练 '+s.model.step+' 步' : ''}` : active ? '正在读取权重…' : '开始对弈后显示检查点信息';
  $('elapsed').textContent = active ? `${s.history.length} 手` + (!finished && lastMoveSeconds != null ? ` · 上手 ${lastMoveSeconds.toFixed(2)}s` : '') : '—';
  $('history-count').textContent = `${s.history.length} MOVES`;
  $('history').replaceChildren();
  if (!s.history.length) $('history').innerHTML = '<p class="muted">棋盘静候第一手。</p>';
  for (const [i,a] of s.history.entries()) {
    const item = document.createElement('span'); item.className = 'move';
    item.innerHTML = `<small>${i+1}</small><span class="stone-icon ${i%2 ? 'white' : 'black'}"></span>${letters[a%15]}${Math.floor(a/15)+1}`;
    $('history').appendChild(item);
  }
  if (s.error) showError(s.error);
  availability();
}
async function mutate(path, data) {
  pending = true; revision++; showError(''); render();
  try { state = await api(path, data); }
  catch (e) { showError(e.message); }
  finally { pending = false; render(); }
}
$('new').onclick = () => {
  if (!checkpointFile($('rule').value, selection)) {
    return showError(`当前规则还没有可用的检查点。${$('rule').value === 'renju' ? '连珠' : '自由五子棋'}模型需要先由训练保存：每轮结束时写入 runs/ 目录，完成后刷新本页即可选择。`);
  }
  mutate('/api/new', {rule:$('rule').value, color, checkpoint:selection, simulations:Number($('simulations').value)});
};
$('undo').onclick = () => mutate('/api/undo', {id:state.id});
$('numbers').onchange = render;
document.querySelectorAll('.color').forEach(b => b.onclick = () => {
  color = b.dataset.color;
  document.querySelectorAll('.color').forEach(x => {x.classList.toggle('active', x === b); x.setAttribute('aria-pressed', String(x === b));});
});
$('rule').onchange = availability;
$('checkpoint').onchange = () => { selection = $('checkpoint').value; availability(); };
let pollCount = 0;
async function poll() {
  try {
    if (!pending) {
      const currentRevision = revision;
      // Refreshing the checkpoint list every 10 s lets a finished round appear without reloading.
      if (!token || pollCount++ % (document.hidden ? 60 : 10) === 0) {
        config = await api('/api/config');
        token = config.token; config.ready = true; availability();
      }
      const next = await api('/api/state');
      if (!pending && currentRevision === revision) {
        if (disconnected) { disconnected = false; if (!next.error) showError(''); }
        lastMoveSeconds = next.seconds !== null && next.seconds !== state?.seconds ? next.seconds : lastMoveSeconds;
        state = next; render();
      }
    }
  } catch (e) { disconnected = true; showError('连接本地服务失败，请确认 Web UI 正在运行。'); }
  // While the model thinks, poll fast enough that the reply is not rounded up to a second.
  setTimeout(poll, state?.busy || pending ? 250 : 1000);
}
$('new').disabled = true;
poll();
