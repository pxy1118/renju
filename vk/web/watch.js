// The read-only spectate page. It only ever GETs /api/watch with the watch
// cookie, and holds neither a session nor the CSRF token, so every play route
// stays closed to it: watching can never move a stone or open a table.
const $ = id => document.getElementById(id);
let tables = [], chosen = null, disconnected = false, signature = '';
const cells = buildBoard($('board'), null);
$('numbers').onchange = render;
$('table').onchange = () => { chosen = $('table').value; render(); };

function ruleName(rule) { return rule === 'renju' ? '连珠' : '自由五子棋'; }
function checkpointLabel(name) { return name === 'best' ? 'Champion / 预训练最佳' : name === 'latest' ? '最新模型' : '指定模型'; }
function showError(message) { $('error').textContent = message || ''; $('error').hidden = !message; }
function seatLabel(t) {
  const over = t.winner !== null ? ' · 已结束' : t.error ? ' · 出错' : t.busy ? ' · 思考中' : '';
  return `桌 ${t.sessions} · ${ruleName(t.model.rule)} · 第 ${t.history.length} 手${over}`;
}
// Follow the table the spectator picked. When it disappears (its guest left
// and the seats were renumbered), fall back to the first game on the service.
function pick() {
  if (chosen !== null && tables.some(t => String(t.sessions) === chosen)) return chosen;
  chosen = tables.length ? String(tables[0].sessions) : null;
  return chosen;
}
function render() {
  const number = pick(), t = tables.find(x => String(x.sessions) === number) || null;
  const options = tables.map(x => `${x.sessions}|${seatLabel(x)}`).join('|');
  if (options !== signature) {
    signature = options;
    const select = $('table');
    select.replaceChildren();
    if (!tables.length) select.appendChild(new Option('还没有对局', ''));
    for (const x of tables) select.appendChild(new Option(seatLabel(x), String(x.sessions)));
  }
  $('table').value = number ?? '';
  $('table-count').textContent = `${tables.length} 局`;
  const blank = {board: new Array(225).fill(0), history: []};
  const s = t || blank;
  paintStones(cells, s, t ? winningLine(t) : new Set(), $('numbers').checked);
  cells.forEach((b, i) => {
    const v = s.board[i];
    b.setAttribute('aria-disabled', 'true');
    b.setAttribute('aria-label', `${letters[i%15]}${Math.floor(i/15)+1}，${v === 1 ? '黑子' : v === -1 ? '白子' : '空位'}`);
  });
  const dot = $('turn-dot'), finished = !!t && t.winner !== null;
  const dotColor = finished ? t.winner : t?.player;
  dot.className = 'stone-icon' + (dotColor === 1 ? ' black' : dotColor === -1 ? ' white' : '') + (finished ? ' result' : '');
  dot.hidden = !t || (finished && t.winner === 0);
  dot.title = !t ? '' : finished ? (t.winner === 0 ? '和棋' : t.winner === 1 ? '黑方获胜' : '白方获胜')
                                 : (t.player === 1 ? '黑方落子' : '白方落子');
  $('status').textContent = !t ? '等待对局开始' : t.error ? '这一桌的模型运行遇到问题'
    : t.busy ? '模型思考中…' : finished ? (t.winner === 0 ? '本局和棋' : t.winner === 1 ? '黑方获胜' : '白方获胜')
    : (t.player === 1 ? '轮到黑方落子' : '轮到白方落子');
  $('count').textContent = `第 ${s.history.length} 手`;
  $('hint').textContent = !t ? '对局自动刷新；落子需要邀请链接。'
    : t.model.rule === 'renju' ? '连珠 · 黑方三三、四四、长连禁手' : '自由五子棋 · 连成五子或以上获胜';
  $('model-name').textContent = t ? `${ruleName(t.model.rule)} / ${checkpointLabel(t.model.checkpoint)}` : '尚未载入';
  $('model-detail').textContent = t ? (t.model.file ? `${t.model.file} · ${t.model.simulations} 次搜索${t.model.step != null ? ' · 训练 ' + t.model.step + ' 步' : ''}` : '正在读取权重…')
                                    : '对局开始后显示检查点信息';
  $('elapsed').textContent = t ? `${s.history.length} 手` + (!finished && t.seconds != null ? ` · 上手 ${t.seconds.toFixed(2)}s` : '') : '—';
  $('history-count').textContent = `${s.history.length} MOVES`;
  $('history').replaceChildren();
  if (!s.history.length) $('history').innerHTML = '<p class="muted">棋盘静候第一手。</p>';
  for (const [i, a] of s.history.entries()) {
    const item = document.createElement('span'); item.className = 'move';
    item.innerHTML = `<small>${i+1}</small><span class="stone-icon ${i%2 ? 'white' : 'black'}"></span>${letters[a%15]}${Math.floor(a/15)+1}`;
    $('history').appendChild(item);
  }
  if (t && t.error) showError(t.error);
  else if (!disconnected) showError('');
}
async function poll() {
  try {
    const response = await fetch('/api/watch', {signal: AbortSignal.timeout(15000)});
    const payload = await response.json();
    if (!response.ok) throw Error(payload.error || '请使用观战链接进入。');
    tables = payload.tables || [];
    disconnected = false;
  } catch (e) {
    disconnected = true;
    showError(e.message || '连接服务失败，请确认 Web UI 正在运行。');
  }
  render();
  const t = tables.find(x => String(x.sessions) === pick());
  // While the watched model thinks, poll fast enough to follow each reply.
  setTimeout(poll, !disconnected && t && t.busy ? 250 : 1000);
}
poll();
