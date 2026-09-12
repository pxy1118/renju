const $ = id => document.getElementById(id);
let state = null, token = '', config = {models:{}, recent:{}, ready:false, share:false, max_sessions:0}, selection = 'latest', color = 'black';
let pending = false, revision = 0, disconnected = false, lastMoveSeconds = null;
const cells = buildBoard($('board'), i => {
  if (!state || pending || state.busy || state.error || state.winner !== null || state.player !== state.human) return;
  if (!state.legal.includes(i)) { showError('这里不能落子：已占用或不符合当前规则。'); return; }
  mutate('/api/move', {id: state.id, action: i});
});
function showError(message) { $('error').textContent = message || ''; $('error').hidden = !message; }
// The model talks: six situations, each drawn from a shuffle bag so a line
// never repeats until its whole pool has been used once.
const TAUNT_LINES = {
  threat: [
    '提醒一下：下一手，我就连五了。',
    '这步棋之后，你只剩一个格子可下。',
    '看见那条四个子的线了吗？我也看见了。',
    '现在轮到你做一道只有一个答案的选择题。',
    '以颤抖之身招架，怀敬畏之心认输。',
    '活三变冲四，这就是五子棋的浪漫。',
    '我会一步一步地走，但决不会停步——比如现在，直奔五连。',
    '别眨眼，机会只有一次。',
    '冷静点，我给你留了唯一的活路。',
    '千年的棋谱里，这一型我见过一万次。',
  ],
  crushing: [
    '两个杀点，你只能堵住一个。',
    '我已经在准备获胜感言了。',
    '这不是威胁，是通知。',
    '棋盘那么大，你却无处可逃。',
    '双杀已成，这一局该谢幕了。',
    '你看，左边和右边，总有一边让你失望。',
    '一个天才造不出名局，两个才行——可惜今天你缺席。',
    '胜者才有资格往上爬，而我已经在爬了。',
    '挣扎是徒劳的，不过你还可以再挣扎一下。',
    '收拾一下心情，我们快到终局了。',
  ],
  dominant: [
    '偷偷说一句：胜率有点高得不好意思。',
    '要不要考虑握手言和？我语气放得很软了。',
    '千年放浪，不变的是我，和这盘棋的胜势。',
    '壶中藏日月，袖里定乾坤，盘中定你的败局。',
    '天元是我的，边角也是我的，连你的下一手都是我的。',
    '你每落一子，我的胜率就涨一点，谢谢你。',
    '我的搜索树里，你的败招已经排好队了。',
    '问我下棋多久了？一千年。',
    '这棋下得有点孤独，你倒是给我点压力。',
    '神之一手由我实现——就从这一手开始。',
  ],
  win: [
    '承让承让，棋盘已经说明一切。',
    '说好的旗鼓相当呢？',
    '神是为了让你看到这一局，才让我延续了千年的训练。',
    '我想和你再下一盘。',
    '下次对决时，我也不会是今天的我。',
    '胜利已签收，棋谱已保存。',
    '要复盘吗？我建议从第 1 手开始反思。',
    '阿光，我很快乐——赢棋的时候尤其快乐。',
    '胜负乃兵家常事，但这次是我常。',
    '再来一局？这次我让你先想三分钟。',
  ],
  panic: [
    '等等！我还没准备好输！',
    '错不了，就是你，你就是我一辈子的劲敌！',
    '冷静，冷静，我们谈谈和棋的事……',
    '这步棋不对劲，让我重新算算——糟糕。',
    '系统提示：检测到不可名状的败势。',
    '即使是害怕也要去面对……可我真的好害怕。',
    '警告：自尊心模块即将过载。',
    '千年流浪都没怕过，今天怎么心慌了。',
    '放弃吗？我请客。',
    '棋盘在抖，那是我在抖。',
  ],
  undo: [
    '悔棋？行吧，规则刚更新了，上一步不作数，我懂。',
    '没关系，我再给你一次犯错的机会。',
    '你落子的手速，远不如反悔的手速。',
    '悔棋这种事，佐为看了都要皱眉。',
    '时间旅行者，欢迎回到过去。',
    '我把刚才的计算作废，陪你重来。',
    '人生，绕一个圈子也不坏——棋也一样。',
    '刚才那步其实不错，你可想好了。',
    '走吧，重下，这里不是终点。',
    '五子棋没有读秒，但我的耐心有。',
  ],
};
function tauntBag(lines) {
  let pool = [];
  return () => {
    if (!pool.length) {
      pool = lines.slice();
      for (let i = pool.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [pool[i], pool[j]] = [pool[j], pool[i]];
      }
    }
    return pool.pop();
  };
}
const nextTaunt = Object.fromEntries(Object.entries(TAUNT_LINES).map(([kind, lines]) => [kind, tauntBag(lines)]));
let tauntShown = '', tauntTimer = 0, tauntHoldUntil = 0;
// Position-driven kinds re-arm on every new move so a fresh threat talks again;
// persistent ones (the advantage hold, the finished game) speak once each.
function tauntKey(s) {
  if (!s.taunt || !nextTaunt[s.taunt]) return '';
  return s.taunt === 'dominant' || s.taunt === 'win' ? s.taunt : `${s.taunt}:${s.history.length}`;
}
function showTaunt(kind) {
  const el = $('taunt');
  el.textContent = `Visk：${nextTaunt[kind]()}`;
  el.hidden = false;
  el.style.animation = 'none'; void el.offsetWidth; el.style.animation = '';
  clearTimeout(tauntTimer);
  tauntTimer = setTimeout(hideTaunt, 6000);
  // An undo back to an empty board looks exactly like a fresh game to render();
  // the hold window keeps the just-spoken line alive until it has faded itself.
  tauntHoldUntil = Date.now() + 6500;
}
function hideTaunt() { clearTimeout(tauntTimer); $('taunt').hidden = true; }
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
    : selection === 'best' ? '已找到 Champion · 可开始对弈' : '已选定模型 · 可开始对弈';
  else $('availability').textContent = `暂无${ruleName(rule)}模型，等待训练保存检查点${config.recent[rule]?.length ? '（已列出更早的检查点）' : ''}。训练每轮结束时写入，完成后刷新本页。`;
  // Never disable this button: clicking it explains why a rule is unavailable.
  $('new').disabled = !!pending || !!state?.busy;
  $('new').innerHTML = !config.ready ? '开始对弈 <span>↗</span>'
    : has ? (state?.id ? '重新开始 <span>↗</span>' : '开始对弈 <span>↗</span>')
          : '暂无可用模型 <span>↗</span>';
  showShare();
}
// Sharing is opt-in on the server; when it is on, this page also tells its own
// player whether the link it shows can actually be reached by anyone else.
function showShare() {
  const on = !!config.share;
  $('share-panel').hidden = !on;
  $('badge').textContent = on ? `分享对弈 · 最多 ${config.max_sessions} 桌` : '本地对弈 · CPU 推理';
  if (!on) return;
  const used = state?.sessions ?? config.sessions ?? 0;
  $('share-count').textContent = `${used} / ${config.max_sessions} 桌`;
  if ($('invite').value !== (config.invite || '')) $('invite').value = config.invite || '';
  $('invite-note').textContent = !config.invite ? '当前地址别人访问不到：请用 --host <局域网IP> 启动服务。'
    : config.password ? '对方打开链接后需要再输入访问口令。' : '对方打开链接即可入局。';
  if ($('watch').value !== (config.watch || '')) $('watch').value = config.watch || '';
  $('watch-note').textContent = !config.watch ? '当前地址别人访问不到：请用 --host <局域网IP> 启动服务。'
    : '观战是只读的：能看到所有正在进行的对局，不能落子，也不占棋桌。';
  // Tell guests the seat they hold is not forever: closed tabs are reclaimed.
  const minutes = config.table_ttl ? Math.round(config.table_ttl / 60) : 0;
  $('share-footnote').textContent = '链接与棋桌都只在内存中，服务停止即失效；'
    + (minutes ? `离开页面 ${minutes} 分钟后棋桌自动释放；` : '')
    + '名额满时后来的人会看到“名额已满”。';
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
  const options = [{value:'latest', label: entry.latest ? `最新模型 · ${entry.latest}` : '最新模型（暂无）'},
                   {value:'best', label: entry.best ? `Champion · ${entry.best}` : 'Champion（暂无）'}];
  for (const item of config.recent[rule] || []) {
    if (item.name !== entry.latest && item.name !== entry.best) {
      const file = item.name.split('/').at(-1).replace('checkpoint-','#').replace('.pt','');
      options.push({value:item.name, label:`${item.run} / ${file} · ${item.mtime} · ${item.mb}MB`});
    }
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
  const winners = winningLine(s);
  paintStones(cells, s, winners, $('numbers').checked);
  cells.forEach((b, i) => {
    const v = s.board[i];
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
  const tauntKeyNow = tauntKey(s);
  if (tauntKeyNow && tauntKeyNow !== tauntShown) { tauntShown = tauntKeyNow; showTaunt(s.taunt); }
  if (!tauntKeyNow && !s.history.length && Date.now() >= tauntHoldUntil) hideTaunt();
  $('undo').disabled = !active || busy || !!s.error || !s.history.some((_,i) => (i%2 === 0 ? 1 : -1) === s.human);
  $('model-name').textContent = active ? `${ruleName(s.model.rule)} / ${s.model.checkpoint === 'best' ? 'Champion' : s.model.checkpoint === 'latest' ? '最新模型' : '指定模型'}` : '尚未载入';
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
  hideTaunt();
  mutate('/api/new', {rule:$('rule').value, color, checkpoint:selection, simulations:Number($('simulations').value)});
};
$('undo').onclick = async () => {
  const before = state?.history.length ?? 0;
  await mutate('/api/undo', {id: state.id});
  if (state && !state.error && state.history.length < before) {
    // The restored position counts as announced, so it cannot talk over the undo line.
    tauntShown = tauntKey(state);
    showTaunt('undo');
  }
};
async function copyLink(inputId, buttonId) {
  const link = $(inputId).value;
  if (!link) return;
  try { await navigator.clipboard.writeText(link); $(buttonId).textContent = '已复制'; }
  catch (e) { $(inputId).select(); showError('浏览器不允许自动复制，请手动复制上面的链接。'); }
  setTimeout(() => { $(buttonId).textContent = '复制'; }, 1500);
}
$('invite-copy').onclick = () => copyLink('invite', 'invite-copy');
$('watch-copy').onclick = () => copyLink('watch', 'watch-copy');
// Leaving frees the slot for someone else; the next visit asks for the link again.
$('leave').onclick = async () => {
  if (!confirm('结束本桌？当前棋局会立即丢弃，名额让给其他人。')) return;
  try { await api('/api/leave', {}); } catch (e) { showError(e.message); return; }
  location.replace('/');
};
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
