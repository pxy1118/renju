// Board geometry and stone painting shared by the player page and the
// read-only watch page. Each page loads this file before its own script.
const letters = 'ABCDEFGHIJKLMNO';
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
// One button per crossing. The player page passes a click handler, the watch
// page passes none, which is what makes its board read-only.
function buildBoard(target, onClick) {
  const cells = [];
  for (let i = 0; i < 225; i++) {
    const r = Math.floor(i / 15), c = i % 15, b = document.createElement('button');
    b.className = 'point' + (r === 0 ? ' top' : '') + (r === 14 ? ' bottom' : '') + (c === 0 ? ' left' : '') + (c === 14 ? ' right' : '') + ([48,56,112,168,176].includes(i) ? ' star' : '');
    b.innerHTML = `<span class="piece"></span>${r === 0 ? `<span class="coord col">${letters[c]}</span>` : ''}${c === 0 ? `<span class="coord row">${r+1}</span>` : ''}`;
    b.setAttribute('aria-label', `${letters[c]}${r+1}，空位`);
    if (onClick) b.addEventListener('click', () => onClick(i));
    cells.push(b); target.appendChild(b);
  }
  return cells;
}
// Stones, the last-move dot, and the winning-line glow. Legality, aria and
// status text stay with each page, because a spectator is never "to move".
function paintStones(cells, state, winners, numbered) {
  const moveNumbers = new Map(state.history.map((a, i) => [a, i+1]));
  cells.forEach((b, i) => {
    const v = state.board[i], p = b.querySelector('.piece');
    p.className = 'piece' + (v ? (v === 1 ? ' black' : ' white') : '') + (state.history.at(-1) === i ? ' last' : '');
    p.classList.toggle('win', winners.has(i));
    p.textContent = v && numbered ? moveNumbers.get(i) : '';
  });
}
