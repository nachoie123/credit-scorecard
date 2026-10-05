/* Credit scorecard landing. Every number drawn here comes from scorecard_results.json (written by scorecard.py). */
(() => {
  document.documentElement.classList.add('js');
  const $ = s => document.querySelector(s);
  const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const fmt = (v, d = 0) => v.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
  const minus = s => String(s).replace('-', '−');

  /* count a number up (jumps straight to the end with reduced motion) */
  function count(el, to, ms, dec = 0) {
    if (still) { el.textContent = fmt(to, dec); return; }
    const from = parseFloat(el.textContent.replace(/,/g, '')) || 0, t0 = performance.now();
    (function f(t) {
      const k = Math.min(1, (t - t0) / ms), e = 1 - Math.pow(1 - k, 3);
      el.textContent = fmt(from + (to - from) * e, dec);
      if (k < 1) requestAnimationFrame(f);
    })(t0);
  }

  /* hero: the application breaks into nine boxes and the total adds up box by box */
  const form = $('#form'), total = $('#total');
  const pts = [...form.querySelectorAll('.p')].map(b => parseInt(b.textContent.replace('+', ''), 10));
  total.textContent = '0';
  setTimeout(() => {
    form.classList.add('split');
    let run = 0;
    pts.forEach((p, i) => setTimeout(() => { run += p; still ? (total.textContent = run) : count(total, run, 260); }, still ? 0 : 400 + i * 110));
  }, still ? 0 : 650);

  /* film: never autoplays; the button starts it with sound, then native controls take over */
  const film = $('.player video'), pb = $('.player .play');
  film.controls = false;
  pb.addEventListener('click', () => { film.preload = 'auto'; film.play().catch(() => { film.controls = true; pb.remove(); }); });
  film.addEventListener('play', () => { film.controls = true; pb.remove(); }, { once: true });

  fetch('scorecard_results.json').then(r => r.json()).then(build).catch(() => {});

  const NAMES = {
    checking_status: 'Checking account', duration_months: 'Loan length', credit_history: 'Credit history',
    purpose: 'Purpose', employment_since: 'In current job', other_installment_plans: 'Other instalment plans',
    savings: 'Savings', credit_amount: 'Amount', installment_rate: 'Instalment, share of income', telephone: 'Telephone'
  };
  const UNIT = { duration_months: ' months', credit_amount: ' DM' };
  const ME = { checking_status: '< 0 DM', duration_months: '(6, 15]', credit_history: 'existing credits paid duly', purpose: 'radio/TV',
    employment_since: '>= 7 years', other_installment_plans: 'none', savings: '>= 1000 DM / 500-1000 DM', credit_amount: '<= 3960.50', installment_rate: '<= 3' };

  function label(col, bin) {
    const u = UNIT[col] || '', n = x => fmt(Math.round(parseFloat(x)));
    if (col === 'installment_rate') return bin === '<= 3' ? 'Band 1 to 3' : 'Band 4';
    const CK = { 'no checking account': 'No account', '>= 200 DM': '200 DM or more', '0-200 DM': '0 to 200 DM', '< 0 DM': 'Overdrawn' };
    if (CK[bin]) return CK[bin];
    let m;
    if ((m = bin.match(/^<= ([\d.]+)$/))) return `${n(m[1])}${u} or less`;
    if ((m = bin.match(/^> ([\d.]+)$/))) return `Over ${n(m[1])}${u}`;
    if ((m = bin.match(/^\(([\d.]+), ([\d.]+)\]$/))) return `${fmt(Math.floor(parseFloat(m[1])) + 1)} to ${n(m[2])}${u}`;  // (6, 15] = 7 to 15 months
    const t = bin.replace('>= ', '≥ ').replace('< ', '< ').replace('/', ' / ').replace(/\s+/g, ' ');
    return t.charAt(0).toUpperCase() + t.slice(1);
  }

  function build(d) {
    const card = d.scorecard;

    /* layer 1: one box under the microscope */
    const ck = card.checking_status.rows;
    const BL = { 'no checking account': ['No account', ''], '>= 200 DM': ['200 DM or more', ''], '0-200 DM': ['0 to 200 DM', ''], '< 0 DM': ['Overdrawn', 'under 0 DM'] };
    $('#bins').innerHTML = ck.map(r => `<div class="brow${r.bin === '< 0 DM' ? ' me' : ''}"><div class="lab">${BL[r.bin][0]}<small>${r.count} people${BL[r.bin][1] ? ', ' + BL[r.bin][1] : ''}</small></div><div class="track"><i class="bar"></i></div><div class="val">0</div></div>`).join('');
    const brows = [...document.querySelectorAll('.brow')];
    function binMode(mode) {
      const tr = $('#bins');
      tr.style.setProperty('--zero', mode === 'woe' ? '50%' : '0%');
      brows.forEach((row, i) => {
        const r = ck[i], bar = row.querySelector('.bar'), val = row.querySelector('.val'), s = row.querySelector('.track').style;
        s.setProperty('--zero', mode === 'woe' ? '50%' : '0%');
        let l = 0, w = 0, c = 'var(--bad)', txt;
        if (mode === 'rate') { w = r.bad_rate / .6 * 100; txt = Math.round(r.bad_rate * 100) + '%'; }
        if (mode === 'woe') { const h = Math.abs(r.woe) / 1.2 * 50; w = h; l = r.woe >= 0 ? 50 : 50 - h; c = r.woe >= 0 ? 'var(--good)' : 'var(--bad)'; txt = (r.woe >= 0 ? '+' : '−') + Math.abs(r.woe).toFixed(2); }
        if (mode === 'points') { w = r.points / 90 * 100; c = r.bin === '< 0 DM' ? 'var(--bad)' : 'var(--ink)'; txt = r.points + ' pts'; }
        bar.style.setProperty('--l', l + '%'); bar.style.setProperty('--w', w + '%'); bar.style.setProperty('--c', c);
        val.textContent = txt;
      });
    }
    binMode('rate');

    /* layer 3: the nine weights, plus the one thrown out */
    const terms = d.model.terms, drop = d.dropped_for_instability;
    $('#betas').innerHTML = terms.map((t, i) => `<div class="beta" style="--i:${i}"><span>${NAMES[t.column]}</span><span class="bt"><i style="--w:${Math.abs(t.coefficient) / 2 * 100}%"></i></span><b>${minus(t.coefficient.toFixed(3))}</b></div>`).join('')
      + drop.map(t => `<div class="beta out" style="--i:9"><span>${NAMES[t.column] || t.pretty}</span><span class="bt"></span><b>${Math.round(t.frequency * 100)}%</b></div>`).join('')
      + `<div class="axis0"><span></span><span><span>β = −2</span><span>−1</span><span>0</span></span><span></span></div>`;

    /* layer 5: nine boxes stacked into one score */
    const order = d.selected, mine = order.map(c => card[c].rows.find(r => r.bin === ME[c]).points);
    const score = mine.reduce((a, b) => a + b, 0), MAX = d.score_range.max, MIN = d.score_range.min, CUT = d.best_cutoff.cutoff;
    const pc = v => v / MAX * 100;
    $('#stack').innerHTML = `<div class="slab"><span>0</span><span>${MAX} best possible</span></div>`
      + `<div class="sbar">${mine.map((p, i) => `<i style="--w:${pc(p)}%;--i:${i}"${i === 0 ? ' class="lo"' : ''}>${p}</i>`).join('')}<span class="rest"></span></div>`
      + `<div class="ticks"><span style="left:${pc(MIN)}%">${MIN}<br>lowest possible</span><span class="tot" style="left:${pc(score)}%">${score}</span><span class="cut" style="left:${pc(CUT)}%;top:30px">cutoff ${CUT}</span></div>`;

    /* layer 6-7: mirrored dot plot of the 300 holdout scores */
    const narrow = matchMedia('(max-width: 899px)').matches;
    const sd = d.score_distribution, W = narrow ? 380 : 640, X0 = 400, X1 = 625, x = s => 20 + (s - X0) / (X1 - X0) * (W - 40);
    const R = narrow ? 2.9 : 4.3, CW = narrow ? 6.6 : 10, ST = narrow ? 6.4 : 9.6, cols = { g: {}, b: {} };
    const place = (arr, k) => arr.slice().sort((a, b) => a - b).map(s => { const c = Math.round(x(s) / CW); const n = cols[k][c] = (cols[k][c] || 0) + 1; return { s, cx: c * CW, n }; });
    const G = place(sd.goods, 'g'), B = place(sd.bads, 'b');
    const top = Math.max(...G.map(p => p.n)), bot = Math.max(...B.map(p => p.n));
    const Y0 = 52 + 8 + top * ST, H = Y0 + 8 + bot * ST + 40;
    const appr = G.filter(p => p.s >= CUT).length + B.filter(p => p.s >= CUT).length, badAppr = B.filter(p => p.s >= CUT).length;
    let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Scores of 300 holdout applicants">`;
    for (let s = 425; s <= 600; s += 25) svg += `<line class="grid" x1="${x(s)}" x2="${x(s)}" y1="30" y2="${H - 26}"/><text x="${x(s)}" y="${H - 8}" text-anchor="middle">${s}</text>`;
    svg += `<g class="cutg"><rect x="${x(CUT)}" y="22" width="${W - x(CUT)}" height="${H - 48}"/><line x1="${x(CUT)}" x2="${x(CUT)}" y1="16" y2="${H - 26}"/><text x="${x(CUT) - 8}" y="16" text-anchor="end">cutoff ${CUT}</text><text class="ap" x="${W - 4}" y="${H - 34}" text-anchor="end">approved ${appr}, ${badAppr} default</text><text x="${x(CUT) - 8}" y="42" text-anchor="end">declined ${300 - appr}</text></g>`;
    svg += `<line class="axis" x1="0" x2="${W}" y1="${Y0}" y2="${Y0}"/>`;
    svg += G.map(p => `<circle class="gd${p.s < CUT ? ' rej' : ''}" cx="${p.cx}" cy="${Y0 - 8 - (p.n - 1) * ST}" r="${R}"/>`).join('');
    svg += B.map(p => `<circle class="bd${p.s < CUT ? ' rej' : ''}" cx="${p.cx}" cy="${Y0 + 8 + (p.n - 1) * ST}" r="${R}"/>`).join('');
    svg += `<text class="lbl g" x="2" y="${Y0 - 14 - top * ST * .55}">repaid, ${sd.goods.length}</text><text class="lbl r" x="2" y="${Y0 + 24 + bot * ST * .55}">defaulted, ${sd.bads.length}</text>`;
    svg += `<g class="me"><line x1="${x(score)}" x2="${x(score)}" y1="20" y2="${Y0}"/><text x="${x(score) + 6}" y="16">our applicant, ${score}</text></g></svg>`;
    $('#swarm').innerHTML = svg;

    /* layer 8: the Gini gauge with its spread */
    const P = d.performance, cv = d.cross_validation, cx = 200, cy = 200, rr = 150;
    const pt = (v, r = rr) => [cx + r * Math.cos(Math.PI * (1 - v)), cy - r * Math.sin(Math.PI * (1 - v))];
    const arc = (a, b) => { const [x1, y1] = pt(a), [x2, y2] = pt(b); return `M${x1.toFixed(1)} ${y1.toFixed(1)} A${rr} ${rr} 0 0 1 ${x2.toFixed(1)} ${y2.toFixed(1)}`; };
    let g = `<svg viewBox="0 0 400 275" role="img" aria-label="Gini ${P.test.gini.toFixed(3)} on the holdout; 90% of 25 rebuilds between ${cv.gini_p05.toFixed(2)} and ${cv.gini_p95.toFixed(2)}">`;
    g += `<path class="base" d="${arc(0, 1)}"/><path class="band" d="${arc(cv.gini_p05, cv.gini_p95)}"/>`;
    [[0, '0 coin flip'], [.25, '0.25'], [.5, '0.5'], [.75, '0.75'], [1, '1 perfect']].forEach(([v, t]) => {
      const [a, b] = pt(v, rr + 16), [c2, e] = pt(v, rr + 34);
      const end = v === 0 || v === 1, [ex] = pt(v);  // the two ends sit under the arc, not beside it
      g += end ? `<text x="${ex}" y="${cy + 36}" text-anchor="middle">${t}</text>`
        : `<line class="tk" x1="${a}" y1="${b}" x2="${pt(v, rr + 22)[0]}" y2="${pt(v, rr + 22)[1]}"/><text x="${c2}" y="${e + 4}" text-anchor="middle">${t}</text>`;
    });
    const [ta, tb] = pt(P.train.gini, rr - 16), [tc, te] = pt(P.train.gini, rr + 16), [tl, tm] = pt(P.train.gini, rr + 40);
    g += `<g class="tr"><line x1="${ta}" y1="${tb}" x2="${tc}" y2="${te}"/><text x="${tl + 6}" y="${tm + 2}" text-anchor="start">train ${P.train.gini.toFixed(2)}</text></g>`;
    g += `<g class="needle" id="needle"><line x1="${cx}" y1="${cy}" x2="${cx}" y2="${cy - rr + 30}"/><circle cx="${cx}" cy="${cy}" r="9"/></g>`;
    g += `<text class="big" id="gini" x="${cx}" y="${cy + 60}" text-anchor="middle">0.000</text></svg>`;
    $('#gauge').innerHTML = g + `<div class="gnote"><span>holdout Gini <b>${P.test.gini.toFixed(3)}</b></span><span>AUC <b>${P.test.auc.toFixed(3)}</b></span><span>KS <b>${P.test.ks.toFixed(3)}</b></span><span class="sw">25 rebuilds, 90% within <b>${cv.gini_p05.toFixed(2)} to ${cv.gini_p95.toFixed(2)}</b></span></div>`;
    const needle = $('#needle'), giniEl = $('#gini');

    /* the steps drive the stage */
    const stage = $('#stage'), eq = $('#eq'), layers = [...stage.querySelectorAll('.layer')];
    const EQ = {
      1: 'bad rate = defaulted ÷ people in the bin',
      2: 'WOE = ln( %repaid ÷ %defaulted )',
      3: `ln(odds of default) = <var>α</var> + Σ <var>β</var>·WOE`,
      4: `points = −(<var>β</var>·WOE + <var>α</var>/n) × 28.85 + 487.12/n`,
      5: `score = ${mine.join(' + ')} = <var>${score}</var>`,
      6: `${sd.goods.length + sd.bads.length} unseen scores: ${sd.goods.length} repaid, ${sd.bads.length} defaulted`,
      7: `cost = 5 × defaulters approved + 1 × repayers declined`,
      8: `Gini = 2 × AUC − 1 = <var>${P.test.gini.toFixed(3)}</var>`
    };
    let cur = 1, swapT;
    function setStep(n) {
      if (n === cur) return; cur = n;
      stage.dataset.step = n;
      layers.forEach(l => l.classList.toggle('on', l.dataset.show.split(' ').includes(String(n))));
      clearTimeout(swapT); eq.classList.add('swap');
      swapT = setTimeout(() => { eq.innerHTML = `<span>${EQ[n]}</span>`; eq.classList.remove('swap'); }, still ? 0 : 220);
      if (n === 1) binMode('rate'); if (n === 2) binMode('woe'); if (n === 4) binMode('points');
      if (n === 8) { needle.style.transform = `rotate(${(P.test.gini - .5) * 180}deg)`; count(giniEl, P.test.gini, 1200, 3); }
    }
    eq.innerHTML = `<span>${EQ[1]}</span>`;
        const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) setStep(+e.target.dataset.step); }),
      { rootMargin: narrow ? '-62% 0px -30% 0px' : '-45% 0px -45% 0px' });
    document.querySelectorAll('.step').forEach(s => io.observe(s));

    /* the whole card: tap one bin per characteristic */
    const pick = {};
    $('#chars').innerHTML = order.map(c => {
      const rows = card[c].rows;
      return `<div class="char"><h3>${NAMES[c]}<small>IV ${card[c].iv.toFixed(2)}, β ${minus(card[c].coefficient.toFixed(2))}</small></h3><div class="opts" role="group" aria-label="${NAMES[c]}">`
        + rows.map(r => `<button class="opt" type="button" data-c="${c}" data-p="${r.points}" data-b="${r.bin}" aria-pressed="false">${label(c, r.bin)} <b>${r.points}</b></button>`).join('') + '</div></div>';
    }).join('');
    const scoreEl = $('#score'), dec = $('#dec'), odds = $('#odds');
    function tally() {
      const s = order.reduce((a, c) => a + pick[c], 0), o = 50 * Math.pow(2, (s - 600) / 20), pd = 1 / (1 + o);
      count(scoreEl, s, 300);
      dec.textContent = s >= CUT ? `Approved, ${CUT} or more` : `Declined, under ${CUT}`;
      dec.classList.toggle('no', s < CUT);
      odds.textContent = (o >= 1 ? `Odds of repaying about ${o >= 10 ? fmt(o) : o.toFixed(1)} to 1` : `Odds of repaying about 1 to ${fmt(1 / o)}`) + `, a ${(pd * 100).toFixed(1)}% chance of default.`;
    }
    function choose(btn) {
      const c = btn.dataset.c;
      btn.parentElement.querySelectorAll('.opt').forEach(b => b.setAttribute('aria-pressed', b === btn));
      pick[c] = +btn.dataset.p;
    }
    function reset() { order.forEach(c => choose(document.querySelector(`.opt[data-c="${c}"][data-b="${CSS.escape(ME[c])}"]`))); tally(); }
    $('#chars').addEventListener('click', e => { const b = e.target.closest('.opt'); if (b) { choose(b); tally(); } });
    $('#reset').addEventListener('click', reset);
    reset();
  }
})();
