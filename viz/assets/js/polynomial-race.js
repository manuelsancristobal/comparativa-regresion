/**
 * Polynomial Regression Visualization - Growing Data
 * Scatter points appear progressively and darken over time.
 */

const PolynomialRaceViz = (() => {
  let data = null;
  let state = {
    currentFrame: 0,
    isPlaying: false,
    speed: 1,
    animationId: null,
    visibleDegrees: { 1: true, 2: true, 3: true, 4: true }
  };

  const SVG_WIDTH = 700;
  const SVG_HEIGHT = 500;
  const DEGREES = [1, 2, 3, 4];
  const COLORS = { 1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728' };

  let els = {};

  const loadData = async () => {
    const response = await fetch('assets/data/polynomial_frames.json');
    data = await response.json();
  };

  // ─── SETUP ───

  const setup = () => {
    controls.setMax(data.frames.length - 1);
    setupPanelA();
    setupPanelB();
    setupPanelC();
    update(0);
  };

  const setupPanelA = () => {
    const m = { top: 20, right: 30, bottom: 40, left: 60 };
    const w = SVG_WIDTH - m.left - m.right;
    const h = SVG_HEIGHT - m.top - m.bottom;

    const container = d3.select('#panel-a-poly').html('');
    container.append('h3').text('MedInc vs MedHouseVal + Curvas Polinómicas');
    const svg = container.append('svg').attr('width', SVG_WIDTH).attr('height', SVG_HEIGHT);
    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    const xScale = d3.scaleLinear().domain(d3.extent(data.scatter.x)).range([0, w]);
    const yScale = d3.scaleLinear().domain(d3.extent(data.scatter.y)).range([h, 0]);

    g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(xScale));
    g.append('g').call(d3.axisLeft(yScale));

    const scatterGroup = g.append('g').attr('class', 'scatter-group');

    const xDomain = xScale.domain();
    const xRange = d3.range(xDomain[0], xDomain[1], (xDomain[1] - xDomain[0]) / 100);

    const curvePaths = {};
    DEGREES.forEach(deg => {
      curvePaths[deg] = g.append('path')
        .attr('fill', 'none').attr('stroke', COLORS[deg])
        .attr('stroke-width', 2).attr('opacity', 0.8);
    });

    const counter = g.append('text')
      .attr('x', w - 10).attr('y', 15)
      .attr('text-anchor', 'end')
      .attr('font-size', '13px')
      .attr('fill', '#555');

    els.panelA = { xScale, yScale, scatterGroup, curvePaths, xRange, counter, w, h };
  };

  const setupPanelB = () => {
    const m = { top: 20, right: 20, bottom: 40, left: 60 };
    const w = SVG_WIDTH / 2 - 10 - m.left - m.right;
    const h = 300 - m.top - m.bottom;

    const container = d3.select('#panel-b-poly').html('');
    container.append('h3').text('MSE Train vs Test por Grado');
    const svg = container.append('svg').attr('width', SVG_WIDTH / 2 - 10).attr('height', 300);
    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    const xScale = d3.scaleBand().domain(DEGREES.map(d => `Deg ${d}`)).range([0, w]).padding(0.2);
    g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(xScale));
    const yAxisG = g.append('g');

    const trainBars = {};
    const testBars = {};
    DEGREES.forEach(deg => {
      const label = `Deg ${deg}`;
      const bw = xScale.bandwidth() / 2;
      trainBars[deg] = g.append('rect').attr('x', xScale(label)).attr('width', bw)
        .attr('fill', COLORS[deg]).attr('opacity', 0.7);
      testBars[deg] = g.append('rect').attr('x', xScale(label) + bw).attr('width', bw)
        .attr('fill', COLORS[deg]).attr('opacity', 0.35);
    });

    els.panelB = { yAxisG, trainBars, testBars, w, h };
  };

  const setupPanelC = () => {
    const m = { top: 20, right: 20, bottom: 40, left: 60 };
    const w = SVG_WIDTH / 2 - 10 - m.left - m.right;
    const h = 300 - m.top - m.bottom;

    const container = d3.select('#panel-c-poly').html('');
    container.append('h3').text('Evolución de Coeficientes');
    const svg = container.append('svg').attr('width', SVG_WIDTH / 2 - 10).attr('height', 300);
    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    const yAxisG = g.append('g');
    const barsGroup = g.append('g');

    els.panelC = { g, yAxisG, barsGroup, w, h };
  };

  // ─── UPDATE ───

  const update = (frame) => {
    const frameIdx = Math.min(frame, data.frames.length - 1);
    const frameData = data.frames[frameIdx];

    updateScatter(frameData.n_scatter);
    updateCurves(frameData);
    updateMSE(frameData);
    updateCoefficients(frameData);

    els.panelA.counter.text(`n = ${frameData.n_points}`);
    controls.setFrame(frame, data.frames.length - 1);
  };

  const updateScatter = (nScatter) => {
    const { scatterGroup, xScale, yScale } = els.panelA;
    const nShow = Math.min(nScatter, data.scatter.x.length);

    const pointsData = data.scatter.x.slice(0, nShow).map((x, i) => ({
      x, y: data.scatter.y[i], idx: i
    }));

    const sel = scatterGroup.selectAll('.scatter-point').data(pointsData, d => d.idx);

    sel.enter()
      .append('circle')
      .attr('class', 'scatter-point')
      .attr('cx', d => xScale(d.x))
      .attr('cy', d => yScale(d.y))
      .attr('r', 3)
      .attr('fill', '#888')
      .attr('opacity', 0.1);

    scatterGroup.selectAll('.scatter-point')
      .attr('opacity', d => 0.1 + 0.5 * ((nShow - d.idx) / nShow));
  };

  const updateCurves = (frameData) => {
    const { xScale, yScale, curvePaths, xRange } = els.panelA;

    DEGREES.forEach(deg => {
      const key = `degree_${deg}`;
      const visible = state.visibleDegrees[deg] && frameData[key] && frameData[key].coefficients;
      curvePaths[deg].attr('visibility', visible ? 'visible' : 'hidden');
      if (!visible) return;

      const coeffs = frameData[key].coefficients;
      const pts = xRange.map(x => {
        let y = 0;
        for (let i = 0; i < coeffs.length; i++) y += coeffs[i] * Math.pow(x, i);
        return { x, y };
      });

      curvePaths[deg].attr('d', d3.line().x(d => xScale(d.x)).y(d => yScale(d.y))(pts));
    });
  };

  const updateMSE = (frameData) => {
    const { trainBars, testBars, yAxisG, h } = els.panelB;
    const vals = [];
    DEGREES.forEach(d => {
      if (state.visibleDegrees[d] && frameData[`degree_${d}`]) {
        vals.push(frameData[`degree_${d}`].mse_train, frameData[`degree_${d}`].mse_test);
      }
    });
    if (vals.length === 0) return;

    const yScale = d3.scaleLinear().domain([0, d3.max(vals) * 1.1]).range([h, 0]);
    yAxisG.call(d3.axisLeft(yScale).ticks(5));

    DEGREES.forEach(deg => {
      const key = `degree_${deg}`;
      const show = state.visibleDegrees[deg] && frameData[key];
      trainBars[deg].attr('visibility', show ? 'visible' : 'hidden');
      testBars[deg].attr('visibility', show ? 'visible' : 'hidden');
      if (!show) return;
      const tY = yScale(frameData[key].mse_train);
      const eY = yScale(frameData[key].mse_test);
      trainBars[deg].attr('y', tY).attr('height', h - tY);
      testBars[deg].attr('y', eY).attr('height', h - eY);
    });
  };

  const updateCoefficients = (frameData) => {
    const { barsGroup, yAxisG, w, h } = els.panelC;
    barsGroup.selectAll('*').remove();

    const coeffData = [];
    DEGREES.forEach(d => {
      if (!state.visibleDegrees[d]) return;
      const coeffs = frameData[`degree_${d}`]?.coefficients;
      if (!coeffs) return;
      coeffs.forEach((c, i) => coeffData.push({ degree: d, power: i, value: Math.abs(c) }));
    });
    if (coeffData.length === 0) return;

    const xScale = d3.scaleBand().domain(coeffData.map((_, i) => i)).range([0, w]).padding(0.05);
    const yScale = d3.scaleLinear().domain([0, d3.max(coeffData, d => d.value) * 1.1]).range([h, 0]);
    yAxisG.call(d3.axisLeft(yScale).ticks(4));

    barsGroup.selectAll('.coeff-bar').data(coeffData).enter()
      .append('rect')
      .attr('x', (_, i) => xScale(i))
      .attr('y', d => yScale(d.value))
      .attr('width', xScale.bandwidth())
      .attr('height', d => h - yScale(d.value))
      .attr('fill', d => COLORS[d.degree])
      .attr('opacity', 0.7);
  };

  // ─── ANIMATION ───

  const animate = () => {
    if (!state.isPlaying) return;
    state.currentFrame = Math.min(state.currentFrame + state.speed, data.frames.length - 1);
    update(state.currentFrame);
    if (state.currentFrame >= data.frames.length - 1) {
      state.isPlaying = false;
      controls.playButton.classed('active', false).text('Play');
      return;
    }
    state.animationId = requestAnimationFrame(animate);
  };

  const startAnimation = () => {
    if (state.currentFrame >= data.frames.length - 1) state.currentFrame = 0;
    state.isPlaying = true;
    state.animationId = requestAnimationFrame(animate);
  };

  const stopAnimation = () => {
    state.isPlaying = false;
    if (state.animationId) { cancelAnimationFrame(state.animationId); state.animationId = null; }
  };

  const controls = CommonUtils.createPlaybackControls('#controls-poly', {
    onPlayPause: (p) => p ? startAnimation() : stopAnimation(),
    onFrameChange: (f) => { stopAnimation(); state.currentFrame = f; controls.playButton.classed('active', false).text('Play'); update(f); },
    onSpeedChange: (s) => { state.speed = s; }
  });

  DEGREES.forEach(d => {
    d3.select('#methods-poly').append('div').attr('class', 'toggle-item')
      .html(`<input type="checkbox" id="toggle-degree-${d}" checked>
        <label for="toggle-degree-${d}"><span class="color-indicator" style="background: ${COLORS[d]}"></span> Degree ${d}</label>`)
      .select('input').on('change', function() { state.visibleDegrees[d] = this.checked; update(state.currentFrame); });
  });

  return { init: async () => { await loadData(); setup(); } };
})();

document.addEventListener('DOMContentLoaded', () => { PolynomialRaceViz.init(); });
