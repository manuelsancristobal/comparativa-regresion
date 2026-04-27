/**
 * Linear Regression Visualization - 3 Methods with Growing Data
 * Scatter points appear progressively and darken over time.
 */

const LinearRaceViz = (() => {
  let data = null;
  let state = {
    currentFrame: 0,
    isPlaying: false,
    speed: 1,
    animationId: null,
    visibleMethods: { analytical: true, gradient_descent: true, sklearn: true }
  };

  const SVG_WIDTH = 700;
  const SVG_HEIGHT = 500;
  const COLORS = {
    analytical: '#1f77b4',
    gradient_descent: '#d62728',
    sklearn: '#2ca02c'
  };
  const METHODS = ['analytical', 'gradient_descent', 'sklearn'];
  const METHOD_LABELS = { analytical: 'Ana', gradient_descent: 'GD', sklearn: 'SK' };

  let els = {};

  const loadData = async () => {
    const response = await fetch('./assets/data/linear_frames.json');
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

    const container = d3.select('#panel-a').html('');
    container.append('h3').text('MedInc vs MedHouseVal + Rectas de Regresión');
    const svg = container.append('svg').attr('width', SVG_WIDTH).attr('height', SVG_HEIGHT);
    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    const xScale = d3.scaleLinear().domain(d3.extent(data.scatter.x)).range([0, w]);
    const yScale = d3.scaleLinear().domain(d3.extent(data.scatter.y)).range([h, 0]);

    g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(xScale));
    g.append('g').call(d3.axisLeft(yScale));

    // Scatter group (filled progressively)
    const scatterGroup = g.append('g').attr('class', 'scatter-group');

    // Regression lines
    const lines = {};
    const labels = {};
    METHODS.forEach(method => {
      lines[method] = g.append('line')
        .attr('stroke', COLORS[method])
        .attr('stroke-width', 2);
      labels[method] = g.append('text')
        .attr('fill', COLORS[method])
        .attr('font-size', '12px')
        .attr('font-weight', 'bold')
        .text(METHOD_LABELS[method]);
    });

    // Counter label
    const counter = g.append('text')
      .attr('x', w - 10).attr('y', 15)
      .attr('text-anchor', 'end')
      .attr('font-size', '13px')
      .attr('fill', '#555');

    els.panelA = { xScale, yScale, scatterGroup, lines, labels, counter, w, h };
  };

  const setupPanelB = () => {
    const m = { top: 20, right: 20, bottom: 40, left: 60 };
    const w = SVG_WIDTH / 2 - 10 - m.left - m.right;
    const h = 300 - m.top - m.bottom;

    const container = d3.select('#panel-b').html('');
    container.append('h3').text('Carrera MSE (escala log)');
    const svg = container.append('svg').attr('width', SVG_WIDTH / 2 - 10).attr('height', 300);
    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    const xScale = d3.scaleBand()
      .domain(METHODS.map(m => METHOD_LABELS[m]))
      .range([0, w]).padding(0.2);

    g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(xScale));
    const yAxisG = g.append('g');

    const trainBars = {};
    const testBars = {};
    METHODS.forEach(method => {
      const label = METHOD_LABELS[method];
      const bw = xScale.bandwidth() / 2;
      trainBars[method] = g.append('rect')
        .attr('x', xScale(label)).attr('width', bw)
        .attr('fill', COLORS[method]).attr('opacity', 0.7);
      testBars[method] = g.append('rect')
        .attr('x', xScale(label) + bw).attr('width', bw)
        .attr('fill', COLORS[method]).attr('opacity', 0.35);
    });

    els.panelB = { yAxisG, trainBars, testBars, w, h };
  };

  const setupPanelC = () => {
    const m = { top: 20, right: 20, bottom: 40, left: 60 };
    const w = SVG_WIDTH / 2 - 10 - m.left - m.right;
    const h = 300 - m.top - m.bottom;

    const container = d3.select('#panel-c').html('');
    container.append('h3').text('Convergencia de Slope por Método');
    const svg = container.append('svg').attr('width', SVG_WIDTH / 2 - 10).attr('height', 300);
    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    // X axis: frame index (data size)
    const xScale = d3.scaleLinear().domain([0, data.frames.length - 1]).range([0, w]);
    // Y axis: slope values
    const allSlopes = data.frames.flatMap(f =>
      METHODS.map(m => f[m]?.slope).filter(v => v !== undefined)
    );
    const yScale = d3.scaleLinear()
      .domain([d3.min(allSlopes) * 0.9, d3.max(allSlopes) * 1.1])
      .range([h, 0]);

    g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(xScale).ticks(5));
    g.append('g').call(d3.axisLeft(yScale));

    // Path for each method
    const paths = {};
    METHODS.forEach(method => {
      paths[method] = g.append('path')
        .attr('fill', 'none')
        .attr('stroke', COLORS[method])
        .attr('stroke-width', 2)
        .attr('opacity', 0.8);
    });

    els.panelC = { xScale, yScale, paths, w, h };
  };

  // ─── UPDATE ───

  const update = (frame) => {
    const frameIdx = Math.min(frame, data.frames.length - 1);
    const frameData = data.frames[frameIdx];

    updateScatter(frameData.n_scatter);
    updateLines(frameData);
    updateMSE(frameData);
    updateConvergence(frameIdx);

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

    // Darken existing points
    scatterGroup.selectAll('.scatter-point')
      .attr('opacity', d => {
        const age = (nShow - d.idx) / nShow;
        return 0.1 + 0.5 * age;
      });
  };

  const updateLines = (frameData) => {
    const { xScale, yScale, lines, labels } = els.panelA;
    const x0 = xScale.domain()[0];
    const x1 = xScale.domain()[1];

    METHODS.forEach(method => {
      const visible = state.visibleMethods[method];
      const r = frameData[method];
      const show = visible && r && r.slope !== undefined;

      lines[method].attr('visibility', show ? 'visible' : 'hidden');
      labels[method].attr('visibility', show ? 'visible' : 'hidden');

      if (!show) return;

      lines[method]
        .attr('x1', xScale(x0)).attr('y1', yScale(r.intercept + r.slope * x0))
        .attr('x2', xScale(x1)).attr('y2', yScale(r.intercept + r.slope * x1));
      labels[method]
        .attr('x', xScale(x1) - 10)
        .attr('y', yScale(r.intercept + r.slope * x1) - 10);
    });
  };

  const updateMSE = (frameData) => {
    const { trainBars, testBars, yAxisG, h } = els.panelB;

    const vals = [];
    METHODS.forEach(m => {
      if (state.visibleMethods[m] && frameData[m]) {
        vals.push(frameData[m].mse_train, frameData[m].mse_test);
      }
    });
    if (vals.length === 0) return;

    const minV = Math.max(0.001, d3.min(vals) * 0.9);
    const maxV = d3.max(vals) * 1.1;
    const yScale = d3.scaleLog().domain([minV, maxV]).range([h, 0]);
    yAxisG.call(d3.axisLeft(yScale).ticks(4, '.2f'));

    METHODS.forEach(method => {
      const show = state.visibleMethods[method] && frameData[method];
      trainBars[method].attr('visibility', show ? 'visible' : 'hidden');
      testBars[method].attr('visibility', show ? 'visible' : 'hidden');
      if (!show) return;
      const tY = yScale(Math.max(minV, frameData[method].mse_train));
      const eY = yScale(Math.max(minV, frameData[method].mse_test));
      trainBars[method].attr('y', tY).attr('height', h - tY);
      testBars[method].attr('y', eY).attr('height', h - eY);
    });
  };

  const updateConvergence = (frameIdx) => {
    const { xScale, yScale, paths } = els.panelC;

    METHODS.forEach(method => {
      if (!state.visibleMethods[method]) {
        paths[method].attr('visibility', 'hidden');
        return;
      }
      const points = [];
      for (let i = 0; i <= frameIdx; i++) {
        const f = data.frames[i];
        if (f[method] && f[method].slope !== undefined) {
          points.push({ x: i, y: f[method].slope });
        }
      }
      if (points.length > 1) {
        paths[method]
          .attr('d', d3.line().x(d => xScale(d.x)).y(d => yScale(d.y))(points))
          .attr('visibility', 'visible');
      }
    });
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

  // ─── CONTROLS ───

  const controls = CommonUtils.createPlaybackControls('#controls-linear', {
    onPlayPause: (p) => p ? startAnimation() : stopAnimation(),
    onFrameChange: (f) => { stopAnimation(); state.currentFrame = f; controls.playButton.classed('active', false).text('Play'); update(f); },
    onSpeedChange: (s) => { state.speed = s; }
  });

  CommonUtils.createMethodToggles('#methods-linear', Object.keys(COLORS), COLORS,
    (method, visible) => { state.visibleMethods[method] = visible; update(state.currentFrame); }
  );

  return { init: async () => { await loadData(); setup(); } };
})();

document.addEventListener('DOMContentLoaded', () => { LinearRaceViz.init(); });
