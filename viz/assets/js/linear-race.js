/**
 * Linear Regression Visualization - 3 Methods with Growing Data
 * Scatter points appear progressively with entrance animation.
 * Responsive SVGs with viewBox. Throttled animation loop.
 */

const LinearRaceViz = (() => {
  let data = null;
  let state = {
    currentFrame: 0,
    isPlaying: false,
    speed: 1,
    transitionMs: 0,
    visibleMethods: { analytical: true, gradient_descent: true, sklearn: true }
  };

  const SVG_WIDTH = 700;
  const SVG_HEIGHT = 500;
  const SMALL_W = 340;
  const SMALL_H = 300;
  const COLORS = {
    analytical: '#1f77b4',
    gradient_descent: '#d62728',
    sklearn: '#2ca02c'
  };
  const METHODS = ['analytical', 'gradient_descent'];
  const METHODS_ALL = ['analytical', 'gradient_descent', 'sklearn'];
  const METHOD_LABELS = { analytical: 'Ana', gradient_descent: 'GD', sklearn: 'SK' };

  let els = {};
  let sortedIndices = [];
  let animLoop = null;

  const loadData = async () => {
    const response = await fetch('assets/data/linear_frames.json');
    data = await response.json();

    sortedIndices = data.scatter.x.map((x, i) => i);
    sortedIndices.sort((a, b) => {
      const distA = Math.sqrt(data.scatter.x[a] ** 2 + data.scatter.y[a] ** 2);
      const distB = Math.sqrt(data.scatter.x[b] ** 2 + data.scatter.y[b] ** 2);
      return distA - distB;
    });
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
    const svg = container.append('svg')
      .attr('viewBox', `0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');
    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    const xScale = d3.scaleLinear().domain(d3.extent(data.scatter.x)).range([0, w]);
    const yScale = d3.scaleLinear().domain(d3.extent(data.scatter.y)).range([h, 0]);

    g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(xScale));
    g.append('g').call(d3.axisLeft(yScale));

    const scatterGroup = g.append('g').attr('class', 'scatter-group');

    // Línea sklearn fija (referencia dashed)
    const skRef = data.sklearn;
    const skLine = g.append('line')
      .attr('class', 'reference-line')
      .attr('stroke', COLORS.sklearn)
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '6,4')
      .attr('opacity', 0.6)
      .attr('x1', xScale(xScale.domain()[0]))
      .attr('y1', yScale(skRef.intercept + skRef.slope * xScale.domain()[0]))
      .attr('x2', xScale(xScale.domain()[1]))
      .attr('y2', yScale(skRef.intercept + skRef.slope * xScale.domain()[1]));
    const skLabel = g.append('text')
      .attr('class', 'reference-label')
      .attr('fill', COLORS.sklearn)
      .attr('font-size', '11px')
      .attr('opacity', 0.6)
      .attr('x', xScale(xScale.domain()[1]) - 10)
      .attr('y', yScale(skRef.intercept + skRef.slope * xScale.domain()[1]) - 10)
      .text('SK (ref)');

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

    const counter = g.append('text')
      .attr('x', w - 10).attr('y', 15)
      .attr('text-anchor', 'end')
      .attr('font-size', '13px')
      .attr('fill', '#555');

    els.panelA = { xScale, yScale, scatterGroup, lines, labels, skLine, skLabel, counter, w, h };
  };

  const setupPanelB = () => {
    const m = { top: 20, right: 20, bottom: 40, left: 60 };
    const w = SMALL_W - m.left - m.right;
    const h = SMALL_H - m.top - m.bottom;

    const container = d3.select('#panel-b').html('');
    container.append('h3').text('Carrera MSE (escala log)');
    const svg = container.append('svg')
      .attr('viewBox', `0 0 ${SMALL_W} ${SMALL_H}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');
    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    const xScale = d3.scaleBand()
      .domain(METHODS_ALL.map(m => METHOD_LABELS[m]))
      .range([0, w]).padding(0.2);

    g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(xScale));
    const yAxisG = g.append('g');

    const trainBars = {};
    const testBars = {};
    METHODS_ALL.forEach(method => {
      const label = METHOD_LABELS[method];
      const bw = xScale.bandwidth() / 2;
      trainBars[method] = g.append('rect')
        .attr('x', xScale(label)).attr('width', bw)
        .attr('fill', COLORS[method]).attr('opacity', 0.7)
        .attr('y', h).attr('height', 0);
      testBars[method] = g.append('rect')
        .attr('x', xScale(label) + bw).attr('width', bw)
        .attr('fill', COLORS[method]).attr('opacity', 0.35)
        .attr('y', h).attr('height', 0);
    });

    els.panelB = { yAxisG, trainBars, testBars, w, h };
  };

  const setupPanelC = () => {
    const m = { top: 20, right: 20, bottom: 40, left: 60 };
    const w = SMALL_W - m.left - m.right;
    const h = SMALL_H - m.top - m.bottom;

    const container = d3.select('#panel-c').html('');
    container.append('h3').text('Convergencia de Slope por Método');
    const svg = container.append('svg')
      .attr('viewBox', `0 0 ${SMALL_W} ${SMALL_H}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');
    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    const xScale = d3.scaleLinear().domain([0, data.frames.length - 1]).range([0, w]);
    const allSlopes = data.frames.flatMap(f =>
      METHODS.map(m => f[m]?.slope).filter(v => v !== undefined)
    );
    // Incluir slope sklearn para rango del eje Y
    allSlopes.push(data.sklearn.slope);
    const yScale = d3.scaleLinear()
      .domain([d3.min(allSlopes) * 0.9, d3.max(allSlopes) * 1.1])
      .range([h, 0]);

    g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(xScale).ticks(5));
    g.append('g').call(d3.axisLeft(yScale));

    // Línea horizontal de referencia sklearn (slope final)
    const skRefLine = g.append('line')
      .attr('class', 'reference-line')
      .attr('x1', 0).attr('x2', w)
      .attr('y1', yScale(data.sklearn.slope))
      .attr('y2', yScale(data.sklearn.slope))
      .attr('stroke', COLORS.sklearn)
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '6,4')
      .attr('opacity', 0.6);

    const paths = {};
    METHODS.forEach(method => {
      paths[method] = g.append('path')
        .attr('fill', 'none')
        .attr('stroke', COLORS[method])
        .attr('stroke-width', 2)
        .attr('opacity', 0.8);
    });

    els.panelC = { xScale, yScale, paths, skRefLine, w, h };
  };

  // ─── UPDATE ───

  const t = () => state.transitionMs > 0
    ? d3.transition().duration(state.transitionMs)
    : null;

  const applyTransition = (sel) => {
    const tr = t();
    return tr ? sel.transition(tr) : sel;
  };

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

    const allPoints = sortedIndices.map((origIdx, sortPos) => ({
      x: data.scatter.x[origIdx],
      y: data.scatter.y[origIdx],
      idx: origIdx,
      active: sortPos < nShow
    }));

    const sel = scatterGroup.selectAll('.scatter-point').data(allPoints, d => d.idx);

    // Entrance animation: fade in + scale up
    sel.enter()
      .append('circle')
      .attr('class', 'scatter-point')
      .attr('cx', d => xScale(d.x))
      .attr('cy', d => yScale(d.y))
      .attr('r', 0)
      .attr('fill', '#ccc')
      .attr('opacity', 0)
      .transition().duration(300)
      .attr('r', 3)
      .attr('opacity', 0.3);

    // Color transition for active/inactive
    applyTransition(scatterGroup.selectAll('.scatter-point'))
      .attr('fill', d => d.active ? '#74a9cf' : '#ccc')
      .attr('opacity', d => d.active ? 0.7 : 0.3);
  };

  const updateLines = (frameData) => {
    const { xScale, yScale, lines, labels, skLine, skLabel } = els.panelA;
    const x0 = xScale.domain()[0];
    const x1 = xScale.domain()[1];

    // Sklearn referencia: toggle visibilidad
    const skVisible = state.visibleMethods.sklearn !== false;
    skLine.attr('visibility', skVisible ? 'visible' : 'hidden');
    skLabel.attr('visibility', skVisible ? 'visible' : 'hidden');

    METHODS.forEach(method => {
      const visible = state.visibleMethods[method];
      const r = frameData[method];
      const show = visible && r && r.slope !== undefined;

      lines[method].attr('visibility', show ? 'visible' : 'hidden');
      labels[method].attr('visibility', show ? 'visible' : 'hidden');

      if (!show) return;

      applyTransition(lines[method])
        .attr('x1', xScale(x0)).attr('y1', yScale(r.intercept + r.slope * x0))
        .attr('x2', xScale(x1)).attr('y2', yScale(r.intercept + r.slope * x1));
      applyTransition(labels[method])
        .attr('x', xScale(x1) - 10)
        .attr('y', yScale(r.intercept + r.slope * x1) - 10);
    });
  };

  const updateMSE = (frameData) => {
    const { trainBars, testBars, yAxisG, h } = els.panelB;

    // Construir mapa de valores MSE: frames para métodos iterativos, fijo para sklearn
    const mseMap = {};
    METHODS.forEach(m => {
      if (state.visibleMethods[m] && frameData[m]) {
        mseMap[m] = { mse_train: frameData[m].mse_train, mse_test: frameData[m].mse_test };
      }
    });
    if (state.visibleMethods.sklearn !== false) {
      mseMap.sklearn = { mse_train: data.sklearn.mse_train, mse_test: data.sklearn.mse_test };
    }

    const vals = [];
    Object.values(mseMap).forEach(v => { vals.push(v.mse_train, v.mse_test); });
    if (vals.length === 0) return;

    const minV = Math.max(0.001, d3.min(vals) * 0.9);
    const maxV = d3.max(vals) * 1.1;
    const yScale = d3.scaleLog().domain([minV, maxV]).range([h, 0]);
    yAxisG.call(d3.axisLeft(yScale).ticks(4, '.2f'));

    METHODS_ALL.forEach(method => {
      const show = !!mseMap[method];
      trainBars[method].attr('visibility', show ? 'visible' : 'hidden');
      testBars[method].attr('visibility', show ? 'visible' : 'hidden');
      if (!show) return;
      const tY = yScale(Math.max(minV, mseMap[method].mse_train));
      const eY = yScale(Math.max(minV, mseMap[method].mse_test));
      applyTransition(trainBars[method]).attr('y', tY).attr('height', h - tY);
      applyTransition(testBars[method]).attr('y', eY).attr('height', h - eY);
    });
  };

  const updateConvergence = (frameIdx) => {
    const { xScale, yScale, paths, skRefLine } = els.panelC;

    // Toggle visibilidad línea referencia sklearn
    skRefLine.attr('visibility', state.visibleMethods.sklearn !== false ? 'visible' : 'hidden');

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

  // ─── ANIMATION (throttled) ───

  const startAnimation = () => {
    if (state.currentFrame >= data.frames.length - 1) state.currentFrame = 0;
    state.isPlaying = true;
    state.transitionMs = CommonUtils.TRANSITION_MS;

    animLoop = CommonUtils.createAnimationLoop(() => {
      state.currentFrame = Math.min(state.currentFrame + state.speed, data.frames.length - 1);
      update(state.currentFrame);
      if (state.currentFrame >= data.frames.length - 1) {
        state.isPlaying = false;
        state.transitionMs = 0;
        controls.playButton.classed('active', false).text('Play');
        return false;
      }
      return true;
    });
    animLoop.start();
  };

  const stopAnimation = () => {
    state.isPlaying = false;
    state.transitionMs = 0;
    if (animLoop) { animLoop.stop(); animLoop = null; }
  };

  // ─── CONTROLS ───

  const controls = CommonUtils.createPlaybackControls('#controls-linear', {
    onPlayPause: (p) => p ? startAnimation() : stopAnimation(),
    onFrameChange: (f) => { stopAnimation(); state.currentFrame = f; controls.playButton.classed('active', false).text('Play'); update(f); },
    onSpeedChange: (s) => { state.speed = s; }
  });

  CommonUtils.createMethodToggles('#methods-linear', METHODS_ALL, COLORS,
    (method, visible) => { state.visibleMethods[method] = visible; update(state.currentFrame); }
  );

  return { init: async () => { await loadData(); setup(); } };
})();

document.addEventListener('DOMContentLoaded', () => { LinearRaceViz.init(); });
