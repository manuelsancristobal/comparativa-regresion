/**
 * Logistic Regression Visualization - Growing Data
 * Scatter points appear progressively with entrance animation.
 * Responsive SVGs with viewBox. Throttled animation loop.
 */

const LogisticRaceViz = (() => {
  let data = null;
  let state = {
    currentFrame: 0,
    isPlaying: false,
    speed: 1,
    transitionMs: 0,
    visibleMethods: { gradient_descent: true, newton_raphson: true, sklearn: true }
  };

  const SVG_WIDTH = 700;
  const SVG_HEIGHT = 500;
  const SMALL_W = 340;
  const SMALL_H = 300;
  const COLORS = {
    gradient_descent: '#d62728',
    newton_raphson: '#1f77b4',
    sklearn: '#2ca02c'
  };
  const CLASS_COLORS = { 1: '#4a90d9', 0: '#e8943a' };
  const METHODS = ['gradient_descent', 'newton_raphson'];
  const METHODS_ALL = ['gradient_descent', 'newton_raphson', 'sklearn'];
  const METHOD_LABELS = { gradient_descent: 'GD', newton_raphson: 'Newton', sklearn: 'SK' };

  let els = {};
  let sortedIndices = [];
  let animLoop = null;

  const loadData = async () => {
    const response = await fetch('assets/data/logistic_frames.json');
    data = await response.json();

    sortedIndices = data.scatter.x_medinc.map((x, i) => i);
    sortedIndices.sort((a, b) => {
      const distA = Math.sqrt(data.scatter.x_medinc[a] ** 2 + data.scatter.x_houseage[a] ** 2);
      const distB = Math.sqrt(data.scatter.x_medinc[b] ** 2 + data.scatter.x_houseage[b] ** 2);
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

    const container = d3.select('#panel-a-logistic').html('');
    container.append('h3').text('MedInc vs HouseAge + Frontera de Decisión');
    const svg = container.append('svg')
      .attr('viewBox', `0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');
    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    const xScale = d3.scaleLinear().domain(d3.extent(data.scatter.x_medinc)).range([0, w]);
    const yScale = d3.scaleLinear().domain(d3.extent(data.scatter.x_houseage)).range([h, 0]);

    g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(xScale));
    g.append('g').call(d3.axisLeft(yScale));

    const scatterGroup = g.append('g').attr('class', 'scatter-group');

    // Frontera sklearn fija (referencia dashed)
    const skRef = data.sklearn;
    const skW = skRef.weights;
    const skB = skRef.intercept;
    const skPts = [];
    const xDom = xScale.domain();
    const yDom = yScale.domain();
    for (let x = xDom[0]; x <= xDom[1]; x += (xDom[1] - xDom[0]) / 50) {
      const y = -(skW[0] * x + skB) / (skW[1] + 1e-10);
      if (y >= yDom[0] && y <= yDom[1]) skPts.push({ x, y });
    }
    const skBoundary = g.append('path')
      .attr('class', 'reference-line')
      .attr('fill', 'none').attr('stroke', COLORS.sklearn)
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '6,4')
      .attr('opacity', 0.6);
    if (skPts.length > 1) {
      skBoundary.attr('d', d3.line().x(d => xScale(d.x)).y(d => yScale(d.y))(skPts));
    }

    const boundaryPaths = {};
    METHODS.forEach(method => {
      boundaryPaths[method] = g.append('path')
        .attr('fill', 'none').attr('stroke', COLORS[method])
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '5,3')
        .attr('opacity', 0.8);
    });

    const counter = g.append('text')
      .attr('x', w - 10).attr('y', 15)
      .attr('text-anchor', 'end')
      .attr('font-size', '13px')
      .attr('fill', '#555');

    els.panelA = { xScale, yScale, scatterGroup, boundaryPaths, skBoundary, counter, w, h };
  };

  const setupPanelB = () => {
    const m = { top: 20, right: 20, bottom: 40, left: 60 };
    const w = SMALL_W - m.left - m.right;
    const h = SMALL_H - m.top - m.bottom;

    const container = d3.select('#panel-b-logistic').html('');
    container.append('h3').text('Carrera de Log-Loss');
    const svg = container.append('svg')
      .attr('viewBox', `0 0 ${SMALL_W} ${SMALL_H}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');
    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    const xScale = d3.scaleBand()
      .domain(METHODS_ALL.map(m => METHOD_LABELS[m]))
      .range([0, w]).padding(0.3);
    g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(xScale));
    const yAxisG = g.append('g');

    const bars = {};
    const barLabels = {};
    METHODS_ALL.forEach(method => {
      const label = METHOD_LABELS[method];
      bars[method] = g.append('rect')
        .attr('x', xScale(label)).attr('width', xScale.bandwidth())
        .attr('fill', COLORS[method]).attr('opacity', 0.7)
        .attr('y', h).attr('height', 0);
      barLabels[method] = g.append('text')
        .attr('x', xScale(label) + xScale.bandwidth() / 2)
        .attr('text-anchor', 'middle').attr('font-size', '11px');
    });

    els.panelB = { yAxisG, bars, barLabels, w, h };
  };

  const setupPanelC = () => {
    const m = { top: 20, right: 20, bottom: 40, left: 60 };
    const w = SMALL_W - m.left - m.right;
    const h = SMALL_H - m.top - m.bottom;

    const container = d3.select('#panel-c-logistic').html('');
    container.append('h3').text('Curva ROC (AUC)');
    const svg = container.append('svg')
      .attr('viewBox', `0 0 ${SMALL_W} ${SMALL_H}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');
    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    const roc = data.roc_curve;
    const xScale = d3.scaleLinear().domain([0, 1]).range([0, w]);
    const yScale = d3.scaleLinear().domain([0, 1]).range([h, 0]);

    g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(xScale).ticks(5));
    g.append('g').call(d3.axisLeft(yScale).ticks(5));

    g.append('line')
      .attr('x1', xScale(0)).attr('y1', yScale(0))
      .attr('x2', xScale(1)).attr('y2', yScale(1))
      .attr('stroke', '#ccc').attr('stroke-dasharray', '4,4');

    if (roc.fpr && roc.tpr) {
      const rocPoints = roc.fpr.map((fpr, i) => ({ x: fpr, y: roc.tpr[i] }));
      g.append('path')
        .attr('d', d3.line().x(d => xScale(d.x)).y(d => yScale(d.y))(rocPoints))
        .attr('fill', 'none').attr('stroke', COLORS.sklearn).attr('stroke-width', 2);
    }

    g.append('text')
      .attr('x', w - 10).attr('y', 20)
      .attr('text-anchor', 'end').attr('font-size', '12px')
      .text(`AUC = ${CommonUtils.formatNumber(roc.auc, 3)}`);
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
    updateBoundaries(frameData);
    updateLogLoss(frameData);

    els.panelA.counter.text(`n = ${frameData.n_points}`);
    controls.setFrame(frame, data.frames.length - 1);
  };

  const updateScatter = (nScatter) => {
    const { scatterGroup, xScale, yScale } = els.panelA;
    const nShow = Math.min(nScatter, data.scatter.x_medinc.length);

    const allPoints = sortedIndices.map((origIdx, sortPos) => ({
      x: data.scatter.x_medinc[origIdx],
      y: data.scatter.x_houseage[origIdx],
      label: data.scatter.y[origIdx],
      idx: origIdx,
      active: sortPos < nShow
    }));

    const sel = scatterGroup.selectAll('.scatter-point').data(allPoints, d => d.idx);

    sel.enter()
      .append('circle')
      .attr('class', 'scatter-point')
      .attr('cx', d => xScale(d.x))
      .attr('cy', d => yScale(d.y))
      .attr('r', 0)
      .attr('fill', '#ccc')
      .attr('opacity', 0)
      .transition().duration(300)
      .attr('r', 4)
      .attr('opacity', 0.3);

    applyTransition(scatterGroup.selectAll('.scatter-point'))
      .attr('fill', d => d.active ? CLASS_COLORS[d.label] : '#ccc')
      .attr('opacity', d => d.active ? 0.7 : 0.3);
  };

  const updateBoundaries = (frameData) => {
    const { xScale, yScale, boundaryPaths, skBoundary } = els.panelA;
    const xDomain = xScale.domain();
    const yDomain = yScale.domain();

    // Toggle visibilidad frontera sklearn de referencia
    skBoundary.attr('visibility', state.visibleMethods.sklearn !== false ? 'visible' : 'hidden');

    METHODS.forEach(method => {
      const visible = state.visibleMethods[method];
      const md = frameData[method];

      if (!visible || !md || !md.weights || md.weights.length < 2) {
        boundaryPaths[method].attr('visibility', 'hidden');
        return;
      }

      const w = md.weights;
      const b = md.intercept || 0;
      const pts = [];
      for (let x = xDomain[0]; x <= xDomain[1]; x += (xDomain[1] - xDomain[0]) / 50) {
        const y = -(w[0] * x + b) / (w[1] + 1e-10);
        if (y >= yDomain[0] && y <= yDomain[1]) pts.push({ x, y });
      }

      if (pts.length > 1) {
        const pathD = d3.line().x(d => xScale(d.x)).y(d => yScale(d.y))(pts);
        applyTransition(boundaryPaths[method])
          .attr('d', pathD)
          .attr('visibility', 'visible');
      } else {
        boundaryPaths[method].attr('visibility', 'hidden');
      }
    });
  };

  const updateLogLoss = (frameData) => {
    const { bars, barLabels, yAxisG, h } = els.panelB;

    // Construir mapa: frames para métodos iterativos, fijo para sklearn
    const lossMap = {};
    METHODS.forEach(m => {
      if (state.visibleMethods[m] && frameData[m]) {
        lossMap[m] = frameData[m].log_loss_train || 0;
      }
    });
    if (state.visibleMethods.sklearn !== false) {
      lossMap.sklearn = data.sklearn.log_loss_train || 0;
    }

    const vals = Object.values(lossMap);
    if (vals.length === 0) return;

    const yScale = d3.scaleLinear().domain([0, d3.max(vals) * 1.1]).range([h, 0]);
    yAxisG.call(d3.axisLeft(yScale).ticks(5));

    METHODS_ALL.forEach(method => {
      const show = lossMap[method] !== undefined;
      bars[method].attr('visibility', show ? 'visible' : 'hidden');
      barLabels[method].attr('visibility', show ? 'visible' : 'hidden');
      if (!show) return;
      const loss = lossMap[method];
      const bY = yScale(loss);
      applyTransition(bars[method]).attr('y', bY).attr('height', h - bY);
      applyTransition(barLabels[method]).attr('y', bY - 5).text(CommonUtils.formatNumber(loss, 3));
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

  const controls = CommonUtils.createPlaybackControls('#controls-logistic', {
    onPlayPause: (p) => p ? startAnimation() : stopAnimation(),
    onFrameChange: (f) => { stopAnimation(); state.currentFrame = f; controls.playButton.classed('active', false).text('Play'); update(f); },
    onSpeedChange: (s) => { state.speed = s; }
  });

  METHODS_ALL.forEach(method => {
    d3.select('#methods-logistic').append('div').attr('class', 'toggle-item')
      .html(`<input type="checkbox" id="toggle-log-${method}" checked>
        <label for="toggle-log-${method}"><span class="color-indicator" style="background: ${COLORS[method]}"></span> ${method.replace(/_/g, ' ')}</label>`)
      .select('input').on('change', function() { state.visibleMethods[method] = this.checked; update(state.currentFrame); });
  });

  return { init: async () => { await loadData(); setup(); } };
})();

document.addEventListener('DOMContentLoaded', () => { LogisticRaceViz.init(); });
