/**
 * Common utilities for D3.js visualizations
 */

const CommonUtils = (() => {
  // Format numbers with appropriate precision
  const formatNumber = (num, decimals = 2) => {
    if (typeof num !== 'number') return '—';
    if (Math.abs(num) < 0.001 && num !== 0) {
      return num.toExponential(decimals);
    }
    return num.toFixed(decimals);
  };

  // Create SVG scales
  const createScales = (data, width, height, margins = {}) => {
    const m = { top: 20, right: 30, bottom: 30, left: 60, ...margins };
    const w = width - m.left - m.right;
    const h = height - m.top - m.bottom;

    return {
      x: d3.scaleLinear()
        .domain([d3.min(data.x), d3.max(data.x)])
        .range([0, w]),
      y: d3.scaleLinear()
        .domain([d3.min(data.y), d3.max(data.y)])
        .range([h, 0]),
      margins: m,
      width: w,
      height: h
    };
  };

  // Create playback controls (Play, Pause, Slider, Speed)
  const createPlaybackControls = (container, callbacks) => {
    const controlsDiv = d3.select(container)
      .append('div')
      .attr('class', 'controls');

    // Play/Pause button
    const playButton = controlsDiv.append('button')
      .attr('class', 'play-button active')
      .text('Play')
      .on('click', function() {
        const isPlaying = d3.select(this).classed('active');
        d3.select(this)
          .classed('active', !isPlaying)
          .text(isPlaying ? 'Play' : 'Pause');
        callbacks.onPlayPause && callbacks.onPlayPause(!isPlaying);
      });

    // Frame slider
    const sliderGroup = controlsDiv.append('div')
      .attr('class', 'control-group');

    sliderGroup.append('label').text('Frame:');

    const slider = sliderGroup.append('input')
      .attr('type', 'range')
      .attr('min', 0)
      .attr('max', 100)
      .attr('value', 0)
      .on('input', function() {
        callbacks.onFrameChange && callbacks.onFrameChange(+this.value);
      });

    const frameLabel = sliderGroup.append('span')
      .style('min-width', '50px')
      .text('0/100');

    // Speed selector
    const speedGroup = controlsDiv.append('div')
      .attr('class', 'control-group');

    speedGroup.append('label').text('Speed:');

    speedGroup.selectAll('button')
      .data([{ label: '1x', value: 1 }, { label: '2x', value: 2 }, { label: '5x', value: 5 }])
      .enter()
      .append('button')
      .text(d => d.label)
      .on('click', function(event, d) {
        speedGroup.selectAll('button').classed('active', false);
        d3.select(this).classed('active', true);
        callbacks.onSpeedChange && callbacks.onSpeedChange(d.value);
      });

    return {
      playButton,
      slider,
      frameLabel,
      setMax: (max) => slider.attr('max', max),
      setFrame: (frame, total) => {
        slider.property('value', frame);
        frameLabel.text(`${frame}/${total}`);
      },
      isPlaying: () => playButton.classed('active')
    };
  };

  // Create method toggles
  const createMethodToggles = (container, methods, colors, callback) => {
    const toggleDiv = d3.select(container)
      .append('div')
      .attr('class', 'method-toggle');

    toggleDiv.selectAll('.toggle-item')
      .data(methods)
      .enter()
      .append('div')
      .attr('class', 'toggle-item')
      .html(d => `
        <input type="checkbox" id="toggle-${d}" checked>
        <label for="toggle-${d}">
          <span class="color-indicator" style="background: ${colors[d]}"></span>
          ${d.charAt(0).toUpperCase() + d.slice(1).replace(/_/g, ' ')}
        </label>
      `)
      .select('input')
      .on('change', function() {
        const method = this.id.replace('toggle-', '');
        callback(method, this.checked);
      });

    return {
      isMethodVisible: (method) => d3.select(`#toggle-${method}`).property('checked'),
      setMethodVisible: (method, visible) => {
        d3.select(`#toggle-${method}`).property('checked', visible);
      }
    };
  };

  // Create annotation banner
  const createAnnotationBanner = (container, text) => {
    d3.select(container)
      .append('div')
      .attr('class', 'annotation info')
      .text(text);
  };

  // Axis generators
  const createAxes = () => {
    return {
      axisBottom: (scale) => d3.axisBottom(scale),
      axisLeft: (scale) => d3.axisLeft(scale)
    };
  };

  // Throttled animation loop (~15fps for consistent speed across hardware)
  const TARGET_FPS = 15;
  const TRANSITION_MS = 150;

  const createAnimationLoop = (onTick) => {
    let id = null;
    let last = 0;
    const interval = 1000 / TARGET_FPS;

    const tick = (now) => {
      id = requestAnimationFrame(tick);
      if (now - last < interval) return;
      last = now - ((now - last) % interval);
      if (!onTick()) { stop(); }
    };

    const start = () => { last = performance.now(); id = requestAnimationFrame(tick); };
    const stop = () => { if (id) { cancelAnimationFrame(id); id = null; } };

    return { start, stop };
  };

  return {
    formatNumber,
    createScales,
    createPlaybackControls,
    createMethodToggles,
    createAnnotationBanner,
    createAxes,
    createAnimationLoop,
    TRANSITION_MS
  };
})();

// Detect if running inside an iframe (e.g., Jekyll embed) and apply class
document.addEventListener('DOMContentLoaded', () => {
  try {
    if (window.self !== window.top) {
      document.body.classList.add('is-iframe');
    }
  } catch (e) {
    // Cross-origin iframe security error means it IS in an iframe
    document.body.classList.add('is-iframe');
  }
});
