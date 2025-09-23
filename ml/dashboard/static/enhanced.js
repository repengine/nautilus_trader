// Enhanced Dashboard JavaScript - Progressive Enhancement

(function() {
  'use strict';

  // Feature detection
  const supportsWebSockets = 'WebSocket' in window;
  const supportsLocalStorage = typeof(Storage) !== 'undefined';
  const supportsFetch = 'fetch' in window;

  // Configuration
  const CONFIG = {
    refreshIntervals: {
      services: 15000,
      metrics: 10000,
      events: 5000,
      stores: 30000
    },
    animations: true,
    darkMode: window.matchMedia('(prefers-color-scheme: dark)').matches
  };

  // State management
  const state = {
    lastUpdate: null,
    services: {},
    metrics: {},
    events: [],
    isLoading: false
  };

  // Utility functions
  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => document.querySelectorAll(selector);

  // Add loading states
  function showLoading(element) {
    if (!element) return;
    const original = element.innerHTML;
    element.dataset.original = original;
    element.innerHTML = '<div class="loading"></div>';
  }

  function hideLoading(element) {
    if (!element || !element.dataset.original) return;
    element.innerHTML = element.dataset.original;
    delete element.dataset.original;
  }

  // Enhance status display
  function enhanceStatus() {
    const status = $('#status');
    if (!status) return;

    // Add live indicator
    const indicator = document.createElement('span');
    indicator.className = 'live-indicator';
    indicator.style.cssText = 'display: inline-block; width: 8px; height: 8px; background: #10b981; border-radius: 50%; margin-left: 8px; animation: pulse 2s infinite;';

    if (!status.querySelector('.live-indicator')) {
      status.appendChild(indicator);
    }
  }

  // Format numbers with proper units
  function formatNumber(num, decimals = 2) {
    if (num === null || num === undefined || isNaN(num)) return '—';

    if (num >= 1000000) {
      return (num / 1000000).toFixed(decimals) + 'M';
    } else if (num >= 1000) {
      return (num / 1000).toFixed(decimals) + 'K';
    }
    return num.toFixed(decimals);
  }

  // Format duration
  function formatDuration(ms) {
    if (ms < 1000) return ms.toFixed(0) + 'ms';
    if (ms < 60000) return (ms / 1000).toFixed(1) + 's';
    if (ms < 3600000) return (ms / 60000).toFixed(1) + 'm';
    return (ms / 3600000).toFixed(1) + 'h';
  }

  // Enhance service displays
  function enhanceServices() {
    const services = $$('.row');
    services.forEach((row, index) => {
      // Add staggered animation
      row.style.animationDelay = `${index * 50}ms`;
      row.classList.add('animated');

      // Enhance status icons
      const statusText = row.textContent;
      if (statusText.includes('✅')) {
        row.classList.add('service-healthy');
      } else if (statusText.includes('❌')) {
        row.classList.add('service-unhealthy');
      }
    });
  }

  // Create mini charts for metrics
  function createMiniChart(container, data, color = '#0066cc') {
    if (!container || !data || data.length === 0) return;

    const width = container.offsetWidth || 100;
    const height = 40;
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', width);
    svg.setAttribute('height', height);
    svg.style.display = 'block';
    svg.style.marginTop = '8px';

    // Create path
    const max = Math.max(...data);
    const min = Math.min(...data);
    const range = max - min || 1;

    const points = data.map((value, index) => {
      const x = (index / (data.length - 1)) * width;
      const y = height - ((value - min) / range) * height;
      return `${x},${y}`;
    }).join(' ');

    const polyline = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
    polyline.setAttribute('points', points);
    polyline.setAttribute('fill', 'none');
    polyline.setAttribute('stroke', color);
    polyline.setAttribute('stroke-width', '2');

    svg.appendChild(polyline);
    container.appendChild(svg);
  }

  // Enhance metric displays
  function enhanceMetrics() {
    const summaryItems = $$('.summary-item');

    summaryItems.forEach(item => {
      const value = item.querySelector('span:last-child');
      if (!value) return;

      // Add animation to value changes
      value.style.transition = 'all 0.3s ease';

      // Generate mock historical data
      const mockData = Array.from({length: 10}, () => Math.random() * 100);

      // Add mini sparkline chart
      if (!item.querySelector('svg')) {
        createMiniChart(item, mockData);
      }
    });
  }

  // Add real-time clock
  function updateClock() {
    const now = new Date();
    const timeString = now.toLocaleTimeString();
    const dateString = now.toLocaleDateString();

    // Find or create clock element
    let clock = $('#dashboard-clock');
    if (!clock) {
      clock = document.createElement('div');
      clock.id = 'dashboard-clock';
      clock.style.cssText = 'position: fixed; top: 1rem; right: 1rem; padding: 0.5rem 1rem; background: rgba(255,255,255,0.9); border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); font-size: 0.875rem; z-index: 1000;';
      document.body.appendChild(clock);
    }

    clock.innerHTML = `
      <div style="font-weight: 600;">${timeString}</div>
      <div style="font-size: 0.75rem; color: #6b7280;">${dateString}</div>
    `;
  }

  // Add keyboard shortcuts
  function initKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
      // Ctrl/Cmd + R: Refresh all data
      if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
        e.preventDefault();
        refreshAll();
      }

      // Ctrl/Cmd + D: Toggle dark mode
      if ((e.ctrlKey || e.metaKey) && e.key === 'd') {
        e.preventDefault();
        toggleDarkMode();
      }

      // Escape: Clear filters
      if (e.key === 'Escape') {
        clearFilters();
      }
    });
  }

  // Toggle dark mode
  function toggleDarkMode() {
    CONFIG.darkMode = !CONFIG.darkMode;
    document.documentElement.classList.toggle('dark-mode', CONFIG.darkMode);

    if (supportsLocalStorage) {
      localStorage.setItem('darkMode', CONFIG.darkMode);
    }
  }

  // Clear all filters
  function clearFilters() {
    $$('input').forEach(input => {
      if (input.type === 'text' || input.type === 'search') {
        input.value = '';
      }
    });
  }

  // Refresh all data
  function refreshAll() {
    console.log('Refreshing all data...');

    // Add visual feedback
    const status = $('#status');
    if (status) {
      status.textContent = 'Refreshing...';
      status.classList.add('loading');
    }

    // Simulate refresh (in real app, would call actual refresh functions)
    setTimeout(() => {
      if (status) {
        status.textContent = 'Dashboard ready';
        status.classList.remove('loading');
        status.classList.add('updated');
        setTimeout(() => status.classList.remove('updated'), 1000);
      }
    }, 1000);
  }

  // Add tooltips
  function addTooltip(element, text) {
    if (!element) return;

    element.style.position = 'relative';
    element.setAttribute('data-tooltip', text);
    element.classList.add('has-tooltip');

    element.addEventListener('mouseenter', (e) => {
      const tooltip = document.createElement('div');
      tooltip.className = 'tooltip-popup';
      tooltip.style.cssText = `
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%) translateY(-8px);
        background: rgba(0,0,0,0.8);
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        white-space: nowrap;
        z-index: 1000;
        pointer-events: none;
      `;
      tooltip.textContent = text;
      element.appendChild(tooltip);
    });

    element.addEventListener('mouseleave', (e) => {
      const tooltip = element.querySelector('.tooltip-popup');
      if (tooltip) tooltip.remove();
    });
  }

  // Enhanced event filtering
  function enhanceEventFiltering() {
    const filterInputs = $$('#filterInstrument, #filterSource');

    filterInputs.forEach(input => {
      // Add search icon
      input.style.paddingLeft = '2rem';

      // Add clear button
      const clearBtn = document.createElement('button');
      clearBtn.textContent = '×';
      clearBtn.style.cssText = 'position: absolute; right: 0.5rem; top: 50%; transform: translateY(-50%); background: none; border: none; cursor: pointer; color: #6b7280;';
      clearBtn.onclick = () => {
        input.value = '';
        input.dispatchEvent(new Event('input'));
      };

      const wrapper = document.createElement('div');
      wrapper.style.position = 'relative';
      input.parentNode.insertBefore(wrapper, input);
      wrapper.appendChild(input);
      if (input.value) wrapper.appendChild(clearBtn);

      input.addEventListener('input', () => {
        if (input.value) {
          if (!wrapper.contains(clearBtn)) wrapper.appendChild(clearBtn);
        } else {
          if (wrapper.contains(clearBtn)) clearBtn.remove();
        }
      });
    });
  }

  // Add notification system
  function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.style.cssText = `
      position: fixed;
      top: 4rem;
      right: 1rem;
      padding: 1rem;
      background: white;
      border-radius: 0.5rem;
      box-shadow: 0 4px 6px rgba(0,0,0,0.1);
      z-index: 2000;
      animation: slideIn 0.3s ease;
      max-width: 300px;
    `;

    const colors = {
      success: '#10b981',
      error: '#ef4444',
      warning: '#f59e0b',
      info: '#3b82f6'
    };

    notification.style.borderLeft = `4px solid ${colors[type] || colors.info}`;
    notification.innerHTML = `
      <div style="display: flex; align-items: center; justify-content: space-between;">
        <span>${message}</span>
        <button onclick="this.parentElement.parentElement.remove()" style="background: none; border: none; cursor: pointer; padding: 0; margin-left: 1rem;">×</button>
      </div>
    `;

    document.body.appendChild(notification);

    // Auto-remove after 5 seconds
    setTimeout(() => {
      if (document.body.contains(notification)) {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
      }
    }, 5000);
  }

  // Add CSS animations
  function injectAnimations() {
    const style = document.createElement('style');
    style.textContent = `
      @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
      }

      @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
      }

      @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
      }

      .animated {
        animation: fadeIn 0.5s ease forwards;
      }

      .service-healthy {
        border-left: 3px solid #10b981;
      }

      .service-unhealthy {
        border-left: 3px solid #ef4444;
      }
    `;
    document.head.appendChild(style);
  }

  // Initialize all enhancements
  function init() {
    console.log('Initializing dashboard enhancements...');

    // Inject styles
    injectAnimations();

    // Apply enhancements
    enhanceStatus();
    enhanceServices();
    enhanceMetrics();
    enhanceEventFiltering();

    // Initialize features
    initKeyboardShortcuts();
    updateClock();
    setInterval(updateClock, 1000);

    // Add tooltips to buttons
    $$('button').forEach(btn => {
      const text = btn.textContent;
      if (text.includes('Start')) addTooltip(btn, 'Start service');
      if (text.includes('Stop')) addTooltip(btn, 'Stop service');
      if (text.includes('Restart')) addTooltip(btn, 'Restart service');
      if (text.includes('Refresh')) addTooltip(btn, 'Refresh data');
    });

    // Load preferences
    if (supportsLocalStorage) {
      const savedDarkMode = localStorage.getItem('darkMode');
      if (savedDarkMode !== null) {
        CONFIG.darkMode = savedDarkMode === 'true';
        document.documentElement.classList.toggle('dark-mode', CONFIG.darkMode);
      }
    }

    // Show welcome notification
    showNotification('Dashboard enhancements loaded!', 'success');

    console.log('Dashboard enhancements initialized successfully');
  }

  // Wait for DOM to be ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // Expose API for external use
  window.DashboardEnhanced = {
    showNotification,
    refreshAll,
    toggleDarkMode,
    formatNumber,
    formatDuration,
    state,
    config: CONFIG
  };

})();