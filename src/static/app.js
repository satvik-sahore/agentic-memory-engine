/**
 * Self-Learning AI Agent: Frontend Client Application Logic
 * Supports Cards View, Interactive Force-Directed GraphRAG Explorer, and Multi-Tier Scopes.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const userSelect = document.getElementById('userSelect');
  const chatMessages = document.getElementById('chatMessages');
  const chatForm = document.getElementById('chatForm');
  const chatInput = document.getElementById('chatInput');
  const memoryList = document.getElementById('memoryList');
  const searchInput = document.getElementById('searchInput');
  const searchContainer = document.getElementById('searchContainer');
  const memoryCountBadge = document.getElementById('memoryCountBadge');
  const btnRefreshMemories = document.getElementById('btnRefreshMemories');
  const btnClearMemories = document.getElementById('btnClearMemories');
  const dbStatusText = document.getElementById('dbStatusText');
  const toastContainer = document.getElementById('toastContainer');

  // View Switcher & Scope Elements
  const btnViewCards = document.getElementById('btnViewCards');
  const btnViewGraph = document.getElementById('btnViewGraph');
  const graphContainer = document.getElementById('graphContainer');
  const graphCanvas = document.getElementById('graphCanvas');
  const scopePills = document.querySelectorAll('.scope-pill');

  let currentUserId = userSelect.value;
  let currentScope = 'all';
  let currentView = 'cards'; // 'cards' | 'graph'
  let chatHistory = [];
  let isSearching = false;

  // Graph Simulation State
  let graphData = { nodes: [], edges: [] };
  let graphNodes = [];
  let graphEdges = [];
  let draggedNode = null;
  let animationFrameId = null;

  // Initialize
  init();

  function init() {
    checkHealth();
    loadUserMemories(currentUserId);
    bindEvents();
    setupCanvas();
  }

  function bindEvents() {
    // User Switcher
    userSelect.addEventListener('change', (e) => {
      if (e.target.value === 'custom') {
        const customId = prompt('Enter a custom User ID:');
        if (customId && customId.trim()) {
          const newOpt = document.createElement('option');
          newOpt.value = customId.trim().toLowerCase();
          newOpt.textContent = customId.trim();
          newOpt.selected = true;
          userSelect.appendChild(newOpt);
          currentUserId = newOpt.value;
        } else {
          userSelect.value = currentUserId;
          return;
        }
      } else {
        currentUserId = e.target.value;
      }
      chatHistory = [];
      chatMessages.innerHTML = `
        <div class="message assistant">
          <div class="message-bubble">
            Switched to user <strong>${escapeHtml(currentUserId)}</strong>. Ready to assist and learn!
          </div>
        </div>
      `;
      refreshActiveView();
      showToast(`👤 Active user set to: ${currentUserId}`);
    });

    // View Switcher Tabs
    btnViewCards.addEventListener('click', () => {
      currentView = 'cards';
      btnViewCards.classList.add('active');
      btnViewGraph.classList.remove('active');
      memoryList.style.display = 'flex';
      searchContainer.style.display = 'block';
      graphContainer.style.display = 'none';
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
      loadUserMemories(currentUserId);
    });

    btnViewGraph.addEventListener('click', () => {
      currentView = 'graph';
      btnViewGraph.classList.add('active');
      btnViewCards.classList.remove('active');
      memoryList.style.display = 'none';
      searchContainer.style.display = 'none';
      graphContainer.style.display = 'flex';
      loadUserKnowledgeGraph(currentUserId);
    });

    // Scope Filter Pills
    scopePills.forEach((pill) => {
      pill.addEventListener('click', () => {
        scopePills.forEach((p) => p.classList.remove('active'));
        pill.classList.add('active');
        currentScope = pill.getAttribute('data-scope');
        refreshActiveView();
        showToast(`🗂️ Scope filtered to: ${currentScope.toUpperCase()}`);
      });
    });

    // Chat Form Submit
    chatForm.addEventListener('submit', handleChatSubmit);

    // Refresh Memories
    btnRefreshMemories.addEventListener('click', () => {
      searchInput.value = '';
      isSearching = false;
      refreshActiveView();
      showToast('🔄 Memory bank refreshed');
    });

    // Clear Memories
    btnClearMemories.addEventListener('click', async () => {
      if (!confirm(`Are you sure you want to delete all stored memories for '${currentUserId}'?`)) {
        return;
      }
      try {
        const res = await fetch(`/v1/memories/user/${encodeURIComponent(currentUserId)}`, {
          method: 'DELETE',
        });
        if (res.ok) {
          showToast(`🗑️ Cleared all memories for ${currentUserId}`);
          refreshActiveView();
        }
      } catch (err) {
        showToast(`❌ Error clearing memories: ${err.message}`);
      }
    });

    // Semantic Search Input (Debounced)
    let searchTimeout;
    searchInput.addEventListener('input', (e) => {
      clearTimeout(searchTimeout);
      const q = e.target.value.trim();
      if (!q) {
        isSearching = false;
        loadUserMemories(currentUserId);
        return;
      }
      searchTimeout = setTimeout(() => {
        performSemanticSearch(q);
      }, 350);
    });
  }

  function refreshActiveView() {
    if (currentView === 'cards') {
      loadUserMemories(currentUserId);
    } else {
      loadUserKnowledgeGraph(currentUserId);
    }
  }

  // Health check
  async function checkHealth() {
    try {
      const res = await fetch('/healthz');
      const data = await res.json();
      if (data.qdrant_connected) {
        dbStatusText.textContent = `Qdrant Connected (${data.collection})`;
      } else {
        dbStatusText.textContent = 'Qdrant Offline';
        dbStatusText.parentElement.style.color = 'var(--accent-rose)';
      }
    } catch (e) {
      dbStatusText.textContent = 'Server Offline';
    }
  }

  // Load All Memories for User
  async function loadUserMemories(userId) {
    try {
      let url = `/v1/memories/user/${encodeURIComponent(userId)}`;
      if (currentScope !== 'all') {
        url += `?scope=${encodeURIComponent(currentScope)}`;
      }
      const res = await fetch(url);
      if (!res.ok) throw new Error('Failed to load memories');
      const memories = await res.json();
      renderMemoryCards(memories);
    } catch (err) {
      memoryList.innerHTML = `<div class="empty-state"><p style="color:var(--accent-rose);">Error: ${err.message}</p></div>`;
    }
  }

  // Semantic Search
  async function performSemanticSearch(query) {
    isSearching = true;
    try {
      let url = `/v1/memories/search?user_id=${encodeURIComponent(currentUserId)}&query=${encodeURIComponent(query)}&limit=10&score_threshold=0.3`;
      if (currentScope !== 'all') {
        url += `&scope=${encodeURIComponent(currentScope)}`;
      }
      const res = await fetch(url);
      if (!res.ok) throw new Error('Search failed');
      const data = await res.json();
      renderMemoryCards(data.results, true);
    } catch (err) {
      showToast(`Search error: ${err.message}`);
    }
  }

  // Render Memory Cards
  function renderMemoryCards(memories, isSearchMode = false) {
    memoryCountBadge.textContent = `${memories.length} ${isSearchMode ? 'matches' : 'facts'}`;

    if (!memories || memories.length === 0) {
      memoryList.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">${isSearchMode ? '🔍' : '🌱'}</div>
          <p>${isSearchMode ? 'No matching memories found for this query.' : 'No memories stored for this user yet.'}</p>
          <p style="font-size: 0.75rem; color: var(--text-dim); margin-top: 0.3rem;">
            ${isSearchMode ? 'Try asking in a different way or clear search.' : 'Chat on the left to teach the agent new facts!'}
          </p>
        </div>
      `;
      return;
    }

    memoryList.innerHTML = memories.map((m) => {
      const cat = m.category ? m.category.toLowerCase() : 'other';
      const scopeTag = (m.scope || 'user').toUpperCase();
      const freshness = m.freshness_label || '🔥 Fresh';
      let simHtml = '';

      if (isSearchMode && m.composite_score !== undefined && m.composite_score !== null) {
        const compPct = Math.min(Math.round(m.composite_score * 100), 100);
        const simPct = m.score ? Math.min(Math.round(m.score * 100), 100) : compPct;
        simHtml = `
          <div class="sim-bar-container">
            <div class="sim-bar-bg">
              <div class="sim-bar-fill" style="width: ${compPct}%;"></div>
            </div>
            <span class="sim-score" title="Blended Composite Rank (Vector Match ${simPct}% + Recency)">${compPct}% rank</span>
          </div>
        `;
      }

      return `
        <div class="memory-card" data-id="${m.id}">
          <div class="memory-header">
            <div style="display: flex; align-items: center; gap: 0.4rem; flex-wrap: wrap;">
              <span class="category-tag ${cat}">${cat}</span>
              <span class="category-tag" style="background: rgba(255,255,255,0.05); color: var(--text-dim);">${scopeTag}</span>
              <span class="freshness-badge">${freshness}</span>
            </div>
            <button class="btn-delete-mem" onclick="window.deleteSingleMemory('${m.id}')" title="Delete fact">
              ✕
            </button>
          </div>
          <div class="memory-text">${escapeHtml(m.fact)}</div>
          ${simHtml}
          <div class="memory-meta">
            <span>ID: ${m.id.slice(0, 8)}...</span>
            <span>Recalled ${m.access_count || 0}x</span>
          </div>
        </div>
      `;
    }).join('');
  }

  // Global delete handler
  window.deleteSingleMemory = async function(memoryId) {
    try {
      const res = await fetch(`/v1/memories/${encodeURIComponent(memoryId)}`, { method: 'DELETE' });
      if (res.ok) {
        showToast('🗑️ Memory record removed.');
        if (currentView === 'cards') {
          if (isSearching) {
            performSemanticSearch(searchInput.value);
          } else {
            loadUserMemories(currentUserId);
          }
        } else {
          loadUserKnowledgeGraph(currentUserId);
        }
      }
    } catch (e) {
      showToast(`Delete failed: ${e.message}`);
    }
  };

  // Handle Chat Submit
  async function handleChatSubmit(e) {
    e.preventDefault();
    const message = chatInput.value.trim();
    if (!message) return;

    chatInput.value = '';

    // Append user message bubble
    appendMessageBubble('user', message);

    // Append temporary typing indicator
    const typingId = 'typing-' + Date.now();
    const typingElem = document.createElement('div');
    typingElem.id = typingId;
    typingElem.className = 'message assistant';
    typingElem.innerHTML = `<div class="message-bubble" style="opacity: 0.7;">Thinking & recalling memories...</div>`;
    chatMessages.appendChild(typingElem);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
      const res = await fetch('/v1/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: currentUserId,
          message: message,
          scope: currentScope === 'all' ? 'user' : currentScope,
          history: chatHistory,
        }),
      });

      document.getElementById(typingId)?.remove();

      if (!res.ok) {
        throw new Error(`Server returned ${res.status}`);
      }

      const data = await res.json();

      // Append assistant message with recalled memories accordion
      appendMessageBubble('assistant', data.reply, data.recalled_memories);

      // Record to history
      chatHistory.push({ role: 'user', content: message });
      chatHistory.push({ role: 'assistant', content: data.reply });

      if (data.async_job_id) {
        showToast(`⚡ Extracting & reconciling in background...`);
        pollJobStatus(data.async_job_id);
      }

    } catch (err) {
      document.getElementById(typingId)?.remove();
      appendMessageBubble('assistant', `⚠️ Sorry, I encountered an error: ${err.message}`);
    }
  }

  // Poll Async Ingestion Job Status
  async function pollJobStatus(jobId, attempts = 0) {
    if (attempts >= 15) return; // Stop after 15 attempts (9s)

    try {
      const res = await fetch(`/v1/memories/jobs/${encodeURIComponent(jobId)}`);
      if (!res.ok) return;
      const data = await res.json();

      if (data.status === 'completed') {
        const ops = data.operations || [];
        const validOps = ops.filter(o => o.operation !== 'NOOP');
        if (validOps.length > 0) {
          validOps.forEach(op => {
            showToast(`⚡ [${op.operation}] ${op.fact}`);
          });
        }
        // Auto-refresh active view (Cards or Graph) instantly
        refreshActiveView();
      } else {
        // Poll again in 600ms
        setTimeout(() => pollJobStatus(jobId, attempts + 1), 600);
      }
    } catch (e) {
      // Ignore polling transient network errors
    }
  }

  // Append Message Bubble to Chat
  function appendMessageBubble(role, text, recalledMemories = []) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;

    let recalledHtml = '';
    if (recalledMemories && recalledMemories.length > 0) {
      recalledHtml = `
        <div class="recalled-box">
          <div class="recalled-header" onclick="this.nextElementSibling.style.display = this.nextElementSibling.style.display === 'none' ? 'flex' : 'none'">
            <span>🧠</span>
            <span>Recalled ${recalledMemories.length} relevant memories</span>
            <span style="font-size: 0.65rem; margin-left: auto;">▼</span>
          </div>
          <div class="recalled-items" style="display: none;">
            ${recalledMemories.map(m => `<span class="recalled-item">• ${escapeHtml(m.fact)}</span>`).join('')}
          </div>
        </div>
      `;
    }

    msgDiv.innerHTML = `
      <div class="message-bubble">
        ${escapeHtml(text)}
      </div>
      ${recalledHtml}
    `;

    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  // ============================================================================
  // 🕸️ Interactive Knowledge Graph Visualizer (HTML5 Canvas Force Physics)
  // ============================================================================

  async function loadUserKnowledgeGraph(userId) {
    try {
      let url = `/v1/graph/${encodeURIComponent(userId)}`;
      if (currentScope !== 'all') {
        url += `?scope=${encodeURIComponent(currentScope)}`;
      }
      const res = await fetch(url);
      if (!res.ok) throw new Error('Failed to load knowledge graph');
      graphData = await res.json();
      memoryCountBadge.textContent = `${graphData.nodes.length} nodes`;
      initGraphSimulation(graphData);
    } catch (err) {
      showToast(`Graph error: ${err.message}`);
    }
  }

  function ensureCanvasSize() {
    const w = graphContainer.clientWidth || 600;
    const h = graphContainer.clientHeight || 450;
    graphCanvas.width = Math.max(w, 300);
    graphCanvas.height = Math.max(h, 300);
  }

  function setupCanvas() {
    window.addEventListener('resize', ensureCanvasSize);
    ensureCanvasSize();

    // Mouse Interaction for Dragging Nodes
    graphCanvas.addEventListener('mousedown', (e) => {
      const rect = graphCanvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      for (let n of graphNodes) {
        if (!isFinite(n.x) || !isFinite(n.y)) continue;
        const dist = Math.hypot(n.x - x, n.y - y);
        if (dist <= (n.radius || 14) + 8) {
          draggedNode = n;
          break;
        }
      }
    });

    window.addEventListener('mousemove', (e) => {
      if (!draggedNode) return;
      const rect = graphCanvas.getBoundingClientRect();
      draggedNode.x = e.clientX - rect.left;
      draggedNode.y = e.clientY - rect.top;
      draggedNode.vx = 0;
      draggedNode.vy = 0;
    });

    window.addEventListener('mouseup', () => {
      draggedNode = null;
    });
  }

  function initGraphSimulation(data) {
    if (animationFrameId) cancelAnimationFrame(animationFrameId);

    ensureCanvasSize();
    const width = Math.max(graphCanvas.width, 300);
    const height = Math.max(graphCanvas.height, 300);
    const centerX = width / 2;
    const centerY = height / 2;

    if (!data.nodes || data.nodes.length === 0) {
      graphNodes = [];
      graphEdges = [];
      const ctx = graphCanvas.getContext('2d');
      ctx.clearRect(0, 0, width, height);
      return;
    }

    // Initialize node positions in a radial circle with non-zero offsets
    const numNodes = data.nodes.length;
    graphNodes = data.nodes.map((n, i) => {
      const angle = (i / Math.max(numNodes, 1)) * Math.PI * 2 + (Math.random() * 0.1);
      const r = n.type === 'user' ? 0 : 80 + (i % 3) * 35;
      const initX = centerX + Math.cos(angle) * r;
      const initY = centerY + Math.sin(angle) * r;

      return {
        ...n,
        x: isFinite(initX) ? initX : centerX + (i * 20),
        y: isFinite(initY) ? initY : centerY + (i * 20),
        vx: 0,
        vy: 0,
        radius: Math.max(Number(n.size) || 14, 8),
      };
    });

    // Map Edges
    const nodeById = new Map(graphNodes.map(n => [n.id, n]));
    graphEdges = (data.edges || []).map(e => ({
      ...e,
      sourceNode: nodeById.get(e.source),
      targetNode: nodeById.get(e.target),
    })).filter(e => e.sourceNode && e.targetNode);

    // Run Physics Animation Loop
    runSimulationLoop();
  }

  function runSimulationLoop() {
    const ctx = graphCanvas.getContext('2d');

    function step() {
      ensureCanvasSize();
      const width = Math.max(graphCanvas.width, 300);
      const height = Math.max(graphCanvas.height, 300);
      const centerX = width / 2;
      const centerY = height / 2;

      // 1. Clear Canvas
      ctx.clearRect(0, 0, width, height);

      // 2. Compute Forces
      // Repulsion between all nodes (Coulomb)
      for (let i = 0; i < graphNodes.length; i++) {
        for (let j = i + 1; j < graphNodes.length; j++) {
          const n1 = graphNodes[i];
          const n2 = graphNodes[j];
          const dx = (n2.x - n1.x) || (Math.random() * 2 - 1);
          const dy = (n2.y - n1.y) || (Math.random() * 2 - 1);
          const dist = Math.max(Math.hypot(dx, dy), 10);
          const force = Math.min(6000 / (dist * dist), 15);
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;

          if (n1 !== draggedNode) { n1.vx -= fx; n1.vy -= fy; }
          if (n2 !== draggedNode) { n2.vx += fx; n2.vy += fy; }
        }
      }

      // Spring attraction on edges
      for (let edge of graphEdges) {
        const s = edge.sourceNode;
        const t = edge.targetNode;
        if (!s || !t) continue;
        const dx = (t.x - s.x) || 1;
        const dy = (t.y - s.y) || 1;
        const dist = Math.max(Math.hypot(dx, dy), 1);
        const desiredDist = 110;
        const springForce = (dist - desiredDist) * 0.03;
        const fx = (dx / dist) * springForce;
        const fy = (dy / dist) * springForce;

        if (s !== draggedNode) { s.vx += fx; s.vy += fy; }
        if (t !== draggedNode) { t.vx -= fx; t.vy -= fy; }
      }

      // Centering Gravity & Position Update
      for (let n of graphNodes) {
        if (n !== draggedNode) {
          n.vx = (n.vx + (centerX - n.x) * 0.01) * 0.82;
          n.vy = (n.vy + (centerY - n.y) * 0.01) * 0.82;

          if (isFinite(n.vx)) n.x += n.vx;
          if (isFinite(n.vy)) n.y += n.vy;

          // Keep within bounds
          n.x = Math.max(n.radius + 10, Math.min(width - n.radius - 10, n.x));
          n.y = Math.max(n.radius + 10, Math.min(height - n.radius - 10, n.y));
        }
      }

      // 3. Draw Edges
      for (let edge of graphEdges) {
        const s = edge.sourceNode;
        const t = edge.targetNode;
        if (!s || !t || !isFinite(s.x) || !isFinite(s.y) || !isFinite(t.x) || !isFinite(t.y)) continue;

        ctx.beginPath();
        ctx.strokeStyle = 'rgba(99, 102, 241, 0.3)';
        ctx.lineWidth = 1.5;
        ctx.moveTo(s.x, s.y);
        ctx.lineTo(t.x, t.y);
        ctx.stroke();

        // Edge Label Tag
        if (edge.label) {
          const midX = (s.x + t.x) / 2;
          const midY = (s.y + t.y) / 2;
          ctx.font = '9px JetBrains Mono, monospace';
          ctx.fillStyle = '#94a3b8';
          ctx.textAlign = 'center';
          ctx.fillText(edge.label, midX, midY - 3);
        }
      }

      // 4. Draw Nodes
      for (let n of graphNodes) {
        if (!isFinite(n.x) || !isFinite(n.y)) continue;

        const rad = Math.max(n.radius || 14, 8);
        const color = n.color || '#6366f1';

        // Glowing Halo with finite checks
        try {
          const grad = ctx.createRadialGradient(n.x, n.y, Math.max(rad * 0.4, 1), n.x, n.y, Math.max(rad * 1.8, 2));
          grad.addColorStop(0, color);
          grad.addColorStop(1, 'rgba(0,0,0,0)');
          ctx.beginPath();
          ctx.fillStyle = grad;
          ctx.arc(n.x, n.y, rad * 1.8, 0, Math.PI * 2);
          ctx.fill();
        } catch (e) {
          // Fallback if gradient error
        }

        // Node Circle
        ctx.beginPath();
        ctx.fillStyle = color;
        ctx.arc(n.x, n.y, rad, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Node Label Text
        ctx.font = '11px Outfit, sans-serif';
        ctx.fillStyle = '#f8fafc';
        ctx.textAlign = 'center';
        ctx.fillText(n.label || '', n.x, n.y + rad + 14);
      }

      animationFrameId = requestAnimationFrame(step);
    }

    step();
  }

  // Toast Notification Helper
  function showToast(message) {
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.innerHTML = `<span>${message}</span>`;
    toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }
});
