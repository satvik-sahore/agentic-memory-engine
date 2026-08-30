/**
 * Self-Learning AI Agent: Frontend Client Application Logic
 * Supports Cards View, Interactive Force-Directed GraphRAG Explorer with Zoom/Pan & Tooltips, and Multi-Tier Scopes.
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
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

  // Tooltip & Zoom Elements
  const graphTooltip = document.getElementById('graphTooltip');
  const tooltipBadge = document.getElementById('tooltipBadge');
  const tooltipTitle = document.getElementById('tooltipTitle');
  const tooltipFact = document.getElementById('tooltipFact');
  const btnZoomIn = document.getElementById('btnZoomIn');
  const btnZoomOut = document.getElementById('btnZoomOut');
  const btnZoomReset = document.getElementById('btnZoomReset');

  // State
  let currentUserId = userSelect ? userSelect.value : 'satvik';
  let currentScope = 'all';
  let currentView = 'graph'; // Default to Graph View
  let chatHistory = [];
  let isSearching = false;

  // Graph Simulation State
  let graphData = { nodes: [], edges: [] };
  let graphNodes = [];
  let graphEdges = [];
  let draggedNode = null;
  let hoveredNode = null;
  let isPanning = false;
  let panStartX = 0, panStartY = 0;
  let camera = { x: 0, y: 0, zoom: 1.0 };
  let animationFrameId = null;

  // Initialize
  init();

  function init() {
    checkHealth();
    loadUserKnowledgeGraph(currentUserId);
    bindEvents();
    setupCanvas();
  }

  function bindEvents() {
    // User Switcher
    if (userSelect) {
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
    }

    // View Switcher Tabs
    if (btnViewCards && btnViewGraph) {
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
    }

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
    if (chatForm) {
      chatForm.addEventListener('submit', handleChatSubmit);
    }

    // Refresh Memories
    if (btnRefreshMemories) {
      btnRefreshMemories.addEventListener('click', () => {
        if (searchInput) searchInput.value = '';
        isSearching = false;
        refreshActiveView();
        showToast('🔄 Memory bank refreshed');
      });
    }

    // Clear Memories
    if (btnClearMemories) {
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
    }

    // Semantic Search Input (Debounced)
    if (searchInput) {
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
      if (dbStatusText) {
        if (data.qdrant_connected) {
          dbStatusText.textContent = `Qdrant Connected (${data.collection})`;
        } else {
          dbStatusText.textContent = 'Qdrant Offline';
          dbStatusText.parentElement.style.color = 'var(--accent-rose)';
        }
      }
    } catch (e) {
      if (dbStatusText) dbStatusText.textContent = 'Server Offline';
    }
  }

  // Load All Memories for User (Cards View)
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
      if (memoryList) {
        memoryList.innerHTML = `<div class="empty-state"><p style="color:var(--accent-rose);">Error: ${err.message}</p></div>`;
      }
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
    if (memoryCountBadge) {
      memoryCountBadge.textContent = `${memories.length} ${isSearchMode ? 'matches' : 'facts'}`;
    }

    if (!memoryList) return;

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
          if (isSearching && searchInput) {
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
    if (!chatInput) return;
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
    if (!chatMessages) return;
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
  // 🕸️ Interactive Knowledge Graph Visualizer (HTML5 Canvas Force Physics & Zoom/Pan)
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
      if (memoryCountBadge) {
        memoryCountBadge.textContent = `${graphData.nodes.length} nodes`;
      }
      initGraphSimulation(graphData);
    } catch (err) {
      showToast(`Graph error: ${err.message}`);
    }
  }

  function ensureCanvasSize() {
    if (!graphContainer || !graphCanvas) return;
    const w = graphContainer.clientWidth || 600;
    const h = graphContainer.clientHeight || 450;
    graphCanvas.width = Math.max(w, 300);
    graphCanvas.height = Math.max(h, 300);
  }

  function setupCanvas() {
    if (!graphCanvas || !graphContainer) return;

    window.addEventListener('resize', ensureCanvasSize);
    ensureCanvasSize();

    // Zoom Controls
    if (btnZoomIn) {
      btnZoomIn.addEventListener('click', () => { camera.zoom = Math.min(camera.zoom * 1.25, 3.0); });
    }
    if (btnZoomOut) {
      btnZoomOut.addEventListener('click', () => { camera.zoom = Math.max(camera.zoom / 1.25, 0.35); });
    }
    if (btnZoomReset) {
      btnZoomReset.addEventListener('click', () => { camera.zoom = 1.0; camera.x = 0; camera.y = 0; });
    }

    // Wheel Zoom
    graphCanvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
      camera.zoom = Math.max(0.35, Math.min(camera.zoom * zoomFactor, 3.0));
    }, { passive: false });

    // Mouse Interaction for Dragging Nodes & Panning Canvas
    graphCanvas.addEventListener('mousedown', (e) => {
      const rect = graphCanvas.getBoundingClientRect();
      const rawX = e.clientX - rect.left;
      const rawY = e.clientY - rect.top;

      // Transform raw screen coord to world coord
      const worldX = (rawX - graphCanvas.width / 2 - camera.x) / camera.zoom + graphCanvas.width / 2;
      const worldY = (rawY - graphCanvas.height / 2 - camera.y) / camera.zoom + graphCanvas.height / 2;

      // Check if clicking a node
      for (let n of graphNodes) {
        if (!isFinite(n.x) || !isFinite(n.y)) continue;
        const dist = Math.hypot(n.x - worldX, n.y - worldY);
        if (dist <= (n.radius || 14) + 10) {
          draggedNode = n;
          break;
        }
      }

      // If not clicking a node, start canvas panning
      if (!draggedNode) {
        isPanning = true;
        panStartX = e.clientX - camera.x;
        panStartY = e.clientY - camera.y;
      }
    });

    window.addEventListener('mousemove', (e) => {
      const rect = graphCanvas.getBoundingClientRect();
      const rawX = e.clientX - rect.left;
      const rawY = e.clientY - rect.top;

      if (draggedNode) {
        const worldX = (rawX - graphCanvas.width / 2 - camera.x) / camera.zoom + graphCanvas.width / 2;
        const worldY = (rawY - graphCanvas.height / 2 - camera.y) / camera.zoom + graphCanvas.height / 2;
        draggedNode.x = worldX;
        draggedNode.y = worldY;
        draggedNode.vx = 0;
        draggedNode.vy = 0;
        hideTooltip();
        return;
      }

      if (isPanning) {
        camera.x = e.clientX - panStartX;
        camera.y = e.clientY - panStartY;
        hideTooltip();
        return;
      }

      // Hover Check for Tooltip & Focus Highlight
      if (rawX >= 0 && rawX <= graphCanvas.width && rawY >= 0 && rawY <= graphCanvas.height) {
        const worldX = (rawX - graphCanvas.width / 2 - camera.x) / camera.zoom + graphCanvas.width / 2;
        const worldY = (rawY - graphCanvas.height / 2 - camera.y) / camera.zoom + graphCanvas.height / 2;

        let found = null;
        for (let n of graphNodes) {
          if (!isFinite(n.x) || !isFinite(n.y)) continue;
          const dist = Math.hypot(n.x - worldX, n.y - worldY);
          if (dist <= (n.radius || 14) + 8) {
            found = n;
            break;
          }
        }
        hoveredNode = found;

        if (hoveredNode) {
          showTooltip(hoveredNode, rawX, rawY);
        } else {
          hideTooltip();
        }
      } else {
        hoveredNode = null;
        hideTooltip();
      }
    });

    window.addEventListener('mouseup', () => {
      draggedNode = null;
      isPanning = false;
    });
  }

  function showTooltip(node, screenX, screenY) {
    if (!graphTooltip) return;
    const cat = (node.type || 'profile').toLowerCase();
    tooltipBadge.textContent = cat;
    tooltipBadge.className = `tooltip-badge category-tag ${cat}`;
    tooltipTitle.textContent = node.id || node.label;
    tooltipFact.textContent = node.full_text || `Knowledge entity linked in long-term memory.`;
    graphTooltip.style.display = 'flex';
    graphTooltip.style.left = `${screenX}px`;
    graphTooltip.style.top = `${screenY}px`;
  }

  function hideTooltip() {
    if (graphTooltip) graphTooltip.style.display = 'none';
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

    // Initialize node positions with wide spread
    const numNodes = data.nodes.length;
    graphNodes = data.nodes.map((n, i) => {
      const angle = (i / Math.max(numNodes, 1)) * Math.PI * 2;
      const r = n.type === 'user' ? 0 : 120 + (i % 4) * 45;
      const initX = centerX + Math.cos(angle) * r;
      const initY = centerY + Math.sin(angle) * r;

      return {
        ...n,
        x: isFinite(initX) ? initX : centerX + (i * 25),
        y: isFinite(initY) ? initY : centerY + (i * 25),
        vx: 0,
        vy: 0,
        radius: n.type === 'user' ? 18 : 12,
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

      // 2. Compute Forces with Wide Breathing Room
      for (let i = 0; i < graphNodes.length; i++) {
        for (let j = i + 1; j < graphNodes.length; j++) {
          const n1 = graphNodes[i];
          const n2 = graphNodes[j];
          const dx = (n2.x - n1.x) || (Math.random() * 2 - 1);
          const dy = (n2.y - n1.y) || (Math.random() * 2 - 1);
          const dist = Math.max(Math.hypot(dx, dy), 12);
          const force = Math.min(14000 / (dist * dist), 18);
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;

          if (n1 !== draggedNode) { n1.vx -= fx; n1.vy -= fy; }
          if (n2 !== draggedNode) { n2.vx += fx; n2.vy += fy; }
        }
      }

      // Spring attraction on edges (Hooke's Law - Relaxed Distance 190px)
      for (let edge of graphEdges) {
        const s = edge.sourceNode;
        const t = edge.targetNode;
        if (!s || !t) continue;
        const dx = (t.x - s.x) || 1;
        const dy = (t.y - s.y) || 1;
        const dist = Math.max(Math.hypot(dx, dy), 1);
        const desiredDist = 190;
        const springForce = (dist - desiredDist) * 0.025;
        const fx = (dx / dist) * springForce;
        const fy = (dy / dist) * springForce;

        if (s !== draggedNode) { s.vx += fx; s.vy += fy; }
        if (t !== draggedNode) { t.vx -= fx; t.vy -= fy; }
      }

      // Gentle Centering Gravity & Damping
      for (let n of graphNodes) {
        if (n !== draggedNode) {
          n.vx = (n.vx + (centerX - n.x) * 0.008) * 0.84;
          n.vy = (n.vy + (centerY - n.y) * 0.008) * 0.84;

          if (isFinite(n.vx)) n.x += n.vx;
          if (isFinite(n.vy)) n.y += n.vy;
        }
      }

      // 3. Render Canvas with Camera Transform (Zoom & Pan)
      ctx.save();
      ctx.translate(width / 2 + camera.x, height / 2 + camera.y);
      ctx.scale(camera.zoom, camera.zoom);
      ctx.translate(-width / 2, -height / 2);

      // Connected Nodes Set for Focus Highlighting
      const activeNodeIds = new Set();
      if (hoveredNode) {
        activeNodeIds.add(hoveredNode.id);
        for (let edge of graphEdges) {
          if (edge.sourceNode.id === hoveredNode.id) activeNodeIds.add(edge.targetNode.id);
          if (edge.targetNode.id === hoveredNode.id) activeNodeIds.add(edge.sourceNode.id);
        }
      }

      // Draw Edges
      for (let edge of graphEdges) {
        const s = edge.sourceNode;
        const t = edge.targetNode;
        if (!s || !t || !isFinite(s.x) || !isFinite(s.y) || !isFinite(t.x) || !isFinite(t.y)) continue;

        const isHighlighted = hoveredNode && (s.id === hoveredNode.id || t.id === hoveredNode.id);
        const isDimmed = hoveredNode && !isHighlighted;

        ctx.save();
        ctx.globalAlpha = isDimmed ? 0.08 : (isHighlighted ? 0.9 : 0.3);
        ctx.beginPath();
        ctx.strokeStyle = isHighlighted ? '#a5b4fc' : 'rgba(99, 102, 241, 0.4)';
        ctx.lineWidth = isHighlighted ? 2.5 : 1.2;
        ctx.moveTo(s.x, s.y);
        ctx.lineTo(t.x, t.y);
        ctx.stroke();

        // Edge Labels: ONLY draw when highlighted on hover for clean uncluttered aesthetic!
        if (isHighlighted && edge.label) {
          const midX = (s.x + t.x) / 2;
          const midY = (s.y + t.y) / 2;

          ctx.font = '10px JetBrains Mono, monospace';
          const textMetrics = ctx.measureText(edge.label);
          const bgW = textMetrics.width + 10;
          const bgH = 16;

          ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
          ctx.beginPath();
          ctx.roundRect(midX - bgW / 2, midY - bgH / 2, bgW, bgH, 4);
          ctx.fill();
          ctx.strokeStyle = 'rgba(99, 102, 241, 0.5)';
          ctx.lineWidth = 1;
          ctx.stroke();

          ctx.fillStyle = '#e2e8f0';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(edge.label, midX, midY);
        }
        ctx.restore();
      }

      // Draw Nodes
      for (let n of graphNodes) {
        if (!isFinite(n.x) || !isFinite(n.y)) continue;

        const isHovered = hoveredNode && n.id === hoveredNode.id;
        const isConnected = hoveredNode && activeNodeIds.has(n.id);
        const isDimmed = hoveredNode && !isConnected;

        const rad = Math.max(n.radius || 12, 8);
        const color = n.color || '#6366f1';

        ctx.save();
        ctx.globalAlpha = isDimmed ? 0.15 : 1.0;

        // Glowing Halo
        try {
          const haloSize = isHovered ? rad * 2.4 : rad * 1.6;
          const grad = ctx.createRadialGradient(n.x, n.y, rad * 0.4, n.x, n.y, haloSize);
          grad.addColorStop(0, color);
          grad.addColorStop(1, 'rgba(0,0,0,0)');
          ctx.beginPath();
          ctx.fillStyle = grad;
          ctx.arc(n.x, n.y, haloSize, 0, Math.PI * 2);
          ctx.fill();
        } catch (e) {}

        // Node Circle
        ctx.beginPath();
        ctx.fillStyle = color;
        ctx.arc(n.x, n.y, rad, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = isHovered ? '#ffffff' : 'rgba(255, 255, 255, 0.7)';
        ctx.lineWidth = isHovered ? 2.5 : 1.5;
        ctx.stroke();

        // Crisp Label Pill
        const labelText = n.label || '';
        if (labelText) {
          ctx.font = isHovered ? 'bold 11px Outfit, sans-serif' : '10px Outfit, sans-serif';
          const labelWidth = ctx.measureText(labelText).width;
          const pillW = labelWidth + 8;
          const pillH = 15;
          const pillY = n.y + rad + 4;

          ctx.fillStyle = 'rgba(10, 14, 23, 0.8)';
          ctx.beginPath();
          ctx.roundRect(n.x - pillW / 2, pillY, pillW, pillH, 4);
          ctx.fill();

          ctx.fillStyle = isHovered ? '#ffffff' : '#cbd5e1';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(labelText, n.x, pillY + pillH / 2);
        }

        ctx.restore();
      }

      ctx.restore();

      animationFrameId = requestAnimationFrame(step);
    }

    step();
  }

  // Toast Notification Helper
  function showToast(message) {
    if (!toastContainer) return;
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
