/**
 * Self-Learning AI Agent: Frontend Client Application Logic
 */

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const userSelect = document.getElementById('userSelect');
  const chatMessages = document.getElementById('chatMessages');
  const chatForm = document.getElementById('chatForm');
  const chatInput = document.getElementById('chatInput');
  const memoryList = document.getElementById('memoryList');
  const searchInput = document.getElementById('searchInput');
  const memoryCountBadge = document.getElementById('memoryCountBadge');
  const btnRefreshMemories = document.getElementById('btnRefreshMemories');
  const btnClearMemories = document.getElementById('btnClearMemories');
  const dbStatusText = document.getElementById('dbStatusText');
  const toastContainer = document.getElementById('toastContainer');

  let currentUserId = userSelect.value;
  let chatHistory = [];
  let isSearching = false;

  // Initialize
  init();

  function init() {
    checkHealth();
    loadUserMemories(currentUserId);
    bindEvents();
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
      loadUserMemories(currentUserId);
      showToast(`👤 Active user set to: ${currentUserId}`);
    });

    // Chat Form Submit
    chatForm.addEventListener('submit', handleChatSubmit);

    // Refresh Memories
    btnRefreshMemories.addEventListener('click', () => {
      searchInput.value = '';
      isSearching = false;
      loadUserMemories(currentUserId);
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
          loadUserMemories(currentUserId);
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
      const res = await fetch(`/v1/memories/user/${encodeURIComponent(userId)}`);
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
      const res = await fetch(`/v1/memories/search?user_id=${encodeURIComponent(currentUserId)}&query=${encodeURIComponent(query)}&limit=10&score_threshold=0.3`);
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
      let simHtml = '';
      if (isSearchMode && m.score !== undefined && m.score !== null) {
        const pct = Math.min(Math.round(m.score * 100), 100);
        simHtml = `
          <div class="sim-bar-container">
            <div class="sim-bar-bg">
              <div class="sim-bar-fill" style="width: ${pct}%;"></div>
            </div>
            <span class="sim-score">${pct}% match</span>
          </div>
        `;
      }

      return `
        <div class="memory-card" data-id="${m.id}">
          <div class="memory-header">
            <span class="category-tag ${cat}">${cat}</span>
            <button class="btn-delete-mem" onclick="window.deleteSingleMemory('${m.id}')" title="Delete fact">
              ✕
            </button>
          </div>
          <div class="memory-text">${escapeHtml(m.fact)}</div>
          ${simHtml}
          <div class="memory-meta">
            <span>ID: ${m.id.slice(0, 8)}...</span>
            <span>${m.created_at ? new Date(m.created_at).toLocaleDateString() : 'Persisted'}</span>
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
        if (isSearching) {
          performSemanticSearch(searchInput.value);
        } else {
          loadUserMemories(currentUserId);
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
          history: chatHistory,
        }),
      });

      // Remove typing bubble
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

      // Notify of any new operations performed
      if (data.operations_performed && data.operations_performed.length > 0) {
        for (const op of data.operations_performed) {
          if (op.operation !== 'NOOP') {
            showToast(`⚡ [${op.operation}] ${op.fact}`);
          }
        }
        // Auto-refresh memory bank
        if (!isSearching) {
          loadUserMemories(currentUserId);
        }
      }

    } catch (err) {
      document.getElementById(typingId)?.remove();
      appendMessageBubble('assistant', `⚠️ Sorry, I encountered an error: ${err.message}`);
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

  // Utility to prevent XSS
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
