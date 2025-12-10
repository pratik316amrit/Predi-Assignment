let currentMode = 'API';
document.body.setAttribute('data-theme', 'green'); // Start with Green
document.body.setAttribute('data-theme', 'green'); // Initialize with Green Theme

function setMode(mode) {
    currentMode = mode;

    // Update UI buttons - using new classes
    document.querySelectorAll('.nav-item').forEach(btn => {
        btn.classList.remove('active');
        btn.querySelector('.dot').classList.remove('active');
    });

    const activeBtn = document.getElementById(`btn-${mode.toLowerCase()}`);
    if (activeBtn) {
        activeBtn.classList.add('active');
        activeBtn.querySelector('.dot').classList.add('active');
    }

    // Update Mode Display
    document.getElementById('current-mode-display').innerText = mode.toUpperCase() + " MODE";

    // Toggle Warnings
    const apiWarn = document.getElementById('api-warning');
    const localWarn = document.getElementById('local-warning');

    if (mode === 'API') {
        apiWarn.classList.remove('hidden');
        localWarn.classList.add('hidden');
        document.body.setAttribute('data-theme', 'green');
    } else {
        apiWarn.classList.add('hidden');
        localWarn.classList.remove('hidden');
        document.body.setAttribute('data-theme', 'red');
    }

    // Notify user in chat
    addCodeMessage(`> SWITCHING COMPUTE CORE: [${mode.toUpperCase()}]ESTABLISHED.`);
}

function handleKey(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

async function sendMessage() {
    const input = document.getElementById('user-input');
    const text = input.value.trim();

    if (!text) return;

    // Clear input
    input.value = '';

    // Add user message
    addUserMessage(text);

    // Show thinking state
    const loadingId = addBotMessage('<span class="blink">PROCESSING REQUEST...</span>');

    try {
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: text,
                mode: currentMode
            })
        });

        const data = await response.json();
        const contentDiv = document.getElementById(loadingId);

        if (data.data && data.data.length > 0) {
            let html = `<strong>EXTRACTION COMPLETE</strong><br>`;
            html += `<pre>${JSON.stringify(data.data, null, 2)}</pre>`;
            html += `<div style="margin-top:8px; font-size:0.8em; color:rgba(255,255,255,0.5);">Source Pages: ${data.pages}</div>`;
            contentDiv.innerHTML = html;
        } else {
            contentDiv.innerHTML = `No data extracted. Context found but structure was unclear.`;
        }

    } catch (error) {
        document.getElementById(loadingId).innerHTML = `<strong>ERROR:</strong> ${error.message}`;
    } finally {
        // Force scroll to bottom after content update
        const history = document.getElementById('chat-history');
        history.scrollTop = history.scrollHeight;
    }
}

function addUserMessage(text) {
    const history = document.getElementById('chat-history');
    const div = document.createElement('div');
    div.className = 'message user';
    div.innerHTML = `<div class="content">${text}</div>`;
    history.appendChild(div);
    history.scrollTop = history.scrollHeight;
}

function addBotMessage(html) {
    const history = document.getElementById('chat-history');
    const div = document.createElement('div');
    div.className = 'message bot';
    const id = 'msg-' + Date.now();
    div.innerHTML = `<div class="content" id="${id}">${html}</div>`;
    history.appendChild(div);
    history.scrollTop = history.scrollHeight;
    return id;
}

function addCodeMessage(text) {
    const history = document.getElementById('chat-history');
    const div = document.createElement('div');
    div.className = 'message system-intro'; // Use system style for status updates
    div.innerHTML = `
        <div class="code-content" style="padding:10px; font-size:11px;">
            ${text}
        </div>`;
    history.appendChild(div);
    history.scrollTop = history.scrollHeight;
}

function clearChat() {
    document.getElementById('chat-history').innerHTML = '';
}
