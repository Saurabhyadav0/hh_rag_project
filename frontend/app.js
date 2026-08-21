/**
 * frontend/app.js
 * Client for the voice/text RAG UI. Talks to /api/voice and /api/text,
 * renders the guardrail status, a proportional latency trace, the answer,
 * and retrieved sources.
 */

document.addEventListener('DOMContentLoaded', () => {
    const recordBtn = document.getElementById('recordBtn');
    const micLabel = document.getElementById('micLabel');
    const micTimer = document.getElementById('micTimer');
    const textForm = document.getElementById('textForm');
    const queryInput = document.getElementById('queryInput');
    const resultCard = document.getElementById('resultCard');

    const statusPill = document.getElementById('statusPill');
    const traceBar = document.getElementById('traceBar');
    const totalMs = document.getElementById('totalMs');
    const transcriptRow = document.getElementById('transcriptRow');
    const transcriptText = document.getElementById('transcriptText');
    const answerText = document.getElementById('answerText');
    const sourcesSection = document.getElementById('sourcesSection');
    const sourcesToggle = document.getElementById('sourcesToggle');
    const sourcesLabel = document.getElementById('sourcesLabel');
    const sourcesList = document.getElementById('sourcesList');

    // These are real numbers from the last committed run of
    // src/benchmark_e2e_latency.py (see data/e2e_latency_benchmark.txt),
    // not a live feed -- rerun the script and update these after any change
    // that could affect retrieval or generation latency.
    const BENCH = { p50: 28.05, p70: 40.46, p100: 573.31, n: 20, withinPct: 95.0 };
    document.getElementById('benchP50').textContent = `${BENCH.p50} ms`;
    document.getElementById('benchP70').textContent = `${BENCH.p70} ms`;
    document.getElementById('benchP100').textContent = `${BENCH.p100} ms`;
    document.getElementById('benchPct').textContent = `${BENCH.withinPct}%`;
    document.getElementById('benchN').textContent = BENCH.n;

    let mediaRecorder = null;
    let audioChunks = [];
    let recording = false;
    let startTime = 0;
    let timerInterval = null;

    // --- Voice recording ---

    recordBtn.addEventListener('click', async () => {
        if (recording) {
            mediaRecorder.stop();
            return;
        }
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioChunks = [];
            let mimeType = 'audio/webm';
            if (!MediaRecorder.isTypeSupported('audio/webm')) {
                mimeType = MediaRecorder.isTypeSupported('audio/mp4') ? 'audio/mp4' : '';
            }
            mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : {});
            mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunks.push(e.data); };
            mediaRecorder.onstop = async () => {
                stream.getTracks().forEach((t) => t.stop());
                setRecording(false);
                const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
                await sendVoiceQuery(blob);
            };
            mediaRecorder.start();
            setRecording(true);
        } catch (err) {
            alert(`Microphone access error: ${err.message}. Check browser permissions.`);
        }
    });

    function setRecording(on) {
        recording = on;
        recordBtn.setAttribute('aria-pressed', String(on));
        if (on) {
            micLabel.textContent = 'Tap to stop';
            micTimer.classList.remove('hidden');
            startTime = Date.now();
            timerInterval = setInterval(() => {
                const s = Math.floor((Date.now() - startTime) / 1000);
                micTimer.textContent = `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
            }, 1000);
        } else {
            micLabel.textContent = 'Tap to speak';
            micTimer.classList.add('hidden');
            clearInterval(timerInterval);
        }
    }

    async function sendVoiceQuery(blob) {
        setBusy('Transcribing and searching…');
        const form = new FormData();
        form.append('file', blob, 'recording.webm');
        try {
            const res = await fetch('/api/voice', { method: 'POST', body: form });
            const data = await safeJson(res);
            if (!data) return showError('The server did not return a valid response.');
            renderVoice(data);
        } catch (err) {
            showError('Network error: could not reach the API.');
        }
    }

    // --- Text query ---

    textForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = queryInput.value.trim();
        if (!query) return;
        setBusy('Searching…');
        try {
            const res = await fetch('/api/text', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query }),
            });
            const data = await safeJson(res);
            if (!data) return showError('The server did not return a valid response.');
            renderText(data);
        } catch (err) {
            showError(`Request failed: ${err.message || 'could not reach the API.'}`);
        }
    });

    document.querySelectorAll('.chip[data-query]').forEach((chip) => {
        chip.addEventListener('click', () => {
            queryInput.value = chip.getAttribute('data-query');
            textForm.dispatchEvent(new Event('submit'));
        });
    });

    async function safeJson(res) {
        const ct = res.headers.get('content-type') || '';
        if (!ct.includes('application/json')) return null;
        return res.json();
    }

    // --- Rendering ---

    function setBusy(msg) {
        resultCard.classList.remove('hidden');
        statusPill.textContent = '…';
        statusPill.className = 'pill';
        traceBar.innerHTML = '';
        totalMs.textContent = '';
        transcriptRow.classList.add('hidden');
        answerText.textContent = msg;
        sourcesSection.classList.add('hidden');
    }

    function renderVoice(data) {
        renderCommon(data.status, data.grounded, data.latency || {}, data.answer, data.rag_details?.retrieved_context || []);
        if (data.transcript) {
            transcriptRow.classList.remove('hidden');
            transcriptText.textContent = data.transcript;
        }
    }

    function renderText(data) {
        renderCommon(data.status, data.grounded, data.latency || {}, data.answer, data.retrieved_context || []);
    }

    function renderCommon(status, grounded, latency, answer, sources) {
        resultCard.classList.remove('hidden');

        const label = (status || 'unknown').replace(/_/g, ' ');
        statusPill.textContent = label;
        statusPill.className = `pill ${status || ''}`;

        renderTrace(latency);
        totalMs.textContent = `${Math.round(latency.total_ms || 0)} ms`;

        answerText.textContent = answer || 'No answer generated.';
        renderSources(sources);
    }

    function renderTrace(latency) {
        traceBar.innerHTML = '';
        const stages = [
            ['input_guardrail', latency.input_guardrail_ms || 0],
            ['retrieval', latency.retrieval_ms || 0],
            ['generation', latency.generation_ms || 0],
            ['grounding', latency.grounding_ms || 0],
        ];
        const total = stages.reduce((sum, [, v]) => sum + v, 0);
        if (total <= 0) return;
        stages.forEach(([name, ms]) => {
            if (ms <= 0) return;
            const seg = document.createElement('span');
            seg.className = `trace-seg ${name}`;
            seg.style.width = `${(ms / total) * 100}%`;
            seg.title = `${name.replace('_', ' ')}: ${ms.toFixed(1)} ms`;
            traceBar.appendChild(seg);
        });
    }

    function renderSources(sources) {
        if (!sources || sources.length === 0) {
            sourcesSection.classList.add('hidden');
            return;
        }
        sourcesSection.classList.remove('hidden');
        sourcesLabel.textContent = `${sources.length} source${sources.length === 1 ? '' : 's'}`;
        sourcesList.innerHTML = '';
        sources.forEach((src, i) => {
            const item = document.createElement('div');
            item.className = 'source-item';
            const score = typeof src.score === 'number' ? src.score.toFixed(4) : 'n/a';
            item.innerHTML = `
                <div class="source-item-head">
                    <span>#${i + 1} · ${escapeHtml(src.chunk_id || 'unknown')}</span>
                    <span>score ${score}</span>
                </div>
                <p class="source-item-text">${escapeHtml(src.text || '')}</p>
            `;
            sourcesList.appendChild(item);
        });
        sourcesList.classList.add('hidden');
        sourcesToggle.setAttribute('aria-expanded', 'false');
    }

    sourcesToggle.addEventListener('click', () => {
        const open = sourcesToggle.getAttribute('aria-expanded') === 'true';
        sourcesToggle.setAttribute('aria-expanded', String(!open));
        sourcesList.classList.toggle('hidden', open);
    });

    function showError(msg) {
        resultCard.classList.remove('hidden');
        statusPill.textContent = 'error';
        statusPill.className = 'pill rejected';
        traceBar.innerHTML = '';
        totalMs.textContent = '';
        transcriptRow.classList.add('hidden');
        answerText.textContent = msg;
        sourcesSection.classList.add('hidden');
    }

    function escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }
});
