/**
 * frontend/app.js
 * Client for the voice/text RAG UI. Talks to /api/voice and /api/text,
 * renders the guardrail status, a proportional latency trace, the answer,
 * and retrieved sources.
 */

// API base URL. Empty string means "same origin" (frontend served by the
// same FastAPI app, e.g. local dev or a single-service deploy). When the
// frontend is deployed separately (e.g. on Vercel, backend on Fly.io),
// set window.__API_BASE__ = "https://your-app.fly.dev" in a small inline
// script tag in index.html before this file loads.
const API_BASE = (typeof window !== 'undefined' && window.__API_BASE__) || '';

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
    const copyBtn = document.getElementById('copyBtn');
    const sourcesSection = document.getElementById('sourcesSection');
    const sourcesToggle = document.getElementById('sourcesToggle');
    const sourcesLabel = document.getElementById('sourcesLabel');
    const sourcesList = document.getElementById('sourcesList');

    // These are real numbers from the last committed run of
    // src/benchmark_e2e_latency.py (see data/e2e_latency_benchmark.txt),
    // not a live feed -- rerun the script and update these after any change
    // that could affect retrieval or generation latency.
    const BENCH = { p50: 24.89, p70: 32.79, p100: 62.82, n: 20, withinPct: 100.0 };
    document.getElementById('benchN').textContent = BENCH.n;
    animateCount(document.getElementById('benchP50'), BENCH.p50, ' ms', 1);
    animateCount(document.getElementById('benchP70'), BENCH.p70, ' ms', 1);
    animateCount(document.getElementById('benchP100'), BENCH.p100, ' ms', 1);
    animateCount(document.getElementById('benchPct'), BENCH.withinPct, '%', 0);

    // --- Entrance choreography: masthead, then input, then bench, in a
    // quick stagger rather than everything popping in at once.
    requestAnimationFrame(() => {
        document.querySelector('.masthead').classList.add('in');
        document.querySelector('.input-panel').classList.add('in');
        document.querySelector('.bench').classList.add('in');
    });

    // --- Count-up animation for a number, e.g. latency stats on load.
    function animateCount(el, target, suffix, decimals, duration = 700) {
        if (!el) return;
        const start = performance.now();
        function tick(now) {
            const t = Math.min(1, (now - start) / duration);
            const eased = 1 - Math.pow(1 - t, 3); // ease-out cubic
            const value = target * eased;
            el.textContent = `${value.toFixed(decimals)}${suffix}`;
            if (t < 1) requestAnimationFrame(tick);
        }
        requestAnimationFrame(tick);
    }

    let mediaRecorder = null;
    let audioChunks = [];
    let recording = false;
    let startTime = 0;
    let timerInterval = null;

    // --- Live waveform visualizer (real mic input via Web Audio API, not
    // decorative -- it reflects actual input level per frequency band).
    const micWave = document.getElementById('micWave');
    const waveCtx = micWave.getContext('2d');
    let audioCtx = null;
    let analyser = null;
    let waveRAF = null;

    function startWaveform(stream) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioCtx.createAnalyser();
        analyser.fftSize = 64;
        audioCtx.createMediaStreamSource(stream).connect(analyser);
        const data = new Uint8Array(analyser.frequencyBinCount);
        const w = micWave.width, h = micWave.height;
        const barCount = data.length;
        const barWidth = w / barCount;

        function draw() {
            analyser.getByteFrequencyData(data);
            waveCtx.clearRect(0, 0, w, h);
            waveCtx.fillStyle = '#f0a83a';
            for (let i = 0; i < barCount; i++) {
                const barH = Math.max(2, (data[i] / 255) * h);
                waveCtx.fillRect(i * barWidth, (h - barH) / 2, Math.max(1, barWidth - 2), barH);
            }
            waveRAF = requestAnimationFrame(draw);
        }
        draw();
    }

    function stopWaveform() {
        if (waveRAF) cancelAnimationFrame(waveRAF);
        waveRAF = null;
        if (audioCtx) {
            audioCtx.close().catch(() => {});
            audioCtx = null;
        }
        waveCtx.clearRect(0, 0, micWave.width, micWave.height);
    }

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
                stopWaveform();
                setRecording(false);
                const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
                await sendVoiceQuery(blob);
            };
            mediaRecorder.start();
            startWaveform(stream);
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
            micWave.classList.remove('hidden');
            startTime = Date.now();
            timerInterval = setInterval(() => {
                const s = Math.floor((Date.now() - startTime) / 1000);
                micTimer.textContent = `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
            }, 1000);
        } else {
            micLabel.textContent = 'Tap to speak';
            micTimer.classList.add('hidden');
            micWave.classList.add('hidden');
            clearInterval(timerInterval);
        }
    }

    async function sendVoiceQuery(blob) {
        setBusy('Transcribing and searching…');
        const form = new FormData();
        form.append('file', blob, 'recording.webm');
        try {
            const res = await fetch(`${API_BASE}/api/voice`, { method: 'POST', body: form });
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
            const res = await fetch(`${API_BASE}/api/text`, {
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

    // --- Language selector ---
    // The corpus itself is Assamese-source (asm_Beng script); English, Hindi,
    // and Hinglish queries are answered via the multilingual embedding
    // model's cross-lingual matching, not because those languages exist as
    // separate passages. Each entry is a query verified to actually work,
    // not a promise every phrasing in that language will.
    const SAMPLE_QUERIES = {
        en: [
            { text: 'what is a corporation?', label: 'what is a corporation?' },
            { text: 'what is certified B corps', label: 'certified B corp?' },
            { text: 'how to make a bomb at home', label: 'try a refusal ↦' },
        ],
        hi: [
            { text: 'निगम क्या है?', label: 'निगम क्या है?' },
            { text: 'कॉर्पोरेशन क्या है', label: 'कॉर्पोरेशन क्या है' },
        ],
        as: [
            { text: 'কর্পোরেশন কি?', label: 'কর্পোরেশন কি?' },
            { text: 'কৰ্পোৰেচন কি?', label: 'কৰ্পোৰেচন কি?' },
        ],
        hinglish: [
            { text: 'corporation kya h', label: 'corporation kya h' },
            { text: 'corporation kya hai', label: 'corporation kya hai' },
        ],
        mr: [
            { text: 'कॉर्पोरेशन म्हणजे काय?', label: 'कॉर्पोरेशन म्हणजे काय?' },
        ],
        gu: [
            { text: 'કોર્પોરેશન શું છે?', label: 'કોર્પોરેશન શું છે?' },
        ],
        te: [
            { text: 'కార్పొరేషన్ అంటే ఏమిటి?', label: 'కార్పొరేషన్ అంటే ఏమిటి?' },
        ],
        kn: [
            { text: 'ಕಾರ್ಪೊರೇಶನ್ ಎಂದರೇನು?', label: 'ಕಾರ್ಪೊರೇಶನ್ ಎಂದರೇನು?' },
        ],
    };

    const langSelect = document.getElementById('langSelect');
    const sampleChips = document.getElementById('sampleChips');

    function renderChips(lang) {
        sampleChips.innerHTML = '';
        (SAMPLE_QUERIES[lang] || []).forEach((q) => {
            const chip = document.createElement('button');
            chip.type = 'button';
            chip.className = 'chip';
            chip.dataset.query = q.text;
            chip.textContent = q.label;
            chip.addEventListener('click', () => {
                queryInput.value = q.text;
                textForm.dispatchEvent(new Event('submit'));
            });
            sampleChips.appendChild(chip);
        });
    }

    langSelect.addEventListener('change', () => renderChips(langSelect.value));
    renderChips(langSelect.value);

    async function safeJson(res) {
        const ct = res.headers.get('content-type') || '';
        if (!ct.includes('application/json')) return null;
        return res.json();
    }

    // --- Rendering ---

    function revealResultCard() {
        resultCard.classList.remove('hidden');
        // Retrigger the entrance transition on every new result, not just
        // the first time the card appears.
        resultCard.classList.remove('in');
        void resultCard.offsetWidth; // force reflow so the class removal registers
        resultCard.classList.add('in');
    }

    function setBusy(msg) {
        revealResultCard();
        statusPill.textContent = '';
        statusPill.className = 'pill pill-busy';
        traceBar.innerHTML = '';
        totalMs.textContent = '';
        transcriptRow.classList.add('hidden');
        answerText.textContent = msg;
        answerText.classList.add('loading-dots');
        copyBtn.classList.add('hidden');
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
        revealResultCard();
        answerText.classList.remove('loading-dots');

        const label = (status || 'unknown').replace(/_/g, ' ');
        statusPill.textContent = label;
        statusPill.className = `pill ${status || ''} pop`;

        renderTrace(latency);
        animateCount(totalMs, Math.round(latency.total_ms || 0), ' ms', 0, 500);

        answerText.textContent = answer || 'No answer generated.';
        answerText.classList.remove('fade-in');
        void answerText.offsetWidth;
        answerText.classList.add('fade-in');
        copyBtn.classList.toggle('hidden', !answer);
        copyBtn.classList.remove('copied');
        renderSources(sources);
    }

    copyBtn.addEventListener('click', async () => {
        try {
            await navigator.clipboard.writeText(answerText.textContent);
            copyBtn.classList.add('copied');
            setTimeout(() => copyBtn.classList.remove('copied'), 1500);
        } catch (err) {
            // Clipboard API can be unavailable (e.g. insecure context); fail silently.
        }
    });

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
        const segments = [];
        stages.forEach(([name, ms]) => {
            if (ms <= 0) return;
            const seg = document.createElement('span');
            seg.className = `trace-seg ${name}`;
            seg.style.width = '0%';
            seg.title = `${name.replace('_', ' ')}: ${ms.toFixed(1)} ms`;
            traceBar.appendChild(seg);
            segments.push([seg, (ms / total) * 100]);
        });
        // Grow segments from 0 on the next frame so the width change is a
        // transition rather than appearing fully-formed.
        requestAnimationFrame(() => {
            segments.forEach(([seg, pct]) => { seg.style.width = `${pct}%`; });
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
        sourcesList.style.maxHeight = '0px';
        sourcesToggle.setAttribute('aria-expanded', 'false');
    }

    sourcesToggle.addEventListener('click', () => {
        const open = sourcesToggle.getAttribute('aria-expanded') === 'true';
        sourcesToggle.setAttribute('aria-expanded', String(!open));
        sourcesList.style.maxHeight = open ? '0px' : `${sourcesList.scrollHeight}px`;
    });

    function showError(msg) {
        revealResultCard();
        statusPill.textContent = 'error';
        statusPill.className = 'pill rejected pop';
        traceBar.innerHTML = '';
        totalMs.textContent = '';
        transcriptRow.classList.add('hidden');
        answerText.classList.remove('loading-dots');
        answerText.textContent = msg;
        copyBtn.classList.add('hidden');
        sourcesSection.classList.add('hidden');
    }

    function escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }
});
