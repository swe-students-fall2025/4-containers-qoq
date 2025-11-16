const STATIC_DATA = {
    total_runs: "31",
    top_instrument: "PIANO",
    top_instrument_confidence: "98.7",
    unique_instruments_count: "14",

    recent_log: [
        { instrument: "SYNTH", time: "14:34:05", confidence: 98 },
        { instrument: "PIANO", time: "14:33:30", confidence: 92 },
        { instrument: "DRUMS", time: "14:32:45", confidence: 85 },
        { instrument: "BASS", time: "14:32:10", confidence: 91 },
        { instrument: "GUITAR", time: "14:31:55", confidence: 78 },
        { instrument: "SYNTH", time: "14:31:20", confidence: 99 },
        { instrument: "SAX", time: "14:30:50", confidence: 72 },
        { instrument: "PIANO", time: "14:30:20", confidence: 88 },
        { instrument: "DRUMS", time: "14:29:45", confidence: 94 },
    ],
};

document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("total-runs").textContent = STATIC_DATA.total_runs;
    document.getElementById("top-instrument").textContent = STATIC_DATA.top_instrument;
    document.getElementById("top-instrument-confidence").textContent = STATIC_DATA.top_instrument_confidence + "%";
    document.getElementById("diff-instruments").textContent = STATIC_DATA.unique_instruments_count;

    const logContainer = document.getElementById("recent-log-container");
    while (logContainer.firstChild) {
        logContainer.removeChild(logContainer.firstChild);
    }

    if (STATIC_DATA.recent_log.length === 0) {
        const emptyMsg = document.createElement("div");
        emptyMsg.className = "text-gray-500 py-2 font-mono";
        emptyMsg.textContent = "No recent analysis found";
        logContainer.appendChild(emptyMsg);
    } else {
        STATIC_DATA.recent_log.forEach(entry => {
            const logEntry = document.createElement("div");
            logEntry.className = "log-entry flex justify-between font-mono";

            // left side instruments
            const instrumentSpan = document.createElement("span");
            instrumentSpan.textContent = entry.instrument;

            // right side time and conf
            const detailsSpan = document.createElement("span");
            detailsSpan.textContent = `${entry.time} - Confidence: ${entry.confidence}%`;

            logEntry.appendChild(instrumentSpan);
            logEntry.appendChild(detailsSpan);
            logContainer.appendChild(logEntry);
        });
    }
});
