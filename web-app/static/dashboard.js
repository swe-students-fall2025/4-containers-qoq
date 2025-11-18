async function fetchDashboardData(){
    try {
        const res = await fetch("/api/dashboard-data");
        const data = await res.json();

        document.getElementById("total-runs").textContent = data.total_runs;
        document.getElementById("diff-instruments").textContent = data.diff_instruments;

        // top instrument
        if (data.top_instrument){
            document.getElementById("top-instrument").textContent = data.top_instrument;
            document.getElementById("top-instrument-confidence").textContent = (data.top_instrument_confidence * 100).toFixed(1) + "%";
        } else {
            document.getElementById("top-instrument").textContent = "--";
            document.getElementById("top-instrument-confidence").textContent = "--%";
        }

        // last analysis time
        const lastTimeElem = document.getElementById("last-analysis-time");
        lastTimeElem.textContent = data.last_analysis_time ? new Date(data.last_analysis_time).toLocaleString() : "--";

        // recent logs
        renderRecent(data.recent);

    } catch (err) {
        console.error("Dashboard load failed:", err);
    }
}


function renderRecent(list) {
    const container = document.getElementById("recent-log-container")
    // Clear old 
    while (container.firstChild) {
        container.removeChild(container.firstChild);
    }
    if (!list || list.length === 0) {
        const msg = document.createElement("p");
        msg.textContent = "No recent logs.";
        container.appendChild(msg);
        return;
    }
    // Create table
    const table = document.createElement("table");
    table.className = "log-table";

    const headerRow = document.createElement("tr");
    const headers = ["Instrument", "Confidence", "Source", "Captured At"];
    headers.forEach(text => {
        const th = document.createElement("th");
        th.textContent = text;
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);
    // rows
    list.forEach(item => {
        const row = document.createElement("tr");
        // instrument
        const cellInst = document.createElement("td");
        cellInst.textContent = item.instrument;
        row.appendChild(cellInst);
        // confidence
        const cellConf = document.createElement("td");
        cellConf.textContent = (item.confidence * 100).toFixed(1) + "%";
        row.appendChild(cellConf);
        // source
        const cellSource = document.createElement("td");
        cellSource.textContent = item.source;
        row.appendChild(cellSource);
        // timestamp
        const cellTime = document.createElement("td");
        cellTime.textContent = item.captured_at
            ? new Date(item.captured_at).toLocaleString()
            : "--";
        row.appendChild(cellTime);
        table.appendChild(row);
    });
    container.appendChild(table);
}

fetchDashboardData();
setInterval(fetchDashboardData, 4000);