let lineChart, barChart;
let dashboardLogs = [];

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

        if (data.recent && data.recent.length > 0) {
            data.recent.forEach(log => {
                if (!dashboardLogs.find(l => l.captured_at === log.captured_at)) {
                    dashboardLogs.push(log);
                }
            });

            dashboardLogs.sort((a, b) => new Date(a.captured_at) - new Date(b.captured_at));
            renderCharts(dashboardLogs);
        }

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

function renderCharts(logs) {
    // 50 logs
    const recentLogs = logs.slice(-50);
    const labels = recentLogs.map(log => new Date(log.captured_at).toLocaleString()); // use full datetime
    const confidences = recentLogs.map(log => (log.confidence * 100).toFixed(1));
    const instruments = recentLogs.map(log => log.instrument);

    // line chart
    const ctxLine = document.getElementById("lineChart").getContext("2d");
    if (lineChart) lineChart.destroy();
    lineChart = new Chart(ctxLine, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: 'Confidence (%)',
                data: confidences,
                borderColor: '#000',
                backgroundColor: 'rgba(0,0,0,0.1)',
                tension: 0.3,
                fill: true,
                pointRadius: 5,
                pointBackgroundColor: '#3b82f6',
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { min: 60, max: 100, title: { display: true, text: 'CONFIDENCE (%)' } },
                x: { title: { display: true, text: 'TIMESTAMP' } }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: ctx => `Confidence: ${ctx.formattedValue}% | Instrument: ${instruments[ctx.dataIndex]}`
                    }
                }
            }
        }
    });

    // bar chart
    const ctxBar = document.getElementById("barChart").getContext("2d");
    const counts = {};
    recentLogs.forEach(log => counts[log.instrument] = (counts[log.instrument] || 0) + 1);

    const barLabels = Object.keys(counts);
    const barData = Object.values(counts);

    if (barChart) barChart.destroy();
    barChart = new Chart(ctxBar, {
        type: 'bar',
        data: {
            labels: barLabels,
            datasets: [{
                label: 'Frequency',
                data: barData,
                backgroundColor: '#3b82f6',
                borderColor: '#000',
                borderWidth: 1,
                borderRadius: 4,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true, title: { display: true, text: 'COUNT' } },
                x: { title: { display: true, text: 'INSTRUMENTS' } }
            },
            plugins: { legend: { display: false } }
        }
    });
}


fetchDashboardData();
setInterval(fetchDashboardData, 4000);