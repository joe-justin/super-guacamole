import React, { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import Plot from "react-plotly.js";

export default function ServerDetail() {
  const { serverName } = useParams();
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function load() {
      try {
        const resp = await fetch(
          `http://localhost:8000/api/predictions/csv/${encodeURIComponent(serverName)}`
        );
        if (!resp.ok) throw new Error(`CSV fetch failed (${resp.status})`);
        const text = await resp.text();

        const rows = text.trim().split("\n");
        if (rows.length < 2) throw new Error("CSV is empty");

        const headers = rows[0].split(",").map((h) => h.trim());
        const idxTimestamp = headers.indexOf("timestamp");
        const idxCpuActual = headers.indexOf("cpu_actual");
        const idxCpuPred = headers.indexOf("cpu_predicted");
        const idxMemActual = headers.indexOf("mem_actual");
        const idxMemPred = headers.indexOf("mem_predicted");

        const parsed = rows.slice(1).map((r) => {
          const cols = r.split(",");
          return {
            timestamp: cols[idxTimestamp],
            cpu_actual: parseFloat(cols[idxCpuActual]),
            cpu_predicted: parseFloat(cols[idxCpuPred]),
            mem_actual: parseFloat(cols[idxMemActual]),
            mem_predicted: parseFloat(cols[idxMemPred]),
          };
        });

        const valid = parsed.filter(
          (d) =>
            !isNaN(d.cpu_actual) &&
            !isNaN(d.cpu_predicted) &&
            !isNaN(d.mem_actual) &&
            !isNaN(d.mem_predicted)
        );

        setData(valid);
      } catch (e) {
        console.error("Data fetch error:", e);
        setError(e.message);
      }
    }
    load();
  }, [serverName]);

  if (error)
    return (
      <div style={{ padding: 20, color: "red" }}>
        Error loading data: {error} <br />
        <Link to="/landing">Back</Link>
      </div>
    );

  if (!data)
    return (
      <div style={{ padding: 20 }}>
        Loading data... <br />
        <Link to="/landing">Back</Link>
      </div>
    );

  const times = data.map((r) => r.timestamp);
  const cpuActual = data.map((r) => r.cpu_actual);
  const cpuPred = data.map((r) => r.cpu_predicted);
  const memActual = data.map((r) => r.mem_actual);
  const memPred = data.map((r) => r.mem_predicted);

  if (!cpuActual.length)
    return (
      <div style={{ padding: 20 }}>
        No valid data for {serverName}.
        <div style={{ marginTop: 16 }}>
          <Link to="/landing">Back</Link>
        </div>
      </div>
    );

  return (
    <div style={{ padding: 20 }}>
      <h2>{serverName}</h2>
      <Plot
        data={[
          {
            x: times,
            y: cpuActual,
            type: "scatter",
            mode: "lines",
            name: "CPU Actual",
            line: { color: "blue", width: 2 },
          },
          {
            x: times,
            y: cpuPred,
            type: "scatter",
            mode: "lines",
            name: "CPU Predicted",
            line: { color: "red", dash: "dashdot", width: 2 },
          },
          {
            x: times,
            y: memActual,
            type: "scatter",
            mode: "lines",
            name: "Memory Actual",
            line: { color: "green", width: 2 },
          },
          {
            x: times,
            y: memPred,
            type: "scatter",
            mode: "lines",
            name: "Memory Predicted",
            line: { color: "orange", dash: "dashdot", width: 2 },
          },
        ]}
        layout={{
          width: 1000,
          height: 500,
          title: `${serverName} - CPU & Memory (Actual vs Predicted)`,
          xaxis: { title: "Timestamp" },
          yaxis: { title: "Utilization (%)", range: [0, 100] },
          legend: { orientation: "h", y: -0.2 },
          hovermode: "x unified",
        }}
      />
      <div style={{ marginTop: 16 }}>
        <Link to="/landing">Back to functions</Link>
      </div>
    </div>
  );
}
