import React, { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import Plot from "react-plotly.js";

export default function ServerDetail() {
  const { serverName } = useParams();
  const [predicted, setPredicted] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function load() {
      try {
        const resp = await fetch(
          `http://localhost:8000/api/predictions/csv/${encodeURIComponent(serverName)}`
        );
        if (!resp.ok) throw new Error(`CSV fetch failed (${resp.status})`);
        const text = await resp.text();

        // ✅ Correct newline split
        const rows = text.trim().split("\n");
        if (rows.length < 2) throw new Error("CSV is empty");

        const headers = rows[0].split(",").map(h => h.trim());
        const idxTimestamp = headers.indexOf("timestamp");
        const idxCpu = headers.findIndex(h => h.toLowerCase().includes("cpu_pred_combined"));
        const idxMem = headers.findIndex(h => h.toLowerCase().includes("mem_pred_combined"));

        const df = rows.slice(1).map(r => {
          const cols = r.split(",");
          return {
            timestamp: cols[idxTimestamp],
            cpu_pred_combined: parseFloat(cols[idxCpu]),
            mem_pred_combined: parseFloat(cols[idxMem]),
          };
        });

        // Filter out invalid numeric rows
        const valid = df.filter(
          d => !isNaN(d.cpu_pred_combined) && !isNaN(d.mem_pred_combined)
        );

        setPredicted(valid);
      } catch (e) {
        console.error("Prediction fetch error:", e);
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

  if (!predicted)
    return (
      <div style={{ padding: 20 }}>
        Loading predictions... <br />
        <Link to="/landing">Back</Link>
      </div>
    );

  const times = predicted.map(r => r.timestamp);
  const cpu = predicted.map(r => r.cpu_pred_combined);
  const mem = predicted.map(r => r.mem_pred_combined);

  if (!cpu.length && !mem.length)
    return (
      <div style={{ padding: 20 }}>
        No data available for {serverName}.
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
            y: cpu,
            type: "scatter",
            mode: "lines",
            name: "CPU predicted",
            line: { color: "orange" },
          },
          {
            x: times,
            y: mem,
            type: "scatter",
            mode: "lines",
            name: "Memory predicted",
            line: { color: "teal" },
          },
        ]}
        layout={{
          width: 1000,
          height: 450,
          title: `${serverName} - Predicted CPU & Memory`,
          xaxis: { title: "Timestamp" },
          yaxis: { title: "Utilization (%)" },
        }}
      />
      <div style={{ marginTop: 16 }}>
        <Link to="/landing">Back to functions</Link>
      </div>
    </div>
  );
}
