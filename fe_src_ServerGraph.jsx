import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";

export default function ServerGraph({ server }) {
  const [data, setData] = useState([]);

  useEffect(() => {
    fetch(`http://localhost:8000/api/server/plot/${server}`)
      .then(res => res.json())
      .then(setData)
      .catch(console.error);
  }, [server]);

  if (!data.length) return <div>Loading graph...</div>;

  const timestamps = data.map(d => d.timestamp);

  const cpuActual = {
    x: timestamps,
    y: data.map(d => d.cpu_actual),
    name: "CPU Actual",
    mode: "lines",
    type: "scatter",
  };

  const cpuPred = {
    x: timestamps,
    y: data.map(d => d.cpu_pred),
    name: "CPU Predicted",
    mode: "lines",
    type: "scatter",
    line: { dash: "dot" },
  };

  const memActual = {
    x: timestamps,
    y: data.map(d => d.mem_actual),
    name: "Memory Actual",
    mode: "lines",
    type: "scatter",
  };

  const memPred = {
    x: timestamps,
    y: data.map(d => d.mem_pred),
    name: "Memory Predicted",
    mode: "lines",
    type: "scatter",
    line: { dash: "dot" },
  };

  return (
    <Plot
      data={[cpuActual, cpuPred, memActual, memPred]}
      layout={{
        title: `${server} – CPU & Memory Forecast`,
        xaxis: { title: "Timestamp" },
        yaxis: { title: "Utilization (%)" },
        autosize: true,
        legend: { orientation: "h" },
      }}
      style={{ width: "100%", height: "600px" }}
      useResizeHandler
    />
  );
}
