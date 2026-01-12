import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Brush,
  CartesianGrid,
  Legend
} from "recharts";

export default function ServerDetails() {
  const { server } = useParams();
  const navigate = useNavigate();
  const [chartData, setChartData] = useState([]);

  useEffect(() => {
    fetch(`/api/server/plot/${server}`)
      .then(res => res.text())
      .then(csv => {
        const lines = csv.trim().split("\n");
        const headers = lines[0].split(",");

        const data = lines.slice(1).map(row => {
          const values = row.split(",");
          const obj = {};
          headers.forEach((h, i) => {
            obj[h] = isNaN(values[i]) ? values[i] : Number(values[i]);
          });
          return obj;
        });

        setChartData(
          data.filter(
            d =>
              d.cpu_actual ||
              d.cpu_predicted ||
              d.mem_actual ||
              d.mem_predicted
          )
        );
      });
  }, [server]);

  return (
    <div style={{ padding: "30px" }}>
      <button onClick={() => navigate(-1)}>⬅ Back</button>

      <h2 style={{ marginTop: "20px" }}>
        Server Forecast — <span style={{ color: "red" }}>{server}</span>
      </h2>

      {chartData.length === 0 ? (
        <p style={{ color: "red" }}>No data available</p>
      ) : (
        <div style={{ width: "100%", height: "450px", marginTop: "30px" }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="timestamp" />
              <YAxis />
              <Tooltip />
              <Legend />

              <Line
                dataKey="cpu_actual"
                stroke="#000"
                dot={false}
                name="CPU Actual"
              />
              <Line
                dataKey="cpu_predicted"
                stroke="red"
                dot={false}
                name="CPU Predicted"
              />
              <Line
                dataKey="mem_actual"
                stroke="#555"
                dot={false}
                name="Memory Actual"
              />
              <Line
                dataKey="mem_predicted"
                stroke="orange"
                dot={false}
                name="Memory Predicted"
              />

              {/* Time-Machine Slider */}
              <Brush
                dataKey="timestamp"
                height={30}
                stroke="red"
                startIndex={Math.max(chartData.length - 200, 0)}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
