import { useEffect, useState } from "react";
import { useParams, useNavigate, useLocation } from "react-router-dom";
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

/* ===== Simple Server Card (for multi-server view) ===== */
function ServerCard({ name, onClick }) {
  return (
    <div
      className="flip-card red"
      onClick={onClick}
      style={{ cursor: "pointer" }}
    >
      <div className="flip-inner">
        <div className="flip-front">
          <div className="card-title">{name}</div>
          <div className="card-sub">View Forecast</div>
        </div>
        <div className="flip-back">
          <div className="explore">Open</div>
        </div>
      </div>
    </div>
  );
}

export default function ServerDetails() {
  const { server } = useParams();
  const navigate = useNavigate();
  const location = useLocation();

  const [chartData, setChartData] = useState([]);
  const [activeServer, setActiveServer] = useState(null);
  const [servers, setServers] = useState([]);
  const [growth, setGrowth] = useState(null);

  /* ===============================
     Parse incoming state or query
     =============================== */
  useEffect(() => {
    // Expecting from Welcome.jsx navigation:
    // navigate("/servers", { state: { servers: [...], growth: 30 } })

    if (location.state?.servers?.length) {
      setServers(location.state.servers);
      setGrowth(location.state.growth || null);

      if (location.state.servers.length === 1) {
        setActiveServer(location.state.servers[0]);
      }
    } else if (server) {
      // legacy single-server route
      setActiveServer(server);
      setServers([server]);
    }
  }, [location.state, server]);

  /* ===============================
     Load CSV when server selected
     =============================== */
  useEffect(() => {
    if (!activeServer) return;

    const suffix = growth ? `${activeServer}_${growth}` : activeServer;
    const endpoint = growth
      ? `/api/predictions/csv/${suffix}`
      : `/api/server/plot/${activeServer}`;

    fetch(endpoint)
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
      })
      .catch(err => {
        console.error("Failed to load server CSV", err);
        setChartData([]);
      });
  }, [activeServer, growth]);

  /* ===============================
     MULTI-SERVER GRID VIEW
     =============================== */
  if (servers.length > 1 && !activeServer) {
    return (
      <div style={{ padding: "30px" }}>
        <button onClick={() => navigate(-1)}>⬅ Back</button>
        <h2 style={{ marginTop: "20px" }}>
          Select a Server to View Forecast
        </h2>

        <div className="cards-grid" style={{ marginTop: 30 }}>
          {servers.map(s => (
            <ServerCard
              key={s}
              name={s}
              onClick={() => setActiveServer(s)}
            />
          ))}
        </div>
      </div>
    );
  }

  /* ===============================
     SINGLE SERVER CHART VIEW
     =============================== */
  return (
    <div style={{ padding: "30px" }}>
      <button onClick={() => navigate(-1)}>⬅ Back</button>

      <h2 style={{ marginTop: "20px" }}>
        Server Forecast —{" "}
        <span style={{ color: "red" }}>{activeServer}</span>
        {growth && (
          <span style={{ marginLeft: 10, fontSize: 14, color: "#555" }}>
            (Growth: {growth}%)
          </span>
        )}
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
