import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  ResponsiveContainer
} from "recharts";

export default function Welcome() {
  const navigate = useNavigate();

  const [functions, setFunctions] = useState([]);
  const [applications, setApplications] = useState([]);
  const [servers, setServers] = useState([]);

  const [mode, setMode] = useState(null); // function | application | server
  const [selection, setSelection] = useState("");

  const [metrics, setMetrics] = useState({ cpu: true, mem: false });
  const [growth, setGrowth] = useState(30);

  // Load dropdown values
  useEffect(() => {
    fetch("/api/functions").then(r => r.json()).then(setFunctions);
    fetch("/api/applications").then(r => r.json()).then(setApplications);
    fetch("/api/servers").then(r => r.json()).then(setServers);
  }, []);

  const disabled = (type) => mode && mode !== type;

  // Fake animated banner data
  const bannerData = Array.from({ length: 50 }).map((_, i) => ({
    t: i,
    v: Math.sin(i / 4) * 20 + 60 + Math.random() * 5
  }));

  const submit = () => {
    if (!selection) return alert("Please select Function, Application or Server");

    if (mode === "function") navigate(`/functions/${selection}`);
    if (mode === "application") navigate(`/applications/${selection}`);
    if (mode === "server") navigate(`/servers/${selection}`);
  };

  const tooltip = (text) => ({
    title: text,
    style: { cursor: "help" }
  });

  return (
    <div style={{ padding: "30px" }}>
      {/* BANNER */}
      <div style={{ height: "200px", marginBottom: "40px" }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={bannerData}>
            <XAxis dataKey="t" hide />
            <YAxis hide />
            <Line
              type="monotone"
              dataKey="v"
              stroke="red"
              strokeWidth={2}
              dot={false}
              isAnimationActive={true}
              animationDuration={3000}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <h2 style={{ marginBottom: "20px" }}>
        Welcome to <span style={{ color: "red" }}>System Health Dashboard</span>
      </h2>

      <p style={{ marginBottom: "30px" }}>
        Configure your future forecast by selecting scope, metrics and growth rate.
      </p>

      {/* FILTER PANEL */}
      <div style={{ maxWidth: "500px" }}>

        {/* FUNCTION */}
        <div style={{ marginBottom: "15px" }}>
          <select
            disabled={disabled("function")}
            {...tooltip("Analyze by Business Stream / Function")}
            onChange={e => {
              setMode("function");
              setSelection(e.target.value);
            }}
          >
            <option value="">Business Stream</option>
            {functions.map(f => (
              <option key={f.Function} value={f.Function}>
                {f.Function}
              </option>
            ))}
          </select>
        </div>

        {/* APPLICATION */}
        <div style={{ marginBottom: "15px" }}>
          <select
            disabled={disabled("application")}
            {...tooltip("Analyze by Application")}
            onChange={e => {
              setMode("application");
              setSelection(e.target.value);
            }}
          >
            <option value="">Application</option>
            {applications.map(a => (
              <option key={a.Application} value={a.Application}>
                {a.Application}
              </option>
            ))}
          </select>
        </div>

        {/* SERVER */}
        <div style={{ marginBottom: "20px" }}>
          <select
            disabled={disabled("server")}
            {...tooltip("Analyze by Hostname / Server")}
            onChange={e => {
              setMode("server");
              setSelection(e.target.value);
            }}
          >
            <option value="">Hostname</option>
            {servers.map(s => (
              <option key={s.server} value={s.server}>
                {s.server}
              </option>
            ))}
          </select>
        </div>

        {/* METRICS */}
        <div style={{ marginBottom: "20px" }}>
          <label {...tooltip("Forecast CPU utilization")}>
            <input
              type="checkbox"
              checked={metrics.cpu}
              onChange={() => setMetrics(m => ({ ...m, cpu: !m.cpu }))}
            />{" "}
            CPU
          </label>

          <label style={{ marginLeft: "30px" }} {...tooltip("Forecast Memory utilization")}>
            <input
              type="checkbox"
              checked={metrics.mem}
              onChange={() => setMetrics(m => ({ ...m, mem: !m.mem }))}
            />{" "}
            Memory
          </label>
        </div>

        {/* GROWTH */}
        <div style={{ marginBottom: "30px" }}>
          <span {...tooltip("Expected workload growth rate")}>
            Growth Rate (%)
          </span>{" "}
          <select value={growth} onChange={e => setGrowth(+e.target.value)}>
            {[10,20,30,40,50,60,70,80,90,100].map(g => (
              <option key={g} value={g}>{g}%</option>
            ))}
          </select>
        </div>

        {/* SUBMIT */}
        <button
          onClick={submit}
          style={{
            padding: "10px 25px",
            background: "red",
            color: "white",
            border: "none",
            cursor: "pointer",
            fontSize: "16px"
          }}
        >
          Submit & Explore
        </button>
      </div>
    </div>
  );
}
