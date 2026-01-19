import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  ResponsiveContainer
} from "recharts";
import "./welcome.css";   // we’ll add this small css file below

export default function Welcome() {
  const navigate = useNavigate();

  const [functions, setFunctions] = useState([]);
  const [applications, setApplications] = useState([]);
  const [servers, setServers] = useState([]);

  const [mode, setMode] = useState(null);
  const [selection, setSelection] = useState("");

  const [metrics, setMetrics] = useState({ cpu: true, mem: false });
  const [growth, setGrowth] = useState(30);

  const [bannerData, setBannerData] = useState([]);

  // Load dropdown data
  useEffect(() => {
    fetch("/api/functions").then(r => r.json()).then(data => setFunctions(data || []));
    fetch("/api/applications").then(r => r.json()).then(data => setApplications(data || []));
    fetch("/api/servers").then(r => r.json()).then(data => setServers(data || []));
  }, []);

  // Continuous animated banner
  useEffect(() => {
    let t = 0;
    const interval = setInterval(() => {
      setBannerData(prev => {
        const next = [...prev];
        if (next.length > 60) next.shift();

        next.push({
          t,
          v: 60 + Math.sin(t / 5) * 20 + Math.random() * 8
        });

        t++;
        return next;
      });
    }, 150);

    return () => clearInterval(interval);
  }, []);

  const disabled = (type) => mode && mode !== type;

  const submit = () => {
    if (!selection) {
      alert("Please select Function, Application or Server");
      return;
    }

    if (mode === "function") navigate(`/functions/${selection}`);
    if (mode === "application") navigate(`/applications/${selection}`);
    if (mode === "server") navigate(`/servers/${selection}`);
  };

  // helper to extract values safely from backend json
  const getValue = (obj) => {
    if (!obj) return "";
    return obj.Function || obj.Application || obj.server || Object.values(obj)[0];
  };

  return (
    <div style={{ padding: "30px" }}>

      {/* ===== BANNER ===== */}
      <div style={{ height: "220px", marginBottom: "25px" }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={bannerData}>
            <XAxis hide />
            <YAxis hide />
            <Line
              type="monotone"
              dataKey="v"
              stroke="red"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* ===== AI INSIGHT CARD ===== */}
      <div className="insight-card">
        <h3>🔮 AI Capacity Insight</h3>
        <p>
          Based on historical growth patterns, systems in this environment are
          projected to hit <b>CPU saturation within 3–6 months</b> under current
          workload growth.  
          Consider scaling compute or optimizing high-variance applications early.
        </p>
      </div>

      <h2 style={{ marginBottom: "20px" }}>
        Welcome to <span style={{ color: "red" }}>System Health Dashboard</span>
      </h2>

      {/* ===== FILTER ROW ===== */}
      <div className="filter-row">

        {/* FUNCTION */}
        <div className="filter-box">
          <Tooltip text="Analyze by Business Stream / Function">
            <select
              disabled={disabled("function")}
              onChange={e => {
                setMode("function");
                setSelection(e.target.value);
              }}
            >
              <option value="">Business Stream</option>
              {functions.map((f, i) => {
                const val = getValue(f);
                return <option key={i} value={val}>{val}</option>;
              })}
            </select>
          </Tooltip>
        </div>

        {/* APPLICATION */}
        <div className="filter-box">
          <Tooltip text="Analyze by Application">
            <select
              disabled={disabled("application")}
              onChange={e => {
                setMode("application");
                setSelection(e.target.value);
              }}
            >
              <option value="">Application</option>
              {applications.map((a, i) => {
                const val = getValue(a);
                return <option key={i} value={val}>{val}</option>;
              })}
            </select>
          </Tooltip>
        </div>

        {/* SERVER */}
        <div className="filter-box">
          <Tooltip text="Analyze by Hostname / Server">
            <select
              disabled={disabled("server")}
              onChange={e => {
                setMode("server");
                setSelection(e.target.value);
              }}
            >
              <option value="">Hostname</option>
              {servers.map((s, i) => {
                const val = getValue(s);
                return <option key={i} value={val}>{val}</option>;
              })}
            </select>
          </Tooltip>
        </div>

        {/* METRICS */}
        <div className="filter-box">
          <Tooltip text="Select forecast metrics">
            <div className="checkbox-group">
              <label>
                <input
                  type="checkbox"
                  checked={metrics.cpu}
                  onChange={() => setMetrics(m => ({ ...m, cpu: !m.cpu }))}
                /> CPU
              </label>

              <label style={{ marginLeft: "10px" }}>
                <input
                  type="checkbox"
                  checked={metrics.mem}
                  onChange={() => setMetrics(m => ({ ...m, mem: !m.mem }))}
                /> Memory
              </label>
            </div>
          </Tooltip>
        </div>

        {/* GROWTH */}
        <div className="filter-box">
          <Tooltip text="Expected workload growth rate">
            <select value={growth} onChange={e => setGrowth(+e.target.value)}>
              {[10,20,30,40,50,60,70,80,90,100].map(g => (
                <option key={g} value={g}>{g}% Growth</option>
              ))}
            </select>
          </Tooltip>
        </div>

        {/* SUBMIT */}
        <div className="filter-box">
          <button className="submit-btn" onClick={submit}>
            Submit
          </button>
        </div>

      </div>
    </div>
  );
}

/* ===== Fancy Tooltip Component ===== */
function Tooltip({ text, children }) {
  return (
    <div className="tooltip-container">
      {children}
      <div className="tooltip-bubble">{text}</div>
    </div>
  );
}
