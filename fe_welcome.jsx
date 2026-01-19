import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  ResponsiveContainer
} from "recharts";
import "./welcome.css";

export default function Welcome() {
  const navigate = useNavigate();

  const [functions, setFunctions] = useState([]);
  const [applications, setApplications] = useState([]);
  const [servers, setServers] = useState([]);

  const [mode, setMode] = useState(null);
  const [selection, setSelection] = useState("");

  const [metrics, setMetrics] = useState({ cpu: true, mem: false });
  const [growth, setGrowth] = useState(30);

  const [rawData, setRawData] = useState([]);
  const [viewIndex, setViewIndex] = useState(60);

  // Load dropdown data
  useEffect(() => {
    fetch("/api/functions").then(r => r.json()).then(d => setFunctions(d || []));
    fetch("/api/applications").then(r => r.json()).then(d => setApplications(d || []));
    fetch("/api/servers").then(r => r.json()).then(d => setServers(d || []));
  }, []);

  // Generate continuous animated banner data
  useEffect(() => {
    let t = 0;
    const interval = setInterval(() => {
      setRawData(prev => {
        const next = [...prev];
        if (next.length > 200) next.shift();

        next.push({
          t,
          v: 60 + Math.sin(t / 6) * 25 + Math.random() * 10
        });

        t++;
        return next;
      });
    }, 150);

    return () => clearInterval(interval);
  }, []);

  const visibleData = rawData.slice(
    Math.max(0, rawData.length - viewIndex),
    rawData.length
  );

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

  const getValue = (obj) => {
    if (!obj) return "";
    return obj.Function || obj.Application || obj.server || Object.values(obj)[0];
  };

  return (
    <div className="welcome-container">

      {/* ===== GLASS BANNER WITH SCRUB SLIDER ===== */}
      <div className="glass-banner">
        <div className="banner-header">
          <h2>Future Prediction Stream</h2>
          <div className="slider-box">
            <span>History Window</span>
            <input
              type="range"
              min="20"
              max="200"
              value={viewIndex}
              onChange={e => setViewIndex(+e.target.value)}
            />
          </div>
        </div>

        <div style={{ height: "200px" }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={visibleData}>
              <XAxis hide />
              <YAxis hide />
              <Line
                type="monotone"
                dataKey="v"
                stroke="red"
                strokeWidth={2.5}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* ===== AI INSIGHT GLASS CARD ===== */}
      <div className="glass-card insight-card">
        <h3>🔮 AI Capacity Insight</h3>
        <p>
          Forecast signals indicate rising utilization volatility.  
          At current growth rate, <b>CPU saturation risk exceeds 72%</b> within
          the next quarter.  
          Recommend early scaling for peak-variance workloads.
        </p>
      </div>

      <h2 className="welcome-title">
        Welcome to <span style={{ color: "red" }}>System Health Dashboard</span>
      </h2>

      {/* ===== FILTER GLASS ROW ===== */}
      <div className="glass-filter-row">

        <Filter label="Business Stream" tooltip="Analyze by Function / Business Stream">
          <select
            disabled={disabled("function")}
            onChange={e => {
              setMode("function");
              setSelection(e.target.value);
            }}
          >
            <option value="">Select</option>
            {functions.map((f, i) => {
              const val = getValue(f);
              return <option key={i} value={val}>{val}</option>;
            })}
          </select>
        </Filter>

        <Filter label="Application" tooltip="Analyze by Application">
          <select
            disabled={disabled("application")}
            onChange={e => {
              setMode("application");
              setSelection(e.target.value);
            }}
          >
            <option value="">Select</option>
            {applications.map((a, i) => {
              const val = getValue(a);
              return <option key={i} value={val}>{val}</option>;
            })}
          </select>
        </Filter>

        <Filter label="Hostname" tooltip="Analyze by Server / Hostname">
          <select
            disabled={disabled("server")}
            onChange={e => {
              setMode("server");
              setSelection(e.target.value);
            }}
          >
            <option value="">Select</option>
            {servers.map((s, i) => {
              const val = getValue(s);
              return <option key={i} value={val}>{val}</option>;
            })}
          </select>
        </Filter>

        <Filter label="Metrics" tooltip="Select metrics to forecast">
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
        </Filter>

        <Filter label="Growth Rate" tooltip="Expected workload growth rate">
          <select value={growth} onChange={e => setGrowth(+e.target.value)}>
            {[10,20,30,40,50,60,70,80,90,100].map(g => (
              <option key={g} value={g}>{g}%</option>
            ))}
          </select>
        </Filter>

        <div className="filter-submit">
          <button className="submit-btn" onClick={submit}>
            Forecast →
          </button>
        </div>
      </div>
    </div>
  );
}

/* ===== Reusable Filter with Fancy Tooltip ===== */
function Filter({ label, tooltip, children }) {
  return (
    <div className="filter-box tooltip-container">
      <label>{label}</label>
      {children}
      <div className="tooltip-bubble">{tooltip}</div>
    </div>
  );
}
