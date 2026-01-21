import { useEffect, useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import "./welcome.css";

export default function Welcome() {
  const navigate = useNavigate();
  const canvasRef = useRef(null);

  const [functions, setFunctions] = useState([]);
  const [applications, setApplications] = useState([]);
  const [servers, setServers] = useState([]);

  const [mode, setMode] = useState(null);
  const [selection, setSelection] = useState("");

  const [metrics, setMetrics] = useState({ cpu: true, mem: false });
  const [growth, setGrowth] = useState(30);
  const [viewIndex, setViewIndex] = useState(60);

  useEffect(() => {
    fetch("/api/functions").then(r => r.json()).then(d => setFunctions(d || []));
    fetch("/api/applications").then(r => r.json()).then(d => setApplications(d || []));
    fetch("/api/servers").then(r => r.json()).then(d => setServers(d || []));
  }, []);

  /* ===========================
     DIGITAL HORIZON + AI GRID
     =========================== */
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    let width, height, frame = 0;

    const resize = () => {
      width = canvas.width = canvas.offsetWidth;
      height = canvas.height = canvas.offsetHeight;
    };
    resize();
    window.addEventListener("resize", resize);

    const draw = () => {
      frame++;
      ctx.clearRect(0, 0, width, height);

      // Background glow
      ctx.fillStyle = "rgba(255,255,255,0.03)";
      ctx.fillRect(0, 0, width, height);

      // Horizon line
      ctx.beginPath();
      ctx.moveTo(0, height * 0.65);
      ctx.lineTo(width, height * 0.65);
      ctx.strokeStyle = "rgba(255,0,0,0.4)";
      ctx.lineWidth = 2;
      ctx.stroke();

      // Perspective grid
      for (let i = 0; i < 20; i++) {
        const y = height * 0.65 + i * 8;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.strokeStyle = "rgba(255,0,0,0.05)";
        ctx.stroke();
      }

      for (let i = -20; i <= 20; i++) {
        ctx.beginPath();
        ctx.moveTo(width / 2 + i * 30, height * 0.65);
        ctx.lineTo(width / 2 + i * 150, height);
        ctx.strokeStyle = "rgba(255,0,0,0.05)";
        ctx.stroke();
      }

      // Neural pulses
      for (let i = 0; i < 40; i++) {
        const x = (i * 40 + frame * 1.5) % width;
        const y = height * 0.65 - Math.sin((x + frame) / 40) * 40;
        ctx.beginPath();
        ctx.arc(x, y, 2.2, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(255,0,0,0.7)";
        ctx.fill();
      }

      requestAnimationFrame(draw);
    };

    draw();
    return () => window.removeEventListener("resize", resize);
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

  const getValue = (obj) =>
    obj?.Function || obj?.Application || obj?.server || Object.values(obj)[0];

  return (
    <div className="welcome-container">

      {/* ===== DIGITAL HORIZON BANNER ===== */}
      <div className="glass-banner">
        <div className="banner-header">
          <h2>Digital Horizon — AI Prediction Field</h2>
          <div className="slider-box">
            <span>Time Focus</span>
            <input
              type="range"
              min="20"
              max="200"
              value={viewIndex}
              onChange={e => setViewIndex(+e.target.value)}
            />
          </div>
        </div>

        <canvas
          ref={canvasRef}
          style={{
            width: "100%",
            height: "200px",
            borderRadius: "12px"
          }}
        />
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
          <select disabled={disabled("function")}
            onChange={e => { setMode("function"); setSelection(e.target.value); }}>
            <option value="">Select</option>
            {functions.map((f, i) => {
              const val = getValue(f);
              return <option key={i} value={val}>{val}</option>;
            })}
          </select>
        </Filter>

        <Filter label="Application" tooltip="Analyze by Application">
          <select disabled={disabled("application")}
            onChange={e => { setMode("application"); setSelection(e.target.value); }}>
            <option value="">Select</option>
            {applications.map((a, i) => {
              const val = getValue(a);
              return <option key={i} value={val}>{val}</option>;
            })}
          </select>
        </Filter>

        <Filter label="Hostname" tooltip="Analyze by Server / Hostname">
          <select disabled={disabled("server")}
            onChange={e => { setMode("server"); setSelection(e.target.value); }}>
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
              <input type="checkbox" checked={metrics.cpu}
                onChange={() => setMetrics(m => ({ ...m, cpu: !m.cpu }))} /> CPU
            </label>
            <label style={{ marginLeft: "10px" }}>
              <input type="checkbox" checked={metrics.mem}
                onChange={() => setMetrics(m => ({ ...m, mem: !m.mem }))} /> Memory
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
          <button className="submit-btn" onClick={submit}>Forecast →</button>
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
