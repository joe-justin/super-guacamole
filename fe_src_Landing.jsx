import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

export default function Landing() {
  const navigate = useNavigate();

  const [functions, setFunctions] = useState([]);
  const [applications, setApplications] = useState([]);
  const [servers, setServers] = useState([]);

  const [mode, setMode] = useState(null); // function | application | server
  const [selection, setSelection] = useState("");
  const [metrics, setMetrics] = useState({ cpu: true, mem: false });
  const [growth, setGrowth] = useState(30);

  useEffect(() => {
    fetch("/api/functions").then(r => r.json()).then(setFunctions);
    fetch("/api/applications").then(r => r.json()).then(setApplications);
    fetch("/api/servers").then(r => r.json()).then(setServers);
  }, []);

  const disabled = (type) => mode && mode !== type;

  const goNext = () => {
    if (!selection) return;

    if (mode === "function") navigate(`/functions/${selection}`);
    if (mode === "application") navigate(`/applications/${selection}`);
    if (mode === "server") navigate(`/servers/${selection}`);
  };

  return (
    <div style={{ display: "flex", padding: "40px", gap: "40px" }}>
      {/* LEFT PANE */}
      <div style={{ width: "45%" }}>
        <h1 style={{ fontSize: "32px" }}>
          <span style={{ color: "red" }}>S</span>ystem{" "}
          <span style={{ color: "red" }}>H</span>ealth{" "}
          <span style={{ color: "red" }}>D</span>ashboard
        </h1>

        <p style={{ marginTop: "20px", fontSize: "16px" }}>
          Predict future CPU & Memory behavior of servers using AI-driven
          forecasting models. Analyze by function, application, or hostname.
        </p>

        <img
          src="/dashboard.png"
          alt="dashboard"
          style={{ width: "100%", marginTop: "30px" }}
        />
      </div>

      {/* RIGHT PANE */}
      <div style={{ width: "55%" }}>
        <h3>Select Forecast Scope</h3>

        {/* FUNCTION */}
        <select
          disabled={disabled("function")}
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

        {/* APPLICATION */}
        <select
          disabled={disabled("application")}
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

        {/* SERVER */}
        <select
          disabled={disabled("server")}
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

        {/* METRICS */}
        <div style={{ marginTop: "20px" }}>
          <label>
            <input
              type="checkbox"
              checked={metrics.cpu}
              onChange={() =>
                setMetrics(m => ({ ...m, cpu: !m.cpu }))
              }
            />
            CPU
          </label>

          <label style={{ marginLeft: "20px" }}>
            <input
              type="checkbox"
              checked={metrics.mem}
              onChange={() =>
                setMetrics(m => ({ ...m, mem: !m.mem }))
              }
            />
            Memory
          </label>
        </div>

        {/* GROWTH */}
        <div style={{ marginTop: "20px" }}>
          Growth Rate (%)
          <select value={growth} onChange={e => setGrowth(+e.target.value)}>
            {[10,20,30,40,50,60,70,80,90,100].map(g => (
              <option key={g} value={g}>{g}%</option>
            ))}
          </select>
        </div>

        <button
          onClick={goNext}
          style={{
            marginTop: "30px",
            padding: "10px 20px",
            background: "red",
            color: "white",
            border: "none",
            cursor: "pointer"
          }}
        >
          Explore
        </button>
      </div>
    </div>
  );
}
