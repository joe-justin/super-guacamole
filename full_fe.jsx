# Project: frontend (split components)

This document contains the full set of React + Tailwind source files to replace `src/` in your existing frontend. Files are separated by markers in the format:

==== FILE: <relative path> ====
<file content>

Place each file exactly as listed under `src/`.

----

==== FILE: main.jsx ====
import React from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import './index.css'

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>
)

==== FILE: App.jsx ====
import React from 'react'
import { Routes, Route } from 'react-router-dom'
import Layout from './Layout'
import Login from './Login'
import Landing from './pages/Landing'
import FunctionsPage from './pages/FunctionsPage'
import ApplicationsPage from './pages/ApplicationsPage'
import ServersPage from './pages/ServersPage'
import ServerDetail from './pages/ServerDetail'

export default function App(){
  return (
    <Routes>
      <Route path="/login" element={<Login/>} />
      <Route path="/*" element={<Layout/>}>
        <Route index element={<Landing/>} />
        <Route path="landing" element={<Landing/>} />
        <Route path="functions" element={<FunctionsPage/>} />
        <Route path="applications" element={<ApplicationsPage/>} />
        <Route path="servers" element={<ServersPage/>} />
        <Route path="servers/:serverName" element={<ServerDetail/>} />
      </Route>
    </Routes>
  )
}

==== FILE: Layout.jsx ====
import React from 'react'
import { Outlet, useNavigate } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import TopBar from './components/TopBar'

export default function Layout(){
  const navigate = useNavigate()
  const onLogout = ()=>{ navigate('/login') }
  return (
    <div className="min-h-screen bg-white text-black">
      <div className="flex">
        <Sidebar onLogout={onLogout} />
        <div className="flex-1 flex flex-col">
          <TopBar />
          <main className="p-6">
            <Outlet />
          </main>
        </div>
      </div>
    </div>
  )
}

==== FILE: components/Sidebar.jsx ====
import React from 'react'
import { Link } from 'react-router-dom'
import Logo from './Logo'

export default function Sidebar({onLogout}){
  return (
    <aside className="w-64 border-r border-gray-200 p-4 bg-white h-screen sticky top-0 flex flex-col">
      <div className="mb-6"><Logo/></div>
      <nav className="flex flex-col gap-2 text-sm">
        <Link to="/landing" className="px-3 py-2 rounded hover:bg-gray-50">Home</Link>
        <Link to="/functions" className="px-3 py-2 rounded hover:bg-gray-50">Business Stream / Cluster</Link>
        <Link to="/applications" className="px-3 py-2 rounded hover:bg-gray-50">Applications</Link>
        <Link to="/servers" className="px-3 py-2 rounded hover:bg-gray-50">Servers</Link>
      </nav>
      <div className="mt-auto pt-6">
        <button onClick={onLogout} className="w-full bg-red-600 text-white py-2 rounded">Logout</button>
      </div>
    </aside>
  )
}

==== FILE: components/TopBar.jsx ====
import React from 'react'

export default function TopBar(){
  return (
    <header className="flex items-center justify-between px-6 py-3 border-b border-gray-200 bg-white">
      <div className="text-sm text-gray-700">AI-powered forecasting & insights for your servers</div>
      <div className="text-xs text-gray-500">v1.0</div>
    </header>
  )
}

==== FILE: components/Logo.jsx ====
import React from 'react'

export default function Logo(){
  return (
    <div className="flex items-center gap-3">
      <img src="/assets/logo.png" alt="logo" className="w-10 h-10 object-contain" />
      <div>
        <div className="text-xl font-bold leading-none">
          <span className="text-red-600">S</span>ystem <span className="text-red-600">H</span>ealth <span className="text-red-600">D</span>ashboard
        </div>
        <div className="text-xs text-gray-600">powered by <span className="text-red-600">ABC</span></div>
      </div>
    </div>
  )
}

==== FILE: pages/Landing.jsx ====
import React from 'react'
import { motion } from 'framer-motion'

export default function Landing(){
  return (
    <div className="grid grid-cols-2 gap-6">
      <motion.div initial={{opacity:0, x:-20}} animate={{opacity:1, x:0}} className="p-6 bg-white rounded shadow-sm">
        <h2 className="text-2xl font-semibold mb-3">Explore the predictive intelligence</h2>
        <p className="text-gray-700 mb-6">This tool models every server and looks into the future for CPU, memory and network metrics. Use the buttons to explore by Business Stream, Cluster, Application or Hostname.</p>
        <div className="grid grid-cols-2 gap-3">
          <a href="/functions" className="py-3 rounded border border-gray-200 hover:shadow-sm text-center">Business Stream</a>
          <a href="/functions" className="py-3 rounded border border-gray-200 hover:shadow-sm text-center">Business Cluster</a>
          <a href="/applications" className="py-3 rounded border border-gray-200 hover:shadow-sm text-center">Application</a>
          <a href="/servers" className="py-3 rounded border border-gray-200 hover:shadow-sm text-center">Hostname</a>
        </div>
      </motion.div>

      <motion.div initial={{opacity:0, x:20}} animate={{opacity:1, x:0}} className="p-6 flex items-center justify-center bg-white rounded shadow-sm">
        <div className="w-full">
          <img src="/assets/ai_dashboard.png" alt="ai" className="w-full h-64 object-contain" />
          <p className="mt-4 text-gray-600">AI-driven observability: predictive alerts, anomaly detection and capacity planning.</p>
        </div>
      </motion.div>
    </div>
  )
}

==== FILE: pages/FunctionsPage.jsx ====
import React from 'react'
import ListPage from '../shared/ListPage'

export default function FunctionsPage(){
  return <ListPage title={'Business Streams / Clusters'} apiPath={'/api/functions'} filterLabel={'Business Stream / Cluster'} />
}

==== FILE: pages/ApplicationsPage.jsx ====
import React from 'react'
import ListPage from '../shared/ListPage'

export default function ApplicationsPage(){
  return <ListPage title={'Applications'} apiPath={'/api/applications'} filterLabel={'Application'} />
}

==== FILE: pages/ServersPage.jsx ====
import React from 'react'
import ListPage from '../shared/ListPage'

export default function ServersPage(){
  return <ListPage title={'Servers'} apiPath={'/api/servers'} filterLabel={'Hostname'} />
}

==== FILE: shared/ListPage.jsx ====
import React from 'react'
import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'

function FilterBar({label, options, value, onChange}){
  return (
    <div className="flex items-center gap-3 mb-4">
      <label className="text-sm font-medium w-36">{label}</label>
      <select value={value||""} onChange={e=>onChange(e.target.value)} className="border rounded px-3 py-2">
        <option value="">All</option>
        {options.map(o=> <option key={o} value={o}>{o}</option>)}
      </select>
    </div>
  )
}

export default function ListPage({title, apiPath, filterLabel}){
  const [items, setItems] = React.useState([]);
  const [filter, setFilter] = React.useState("");
  const [filterOptions, setFilterOptions] = React.useState([]);
  const [loading, setLoading] = React.useState(true);
  const navigate = useNavigate();

  React.useEffect(()=>{
    let mounted = true
    async function load(){
      setLoading(true);
      try{
        const resp = await fetch(apiPath);
        const data = await resp.json();
        if(!mounted) return
        setItems(data);
        const names = data.map(d => (d.Function || d.Application || d.server || d.name || d));
        setFilterOptions(Array.from(new Set(names)));
      }catch(e){
        setItems([]);
      }
      setLoading(false);
    }
    load();
    return ()=>{ mounted=false }
  },[apiPath]);

  const filtered = items.filter(it=>{
    if(!filter) return true;
    const name = (it.Function||it.Application||it.server||it.name||it).toString();
    return name === filter;
  });

  return (
    <div className="p-6 bg-white rounded shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold">{title}</h3>
          <button onClick={()=>window.history.back()} className="text-sm text-red-600 underline">Back</button>
        </div>
      </div>
      <FilterBar label={filterLabel} options={filterOptions} value={filter} onChange={setFilter} />
      {loading ? <div>Loading...</div> : (
        <div className="grid grid-cols-3 gap-4">
          {filtered.map((it, idx)=>{
            const name = (it.Function||it.Application||it.server||it.name||it).toString();
            return (
              <motion.div key={idx} whileHover={{scale:1.02}} className="p-4 border rounded">
                <div className="flex items-center justify-between">
                  <div className="font-semibold">{name}</div>
                  <div className="text-xs text-gray-500">{it.total?`${it.total} servers`:''}</div>
                </div>
                <div className="mt-4 flex gap-2">
                  <button onClick={()=>{
                    // Navigate to server-detail or application view
                    if(apiPath.includes('/servers')){
                      navigate(`/servers/${encodeURIComponent(name)}`)
                    } else if(apiPath.includes('/applications')){
                      navigate(`/applications/${encodeURIComponent(name)}`)
                    } else {
                      navigate(`/functions/${encodeURIComponent(name)}`)
                    }
                  }} className="text-sm px-3 py-1 border rounded">Explore</button>
                </div>
              </motion.div>
            )
          })}
        </div>
      )}
    </div>
  )
}

==== FILE: pages/ServerDetail.jsx ====
import React, { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import Plot from 'react-plotly.js'

export default function ServerDetail(){
  const { serverName } = useParams()
  const [rows, setRows] = useState(null)
  useEffect(()=>{
    async function load(){
      try{
        const resp = await fetch(`/api/predictions/csv/${encodeURIComponent(serverName)}`)
        if(!resp.ok) throw new Error('no csv')
        const text = await resp.text()
        const r = text.trim().split('
')
        const headers = r[0].split(',').map(h=>h.trim())
        const data = r.slice(1).map(line=>{
          const cols = line.split(',')
          const obj = {}
          headers.forEach((h,i)=>{ obj[h]=cols[i] })
          return obj
        }).filter(x=>x.timestamp)
        setRows(data)
      }catch(e){ setRows(null) }
    }
    load()
  },[serverName])

  if(!rows) return <div className="p-6 bg-white rounded shadow-sm">Loading...</div>

  // parse arrays for plotting
  const timestamps = rows.map(r=>r.timestamp)
  const cpu_actual = rows.map(r=>parseFloat(r.cpu_actual||r.cpu_current||NaN)).map(v=>isNaN(v)?null:v)
  const mem_actual = rows.map(r=>parseFloat(r.mem_actual||r.mem_current||NaN)).map(v=>isNaN(v)?null:v)
  const cpu_pred = rows.map(r=>parseFloat(r.cpu_predicted||r.cpu_pred_combined||NaN)).map(v=>isNaN(v)?null:v)
  const mem_pred = rows.map(r=>parseFloat(r.mem_predicted||r.mem_pred_combined||NaN)).map(v=>isNaN(v)?null:v)

  return (
    <div className="p-6 bg-white rounded shadow-sm">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Server: {serverName}</h3>
        <button onClick={()=>window.history.back()} className="text-sm text-red-600 underline">Back</button>
      </div>

      <div className="mt-4">
        <Plot
          data=[
            { x: timestamps, y: cpu_actual, type: 'scatter', mode: 'lines', name: 'CPU actual' },
            { x: timestamps, y: cpu_pred, type: 'scatter', mode: 'lines', name: 'CPU predicted' },
            { x: timestamps, y: mem_actual, type: 'scatter', mode: 'lines', name: 'MEM actual' },
            { x: timestamps, y: mem_pred, type: 'scatter', mode: 'lines', name: 'MEM predicted' }
          ]
          layout={{ width: 1000, height: 450, title: `${serverName} - CPU & MEM` }}
        />
      </div>
    </div>
  )
}

==== FILE: Login.jsx ====
// Keep your existing Login.jsx. If you want it styled to match, replace with your current file.
import React from 'react'
export default function Login(){
  return (<div className="p-6">Please keep your existing Login.jsx (unchanged)</div>)
}

==== FILE: index.css ====
@tailwind base;
@tailwind components;
@tailwind utilities;

html, body, #root { height: 100%; }
body { background: white; }

/* small utility to keep charts responsive */
.plotly-graph-div { max-width: 100%; }


====== api.jsx ======
  // src/api.js
// Centralized API helper — controls backend connection

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export async function fetchAPI(path, options = {}) {
  const url = `${API_BASE}${path}`;
  const resp = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Fetch failed (${resp.status}): ${text}`);
  }
  try {
    return await resp.json();
  } catch {
    return await resp.text(); // for CSV/text endpoints
  }
}

// Example POST helper
export async function postAPI(path, body) {
  return fetchAPI(path, {
    method: "POST",
    body: JSON.stringify(body),
  });
}


==== FILE: assets/README.txt ====
Place two images in public/assets/:
 - logo.png  (small logo shown top-left)
 - ai_dashboard.png  (illustration used on landing page)

----

# Integration / install notes

1) Replace your `src/` contents with the files above (create folders `components`, `pages`, `shared`).
2) Copy images to `public/assets/logo.png` and `public/assets/ai_dashboard.png`.
3) Ensure packages installed: run
   npm install framer-motion react-plotly.js plotly.js-basic-dist react-router-dom

4) Tailwind: keep your existing Tailwind setup. If not present, follow Tailwind + Vite setup.
5) Start dev server: npm run dev

If you want, I can also produce a git-style patch or a zip of just the `src/` files. Let me know.
