import React, { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import Plot from 'react-plotly.js'

export default function ServerDetail(){
  const { serverName } = useParams()
  const [historic, setHistoric] = useState(null)
  const [predicted, setPredicted] = useState(null)
  useEffect(()=>{
    async function load(){
      try {
        const resp = await fetch(`http://localhost:8000/api/predictions/csv/${encodeURIComponent(serverName)}`)
        if (!resp.ok) throw new Error('no csv')
        const text = await resp.text()
        const rows = text.split('\\n').filter(Boolean)
        const df = rows.slice(1).map(r => {
          const cols = r.split(',')
          return {
            timestamp: cols[0],
            cpu_pred_combined: parseFloat(cols[3]),
            mem_pred_combined: parseFloat(cols[6])
          }
        })
        setPredicted(df)
        // Also fetch historical from servers_all or individual server if desired
        const infoResp = await fetch(`http://localhost:8000/api/server/${encodeURIComponent(serverName)}`)
        const info = await infoResp.json()
        // historic data is not shipped via API; frontend can request a historical CSV if you provide
        setHistoric(null)
      } catch(e){ setPredicted(null); setHistoric(null) }
    }
    load()
  },[serverName])
  if (!predicted) return <div style={{padding:20}}>Loading predictions... <br/><Link to="/landing">Back</Link></div>
  const times = predicted.map(r => r.timestamp)
  const cpu = predicted.map(r => r.cpu_pred_combined)
  const mem = predicted.map(r => r.mem_pred_combined)
  return (
    <div style={{padding:20}}>
      <h2>Server: {serverName}</h2>
      <Plot
        data={[
          { x: times, y: cpu, type: 'scatter', mode: 'lines', name: 'CPU predicted' },
          { x: times, y: mem, type: 'scatter', mode: 'lines', name: 'Memory predicted' }
        ]}
        layout={{ width: 1000, height: 450, title: `${serverName} - predicted CPU & Memory` }}
      />
      <div style={{marginTop:16}}><Link to="/landing">Back to functions</Link></div>
    </div>
  )
}
