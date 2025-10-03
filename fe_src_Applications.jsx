import React, { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'

function Card({ title, color, subtitle, onClick }) {
  return (
    <div className={`flip-card ${color}`} onClick={onClick}>
      <div className="flip-inner">
        <div className="flip-front">
          <div className="card-title">{title}</div>
          <div className="card-sub">{subtitle}</div>
        </div>
        <div className="flip-back">
          <div className="explore">Explore Servers</div>
        </div>
      </div>
    </div>
  )
}

export default function Applications(){
  const { functionName } = useParams()
  const [apps, setApps] = useState([])
  const nav = useNavigate()
  useEffect(()=>{
    async function load(){
      try {
        const resp = await fetch('http://localhost:8000/api/applications')
        const json = await resp.json()
        const filtered = json.filter(a => a.Functions ? false : true) // keep all, we'll filter servers later
        // apps JSON has Application key; we need to filter by function via servers endpoint
        // Instead fetch servers and reduce by application where Function == functionName
        const srvResp = await fetch('http://localhost:8000/api/servers')
        const servers = await srvResp.json()
        const appsMap = {}
        servers.forEach(s => {
          if (s.Function === functionName) {
            const app = s.Application
            if (!appsMap[app]) appsMap[app] = { total:0, healthy:0 }
            appsMap[app].total += 1
            if (s.status === 'healthy') appsMap[app].healthy += 1
          }
        })
        const mapped = Object.entries(appsMap).map(([app, v]) => {
          const healthy_pct = Math.round((v.healthy / v.total) * 100)
          const color = healthy_pct >=95 ? 'green' : (healthy_pct >=90 ? 'amber' : 'red')
          return { title: app, color, subtitle: `${healthy_pct}% healthy (${v.total})`}
        })
        setApps(mapped)
      } catch(e){ setApps([]) }
    }
    load()
  },[functionName])
  return (
    <div style={{padding:20}}>
      <h2>Applications for {functionName}</h2>
      <div className="cards-grid">
        {apps.map(a => <Card key={a.title} {...a} onClick={() => nav(`/servers/${encodeURIComponent(a.title)}`)} />)}
      </div>
    </div>
  )
}
