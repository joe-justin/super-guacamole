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
          <div className="explore">Explore Server</div>
        </div>
      </div>
    </div>
  )
}

export default function Servers(){
  const { application } = useParams()
  const [servers, setServers] = useState([])
  const nav = useNavigate()
  useEffect(()=>{
    async function load(){
      try {
        const resp = await fetch(`http://localhost:8000/api/servers/by_application/${encodeURIComponent(application)}`)
        const arr = await resp.json()
        const mapped = arr.map(s => {
          const color = s.status === 'healthy' ? 'green' : (s.status === 'needs_attention' ? 'amber' : 'red')
          return { title: s.server, subtitle: `${s.status}`, color }
        })
        setServers(mapped)
      } catch(e){ setServers([]) }
    }
    load()
  },[application])
  return (
    <div style={{padding:20}}>
      <h2>Servers for {application}</h2>
      <div className="cards-grid">
        {servers.map(s => <Card key={s.title} {...s} onClick={() => nav(`/server/${encodeURIComponent(s.title)}`)} />)}
      </div>
    </div>
  )
}
