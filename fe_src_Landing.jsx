import React, { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'

function Card({ title, color, subtitle, onClick }) {
  return (
    <div className={`flip-card ${color}`} onClick={onClick}>
      <div className="flip-inner">
        <div className="flip-front">
          <div className="card-title">{title}</div>
          <div className="card-sub">{subtitle}</div>
        </div>
        <div className="flip-back">
          <div className="explore">Explore Applications</div>
        </div>
      </div>
    </div>
  )
}

export default function Landing(){
  const [items, setItems] = useState([])
  const nav = useNavigate()
  useEffect(()=>{
    async function load(){
      try {
        const resp = await fetch('http://localhost:8000/api/functions')
        const json = await resp.json()
        // determine color by healthy_pct threshold of servers: >=95 green, 90-95 amber, <90 red
        const mapped = json.map(f => {
          const healthy = f.healthy_pct || 0
          const color = healthy >= 95 ? 'green' : (healthy >= 90 ? 'amber' : 'red')
          return { title: f.Function, color, subtitle: `${healthy}% healthy` }
        })
        setItems(mapped)
      } catch(e) { setItems([]) }
    }
    load()
  },[])
  return (
    <div className="landing-root">
      <header><h1>Functions</h1></header>
      <div className="cards-grid">
        {items.map(it => (
          <Card key={it.title} {...it} onClick={() => nav(`/applications/${encodeURIComponent(it.title)}`)} />
        ))}
      </div>
    </div>
  )
}
