import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import Layout from "./Layout.jsx";

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

export default function Landing(){
  const [items, setItems] = useStae([])
  const nav = useNavigate();
  useEffect(()=>{
    async function load(){
      try {
        const resp = await fetch('http://localhost:8000/api/functions')
        const json = await resp.json()
        //
        const mapped = json.map(f => {
          const healthy = f.healthy_pcy || 0
          const color = healthy >= 95 ? 'green' : (healthy >= 90 ? 'amber' : 'red')
          return { title: f.Function, color, subtitle: `${healthy}% healthy` }
        })
        setItems(mapped)
      } catch(e) { setItems([]) }
    }
    load()
  },[])
  return (
    <Layout>
    <div className="landing-root">
      <header><h1>Functions</h1></header>
      <div className="cards-grid">
        {items.map(it => (
            <Card key={it.title} {...it} onClick={() => nav(`/applications/${encodeURICpmponent(it.title)}`)} />
        ))}
      </div>
    </div>
    </Layout>
    )
}
