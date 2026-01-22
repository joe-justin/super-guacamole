import React, { useEffect, useState } from 'react'
import { useNavigate, useParams, useSearchParams } from 'react-router-dom'
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
          <div className="explore">Explore Applications</div>
        </div>
      </div>
    </div>
  )
}

export default function Landing() {
  const [items, setItems] = useState([])
  const nav = useNavigate()

  const { functionName } = useParams()
  const [searchParams] = useSearchParams()
  const functionFromQuery = searchParams.get("function")

  const selectedFunction = functionName || functionFromQuery

  useEffect(() => {
    async function load() {
      try {
        const resp = await fetch('http://localhost:8000/api/functions')
        const json = await resp.json()

        let mapped = json.map(f => {
          const healthy = f.healthy_pct || 0
          const color = healthy >= 95 ? 'green' : (healthy >= 90 ? 'amber' : 'red')
          return { title: f.Function, color, subtitle: `${healthy}% healthy` }
        })

        // ✅ If parameter exists, filter to only that Function
        if (selectedFunction) {
          mapped = mapped.filter(it =>
            it.title.toLowerCase() === selectedFunction.toLowerCase()
          )
        }

        setItems(mapped)
      } catch (e) {
        console.error("Failed loading functions", e)
        setItems([])
      }
    }
    load()
  }, [selectedFunction])

  return (
    <Layout>
      <div className="landing-root">
        <header>
          <h1>
            {selectedFunction ? `Function: ${selectedFunction}` : "Functions"}
          </h1>
        </header>

        <div className="cards-grid">
          {items.map(it => (
            <Card
              key={it.title}
              {...it}
              onClick={() => nav(`/applications/${encodeURIComponent(it.title)}`)}
            />
          ))}
        </div>

        {selectedFunction && items.length === 0 && (
          <div style={{ marginTop: 20, color: "gray" }}>
            No data found for function: {selectedFunction}
          </div>
        )}
      </div>
    </Layout>
  )
}
