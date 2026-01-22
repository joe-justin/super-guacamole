import React, { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
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

export default function Applications() {
  const { functionName, applicationName } = useParams()
  const [apps, setApps] = useState([])
  const nav = useNavigate()

  useEffect(() => {
    async function load() {
      try {
        const srvResp = await fetch('http://localhost:8000/api/servers')
        const servers = await srvResp.json()

        let derivedFunction = functionName

        // 🔍 If application is selected, derive its Function
        if (applicationName) {
          const found = servers.find(
            s => s.Application.toLowerCase() === applicationName.toLowerCase()
          )
          if (found) derivedFunction = found.Function
        }

        // 📊 Build application map
        const appsMap = {}
        servers.forEach(s => {
          if (!derivedFunction || s.Function === derivedFunction) {
            const app = s.Application
            if (!appsMap[app]) appsMap[app] = { total: 0, healthy: 0 }
            appsMap[app].total += 1
            if (s.status === 'healthy') appsMap[app].healthy += 1
          }
        })

        let mapped = Object.entries(appsMap).map(([app, v]) => {
          const healthy_pct = Math.round((v.healthy / v.total) * 100)
          const color = healthy_pct >= 95 ? 'green' : (healthy_pct >= 90 ? 'amber' : 'red')
          return { title: app, color, subtitle: `${healthy_pct}% healthy (${v.total})` }
        })

        // 🎯 If specific application selected → show only that
        if (applicationName) {
          mapped = mapped.filter(a =>
            a.title.toLowerCase() === applicationName.toLowerCase()
          )
        }

        setApps(mapped)
      } catch (e) {
        console.error("Applications load failed", e)
        setApps([])
      }
    }

    load()
  }, [functionName, applicationName])

  return (
    <Layout>
      <div style={{ padding: 20 }}>
        <h2>
          {applicationName
            ? `Application: ${applicationName}`
            : functionName
              ? `Applications for ${functionName}`
              : "Applications"}
        </h2>

        <div className="cards-grid">
          {apps.map(a => (
            <Card
              key={a.title}
              {...a}
              onClick={() => nav(`/servers/${encodeURIComponent(a.title)}`)}
            />
          ))}
        </div>

        {applicationName && apps.length === 0 && (
          <div style={{ marginTop: 20, color: "gray" }}>
            No data found for application: {applicationName}
          </div>
        )}
      </div>
    </Layout>
  )
}
