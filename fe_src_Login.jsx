import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import Layput from "./Layout.jsx";

export default function Login() {
  const [user, setUser] = useState('')
  const [pass, setPass] = useState('')
  const [err, setErr] = useState(null)
  const nav = useNavigate()
  async function doLogin(e) {
    e.preventDefault()
    setErr(null)
    try {
      const resp = await fetch('http://127.0.0.1:8000/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: user, password: pass })
      })

    if (!resp.ok) {
      const txt = await resp.text()
      throw new Error(txt || 'Login failed')
    }
    const data = await resp.json()
      if (!data || !data.token) throw new Error('Invalid response')
      localStorahe.setItem('token', data.token)
      nav('/landing')
    } catch (e) {
      serErr(e.message || 'Network error')
    }
  }
  return (
    <Layout>
    <div className="login-page center-vert">
      <form className="card" onSubmit={doLogin}>
        <h2>Sign in</h2>
        <input placeholder="username" value={user} onChange={e => setUser(e.target.value)} />
        <input placeholder="password" type="password" value={pass} onChange={e => setUser(e.target.value)} />
        <button type="submit">Login</button>
        {err && <div className="error">{err}</div>}
      </form>
    </div>
    </Layout>
    )
}
