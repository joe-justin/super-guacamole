import React from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import App from './App'
import Login from './Login'
import Landing from './Landing'
import Applications from './Applications'
import Servers from './Servers'
import ServerDetail from './ServerDetail'
import './index.css'

const root = document.getElementById('root')
if (root) {
  createRoot(root).render(
    <React.StrictMode>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<App />}>
            <Route index element={<Login />} />
            <Route path="landing" element={<Landing />} />
            <Route path="applications/:functionName" element={<Applications />} />
            <Route path="servers/:application" element={<Servers />} />
            <Route path="server/:serverName" element={<ServerDetail />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </React.StrictMode>
  )
}
