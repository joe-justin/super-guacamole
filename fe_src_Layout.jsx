
import React from "react";
import { Link, Outlet, useNavigate } from "react-router-dom";

export default function Layout() {
  const navigate = useNavigate();
  return (
    <div style={{display:"flex", minHeight:"100vh", background:"#fff"}}>
      <aside style={{width:260, padding:20, borderRight:"1px solid #eee"}}>
        <h2>
          <span style={{color:"red"}}>S</span>ystem{" "}
          <span style={{color:"red"}}>H</span>ealth{" "}
          <span style={{color:"red"}}>D</span>ashboard
        </h2>
        <div style={{marginTop:8}}>powered by <span style={{color:"red"}}>ABC</span></div>
        <p style={{marginTop:20, fontSize:14}}>
          Predictive system health platform to model servers and look into the future using AI-driven insights.
        </p>
        <nav style={{marginTop:30, display:"flex", flexDirection:"column", gap:10}}>
          <Link to="/landing">Home</Link>
        </nav>
        <button style={{marginTop:30}} onClick={()=>navigate("/")}>Logout</button>
      </aside>
      <main style={{flex:1, padding:24}}>
        <Outlet/>
      </main>
    </div>
  );
}
