
import React from "react";
import { useNavigate } from "react-router-dom";

export default function Landing(){
  const nav = useNavigate();
  return (
    <div style={{display:"flex", gap:40}}>
      <div style={{flex:1}}>
        <h1>Welcome</h1>
        <p>
          This tool models any server to predict future machine metrics and provides AI insights.
        </p>
      </div>
      <div style={{flex:1, display:"grid", gridTemplateColumns:"1fr 1fr", gap:20}}>
        <button onClick={()=>nav("/functions")}>Business Stream</button>
        <button onClick={()=>nav("/functions")}>Business Cluster</button>
        <button onClick={()=>nav("/applications")}>Application</button>
        <button onClick={()=>nav("/servers")}>Hostname</button>
      </div>
    </div>
  );
}
