import React from "react";
import { Link } from "react-router-dom";

export default function Layout({ children }) {
  return (
    <div>
      <header
        style={{
          display: "flex",
          alignItems: "center",
          padding: "10px 20px",
          backgroundColor: "#f8f9fa",
          borderBottom: "1px solid #ddd",
        }}
      >
        <Link to="/landing" style={{ display: "flex", alignItems: "center", textDecoration: "none" }}>
          <img
            src="/logo.png"
            alt="Company Logo"
            style={{ height: 40, marginRight: 12 }}
          />
          <span style={{ fontSize: 20, fontWeight: "bold", color: "#333" }}>
            System Health Dashboard
          </span>
        </Link>
      </header>

      <main>{children}</main>
    </div>
  );
}
