import React from "react";
import { Link, useNavigate } from "react-router-dom";

export default function Layout({ children }) {
  const navigate = useNavigate();

  const handleLogout = () => {
    // Clear any stored tokens or session data if used
    localStorage.clear();
    sessionStorage.clear();
    // Redirect to login
    navigate("/login");
  };

  return (
    <div>
      <header
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between", // space between logo/title and logout
          padding: "10px 20px",
          backgroundColor: "#f8f9fa",
          borderBottom: "1px solid #ddd",
        }}
      >
        <Link
          to="/landing"
          style={{
            display: "flex",
            alignItems: "center",
            textDecoration: "none",
          }}
        >
          <img
            src="/logo.png"
            alt="Company Logo"
            style={{ height: 40, marginRight: 12 }}
          />
          <div style={{ display: "flex", flexDirection: "column" }}>
            <span style={{ fontSize: 22, fontWeight: 600, lineHeight: "26px" }}>
              <span style={{ color: "red" }}>S</span>ystem{" "}
              <span style={{ color: "red" }}>H</span>ealth{" "}
              <span style={{ color: "red" }}>D</span>ashboard
            </span>
            <span style={{ marginTop: 6, fontSize: 13, color: "#555" }}>
              powered by <span style={{ color: "red" }}>ABC</span>
            </span>
          </div>
        </Link>

        {/* ✅ Logout Button */}
        <button
          onClick={handleLogout}
          style={{
            backgroundColor: "red",
            color: "white",
            border: "none",
            padding: "8px 16px",
            borderRadius: 6,
            fontSize: 14,
            cursor: "pointer",
            fontWeight: "500",
          }}
        >
          Logout
        </button>
      </header>

      <main>{children}</main>
    </div>
  );
}
