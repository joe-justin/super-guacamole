import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import LandingPage from "./pages/LandingPage";
import ServerDetailPage from "./pages/ServerDetailPage";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/server/:server" element={<ServerDetailPage />} />
      </Routes>
    </Router>
  );
}

export default App;
