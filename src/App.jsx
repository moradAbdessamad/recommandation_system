import React, { useState } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

import ToggleContainer from "./components/ToggleContainer";
import HistorySidebar from "./components/HistorySidebar";
import Content from "./components/Content";
import MovieDetails from "./components/MovieDetails"; 
import "./App.css";

const App = () => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  return (
    <Router>
      <div className="main-container">

        <ToggleContainer toggleSidebar={toggleSidebar} />
        <HistorySidebar isOpen={isSidebarOpen} />

        <Routes>
          <Route path="/" element={<Content />} />
          <Route path="/movie/:id" element={<MovieDetails />} />
        </Routes>

      </div>
    </Router>
  );
};

export default App;
