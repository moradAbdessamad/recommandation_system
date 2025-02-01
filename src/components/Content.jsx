import { useState } from "react";
import TextSection from "./TextSection";
import ImageSection from "./ImageSection";

const Content = () => {
  const [activeSection, setActiveSection] = useState("text");

  return (
    <div className="content">
      
      <a href="/" class="home-link">
        {" "}
        <h2>Text-based Recommendation</h2>{" "}
      </a>

      <div className="method-selection">
        <button onClick={() => setActiveSection("text")}>
          Text-based Recommendation
        </button>
        
        <button onClick={() => setActiveSection("image")}>
          Image-based Recommendation
        </button>
      </div>

      {activeSection === "text" && <TextSection />}
      {activeSection === "image" && <ImageSection />}
    </div>
  );
};

export default Content;
