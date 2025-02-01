const ToggleContainer = ({ toggleSidebar }) => {
    return (
      <div className="toggle-container">
        <button className="toggle-button" onClick={toggleSidebar}>
          ☰
        </button>
      </div>
    )
  }
  
export default ToggleContainer
  
  