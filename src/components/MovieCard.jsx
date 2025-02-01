import "./MovieCard.css"

const MovieCard = ({ title, poster, voteAverage, lastUpdate, runningTime }) => {
  return (
    <div className="movie-card">
      <img src={poster || "/placeholder.svg"} alt={`${title} Poster`} className="movie-poster" />
      <div className="movie-details">
        <div className="title-row">
          <div className="movie-title">{title}</div>
          <div className="vote-average">{voteAverage}/10</div>
        </div>
        <div className="movie-info">
          <div className="info-group">
            <span className="info-label">Last update</span>
            <span className="info-value">{lastUpdate}</span>
          </div>
          <div className="info-group right">
            <span className="info-label">Running time</span>
            <span className="info-value">{runningTime}</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default MovieCard

