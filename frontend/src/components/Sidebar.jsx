import { NavLink } from 'react-router-dom'
import './Sidebar.css'

function Sidebar() {
  return (
    <aside>
      <ul>
        <li>
          <NavLink to="/" end className="nav-link">
            Dashboard
          </NavLink>
        </li>
        <li>
          <NavLink to="/records" className="nav-link">
            Health Records
          </NavLink>
        </li>
        <li>
          <NavLink to="/predictions" className="nav-link">
            AI Predictions
          </NavLink>
        </li>
      </ul>
    </aside>
  )
}

export default Sidebar
