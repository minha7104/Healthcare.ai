import Header from './components/Header'
import Sidebar from './components/Sidebar'
import { Routes, Route } from 'react-router-dom'

import Dashboard from './pages/Dashboard'
import HealthRecords from './pages/HealthRecords'
import Predictions from './pages/Predictions'

import './App.css'

function App() {
  return (
    <div>
      <Header />
      <div className="app-layout">
        <Sidebar />
        <main>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/records" element={<HealthRecords />} />
            <Route path="/predictions" element={<Predictions />} />
          </Routes>
        </main>
      </div>
    </div>
  )
}

export default App
