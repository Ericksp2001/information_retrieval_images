import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import ImageSearch from './components/imagenSearch/imagenSearch'
function App() {
  const [count, setCount] = useState(0)

  return (
    <div className="App">
      <ImageSearch />
    </div>
  )
}

export default App
