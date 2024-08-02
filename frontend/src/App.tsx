import { useState, useEffect } from 'react';
import ImagenSearch from './components/imagenSearch/imagenSearch' // Asegúrate de que la ruta sea correcta

interface Result {
  label: string;
  index: number;
  distance: number;
}

function App() {
  const [results, setResults] = useState<Result[]>([]);

  useEffect(() => {
    // Aquí deberías cargar tus resultados, por ejemplo, de una API
    // Este es solo un ejemplo, reemplázalo con tu lógica real
    const mockResults: Result[] = [
      { label: 'stapler', index: 1, distance: 0.1234 },
      { label: 'sunflower', index: 2, distance: 0.2345 },
      { label: 'bird', index: 3, distance: 0.3456 },
    ];
    setResults(mockResults);
  }, []);

  return (
    <div className="App">
      <h1>Image Search Results</h1>
      <ImagenSearch results={results} />
    </div>
  );
}

export default App;