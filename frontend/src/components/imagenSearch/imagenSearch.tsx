import React, { useState } from 'react';
import './imagenSearch.css';
import logo from '../../assets/images/logoimageretrievalsystem-removebg.png';

interface SimilarImage {
  path: string;
  similarity: number;
}

const ImagenSearch: React.FC = () => {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [imagePreview, setImagePreview] = useState<string | null>(null);
    const [fileName, setFileName] = useState<string>("");
    const [similarImages, setSimilarImages] = useState<SimilarImage[]>([]);
    const [isLoading, setIsLoading] = useState<boolean>(false);

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files && event.target.files[0]) {
            const file = event.target.files[0];
            setSelectedFile(file);
            setFileName(file.name);
            const reader = new FileReader();
            reader.onloadend = () => {
                setImagePreview(reader.result as string);
            };
            reader.readAsDataURL(file);
        }
    };

    const handleSearch = async () => {
        if (selectedFile) {
            setIsLoading(true);
            const formData = new FormData();
            formData.append('file', selectedFile);

            try {
                const response = await fetch('http://localhost:5000/search_image', {
                    method: 'POST',
                    body: formData,
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                if (data.error) {
                    throw new Error(data.error);
                }
                setSimilarImages(data.similar_images);
            } catch (error) {
                console.error("Error al buscar imágenes similares:", error);
                setSimilarImages([]);
            } finally {
                setIsLoading(false);
            }
        } else {
            alert("Por favor, seleccione un archivo antes de buscar.");
        }
    };

    const getLastTwoPathSegments = (path: string): string => {
        const segments = path.split('/');
        return `${segments[segments.length - 2]}/${segments[segments.length - 1]}`;
    };

    return (
        <div className="imagen-search-container">
            <img src={logo} alt="Logo" className="logo"/>
            <div className="search-box">
                <div className="file-input-wrapper">
                    <label htmlFor="file-upload" className="file-label">Elegir archivo</label>
                    <input
                        id="file-upload"
                        type="file"
                        accept="image/*"
                        onChange={handleFileChange}
                        className="file-input"
                    />
                    <span className="file-name">{fileName || "No se ha seleccionado ningún archivo"}</span>
                </div>
                <button onClick={handleSearch} className="search-button" disabled={isLoading}>
                    <svg className="search-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="11" cy="11" r="8"></circle>
                        <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                    </svg>
                    {isLoading ? 'Buscando...' : 'Buscar'}
                </button>
            </div>
            {imagePreview && (
                <div className="image-preview-container">
                    <img src={imagePreview} alt="Imagen seleccionada" className="image-preview" />
                </div>
            )}
            {similarImages.length > 0 && (
                <div className="similar-images-container">
                    <h2>Imágenes similares</h2>
                    <div className="similar-images-grid">
                        {similarImages.map((img, index) => (
                            <div key={index} className="similar-image-item">
                                <img src={`http://localhost:5000/data/101_ObjectCategories/${getLastTwoPathSegments(img.path)}`} alt={`Similar ${index + 1}`} />
                                <p>Similar {index + 1}</p>
                                <p>Similitud: {(img.similarity * 100).toFixed(2)}%</p>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};

export default ImagenSearch;
