import React from 'react';

interface Result {
  label: string;
  index: number;
  distance: number;
}

interface Props {
  results: Result[];
}

const ImageGallery: React.FC<Props> = ({ results }) => (
  <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center' }}>
    {results.map((result, index) => (
      <div key={index} style={{ margin: '10px', textAlign: 'center' }}>
        <img
          src={`http://localhost:5000/images/${result.label}/${result.index}.jpg`}
          alt={`Similar image ${index + 1}`}
          style={{ width: '150px', height: '150px', objectFit: 'cover' }}
        />
        <p>Rank: {index + 1}</p>
        <p>Distance: {result.distance.toFixed(4)}</p>
      </div>
    ))}
  </div>
);

export default ImageGallery;
