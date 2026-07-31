import {StrictMode} from 'react';
import {createRoot} from 'react-dom/client';
import LandingApp from './LandingApp.tsx';
import './index.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <LandingApp />
  </StrictMode>,
);