import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import SelectionPage from './pages/SelectionPage';
import DashboardPage from './pages/DashboardPage';

import EvaluationPage from './pages/EvaluationPage';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<SelectionPage />} />
        <Route path="/dashboard/:symbol" element={<DashboardPage />} />
        <Route path="/evaluation" element={<EvaluationPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
