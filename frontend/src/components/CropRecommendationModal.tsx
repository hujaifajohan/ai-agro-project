import React, { useState } from 'react';
import { apiService } from '../services/api';
import type { CropRecommendationInput, CropRecommendationResult } from '../types';

interface CropRecommendationModalProps {
  isOpen: boolean;
  onClose: () => void;
  selectedField?: any;
}

export const CropRecommendationModal: React.FC<CropRecommendationModalProps> = ({
  isOpen,
  onClose,
  selectedField
}) => {


  // Inputs state pre-populated with realistic default values or selected field values
  const [N, setN] = useState<number>(45);
  const [P, setP] = useState<number>(30);
  const [K, setK] = useState<number>(35);
  const [temperature, setTemperature] = useState<number>(selectedField?.temperature ?? 28);
  const [humidity, setHumidity] = useState<number>(selectedField?.humidity ?? 70);
  const [ph, setPh] = useState<number>(selectedField?.soilPH ?? 6.5);
  const [rainfall, setRainfall] = useState<number>(selectedField?.rainfall ?? 200);

  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<CropRecommendationResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  if (!isOpen) return null;

  const handlePopulateFromField = () => {
    if (selectedField) {
      if (selectedField.temperature !== undefined) setTemperature(selectedField.temperature);
      if (selectedField.humidity !== undefined) setHumidity(selectedField.humidity);
      if (selectedField.soilPH !== undefined) setPh(selectedField.soilPH);
      if (selectedField.rainfall !== undefined) setRainfall(selectedField.rainfall);
    }
  };

  const handleCalculate = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setErrorMessage(null);
    setResult(null);

    const inputData: CropRecommendationInput = {
      N: Number(N),
      P: Number(P),
      K: Number(K),
      temperature: Number(temperature),
      humidity: Number(humidity),
      ph: Number(ph),
      rainfall: Number(rainfall)
    };

    try {
      const res = await apiService.recommendCrop(inputData);
      if (res.success) {
        setResult(res);
      } else {
        setErrorMessage(res.error || 'Failed to generate crop recommendation.');
      }
    } catch {
      setErrorMessage('Unable to communicate with Decision Tree API.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-xs p-4 overflow-y-auto">
      <div className="bg-surface border border-outline-variant/40 rounded-3xl shadow-2xl max-w-3xl w-full p-6 sm:p-8 flex flex-col gap-6 max-h-[90vh] overflow-y-auto animate-in fade-in zoom-in-95 duration-200">
        
        {/* Modal Header */}
        <div className="flex items-start justify-between pb-4 border-b border-outline-variant/30">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-2xl bg-primary/10 text-primary flex items-center justify-center">
              <span className="material-symbols-outlined text-[28px]">park</span>
            </div>
            <div>
              <h2 className="text-xl font-bold font-display text-on-surface">Decision Tree Crop Recommendation</h2>
              <p className="text-xs text-on-surface-variant">Trained ML Model (2,200 Agricultural Samples, 98.86% Accuracy)</p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="w-9 h-9 rounded-full bg-surface-container hover:bg-surface-container-high flex items-center justify-center text-on-surface-variant transition-colors cursor-pointer"
          >
            <span className="material-symbols-outlined text-[20px]">close</span>
          </button>
        </div>

        {/* Input Form & Auto-fill Action */}
        <form onSubmit={handleCalculate} className="flex flex-col gap-5">
          {selectedField && (
            <div className="flex items-center justify-between p-3 bg-surface-container-lowest rounded-xl border border-outline-variant/30 text-xs">
              <span className="text-on-surface-variant font-medium">
                Active Field: <strong className="text-on-surface">{selectedField.name}</strong>
              </span>
              <button
                type="button"
                onClick={handlePopulateFromField}
                className="text-primary hover:underline font-semibold flex items-center gap-1 cursor-pointer"
              >
                <span className="material-symbols-outlined text-[16px]">sync</span>
                Auto-fill Field Measurements
              </button>
            </div>
          )}

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Nitrogen (N) */}
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-semibold text-on-surface-variant flex items-center justify-between">
                <span>Nitrogen (N)</span>
                <span className="text-[10px] text-outline">kg/ha</span>
              </label>
              <input
                type="number"
                step="any"
                required
                value={N}
                onChange={(e) => setN(Number(e.target.value))}
                className="px-3.5 py-2.5 rounded-xl bg-surface-container-lowest border border-outline-variant/40 text-on-surface font-semibold text-sm focus:outline-none focus:border-primary transition-colors"
                placeholder="45"
              />
            </div>

            {/* Phosphorus (P) */}
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-semibold text-on-surface-variant flex items-center justify-between">
                <span>Phosphorus (P)</span>
                <span className="text-[10px] text-outline">kg/ha</span>
              </label>
              <input
                type="number"
                step="any"
                required
                value={P}
                onChange={(e) => setP(Number(e.target.value))}
                className="px-3.5 py-2.5 rounded-xl bg-surface-container-lowest border border-outline-variant/40 text-on-surface font-semibold text-sm focus:outline-none focus:border-primary transition-colors"
                placeholder="30"
              />
            </div>

            {/* Potassium (K) */}
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-semibold text-on-surface-variant flex items-center justify-between">
                <span>Potassium (K)</span>
                <span className="text-[10px] text-outline">kg/ha</span>
              </label>
              <input
                type="number"
                step="any"
                required
                value={K}
                onChange={(e) => setK(Number(e.target.value))}
                className="px-3.5 py-2.5 rounded-xl bg-surface-container-lowest border border-outline-variant/40 text-on-surface font-semibold text-sm focus:outline-none focus:border-primary transition-colors"
                placeholder="35"
              />
            </div>

            {/* Temperature */}
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-semibold text-on-surface-variant flex items-center justify-between">
                <span>Temperature</span>
                <span className="text-[10px] text-outline">°C</span>
              </label>
              <input
                type="number"
                step="any"
                required
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
                className="px-3.5 py-2.5 rounded-xl bg-surface-container-lowest border border-outline-variant/40 text-on-surface font-semibold text-sm focus:outline-none focus:border-primary transition-colors"
                placeholder="28"
              />
            </div>

            {/* Humidity */}
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-semibold text-on-surface-variant flex items-center justify-between">
                <span>Humidity</span>
                <span className="text-[10px] text-outline">%</span>
              </label>
              <input
                type="number"
                step="any"
                required
                value={humidity}
                onChange={(e) => setHumidity(Number(e.target.value))}
                className="px-3.5 py-2.5 rounded-xl bg-surface-container-lowest border border-outline-variant/40 text-on-surface font-semibold text-sm focus:outline-none focus:border-primary transition-colors"
                placeholder="70"
              />
            </div>

            {/* pH */}
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-semibold text-on-surface-variant flex items-center justify-between">
                <span>Soil pH</span>
                <span className="text-[10px] text-outline">0 - 14</span>
              </label>
              <input
                type="number"
                step="any"
                required
                value={ph}
                onChange={(e) => setPh(Number(e.target.value))}
                className="px-3.5 py-2.5 rounded-xl bg-surface-container-lowest border border-outline-variant/40 text-on-surface font-semibold text-sm focus:outline-none focus:border-primary transition-colors"
                placeholder="6.5"
              />
            </div>

            {/* Rainfall */}
            <div className="flex flex-col gap-1.5 sm:col-span-2">
              <label className="text-xs font-semibold text-on-surface-variant flex items-center justify-between">
                <span>Annual Rainfall</span>
                <span className="text-[10px] text-outline">mm</span>
              </label>
              <input
                type="number"
                step="any"
                required
                value={rainfall}
                onChange={(e) => setRainfall(Number(e.target.value))}
                className="px-3.5 py-2.5 rounded-xl bg-surface-container-lowest border border-outline-variant/40 text-on-surface font-semibold text-sm focus:outline-none focus:border-primary transition-colors"
                placeholder="200"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-3 bg-primary hover:bg-primary/90 disabled:opacity-50 text-on-primary font-bold text-sm rounded-xl shadow-md transition-all flex items-center justify-center gap-2 cursor-pointer mt-2"
          >
            {loading ? (
              <>
                <span className="w-4 h-4 border-2 border-on-primary border-t-transparent rounded-full animate-spin"></span>
                <span>Analyzing field conditions...</span>
              </>
            ) : (
              <>
                <span className="material-symbols-outlined text-[20px]">psychology</span>
                <span>Get Crop Recommendation</span>
              </>
            )}
          </button>
        </form>

        {/* Error State Banner */}
        {errorMessage && (
          <div className="p-4 bg-rose-50 border border-rose-200 text-rose-800 rounded-xl text-xs font-semibold flex items-center gap-2">
            <span className="material-symbols-outlined text-[20px] text-rose-600">error</span>
            <span>{errorMessage}</span>
          </div>
        )}

        {/* Recommendation Result Card */}
        {result && result.success && (
          <div className="p-6 bg-emerald-50/70 border border-emerald-200/80 rounded-2xl flex flex-col gap-4 animate-in fade-in slide-in-from-bottom-2 duration-300">
            <div className="flex items-center justify-between flex-wrap gap-2 pb-3 border-b border-emerald-200/60">
              <div className="flex items-center gap-2">
                <span className="px-2.5 py-1 bg-emerald-700 text-white font-bold text-[10px] uppercase tracking-wider rounded-md">
                  AI Decision Tree
                </span>
                <span className="text-xs font-medium text-emerald-800">Model Accuracy: {(result.modelAccuracy ? result.modelAccuracy * 100 : 98.86).toFixed(2)}%</span>
              </div>
              <div className="px-3 py-1 bg-emerald-100 text-emerald-900 rounded-full font-bold text-xs flex items-center gap-1.5 border border-emerald-300">
                <span className="material-symbols-outlined text-[16px] text-emerald-700">verified</span>
                <span>Model Confidence: {result.confidence}%</span>
              </div>
            </div>

            <div className="flex items-center gap-4 py-2">
              <div className="text-4xl sm:text-5xl font-black text-emerald-950 font-display">
                {result.recommendedCrop}
              </div>
            </div>

            <div className="text-xs text-emerald-900/80 italic bg-white/60 p-2.5 rounded-xl border border-emerald-200/40">
              ℹ️ Model Confidence score reflects Decision Tree probability distribution derived from trained scikit-learn model.
            </div>

            {/* Feature Breakdown Table */}
            {result.features && (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs pt-2">
                <div className="bg-white/80 p-2 rounded-lg border border-emerald-200/40">
                  <span className="text-[10px] uppercase font-bold text-emerald-700">Nitrogen (N)</span>
                  <p className="font-bold text-emerald-950">{result.features.N} kg/ha</p>
                </div>
                <div className="bg-white/80 p-2 rounded-lg border border-emerald-200/40">
                  <span className="text-[10px] uppercase font-bold text-emerald-700">Phosphorus (P)</span>
                  <p className="font-bold text-emerald-950">{result.features.P} kg/ha</p>
                </div>
                <div className="bg-white/80 p-2 rounded-lg border border-emerald-200/40">
                  <span className="text-[10px] uppercase font-bold text-emerald-700">Potassium (K)</span>
                  <p className="font-bold text-emerald-950">{result.features.K} kg/ha</p>
                </div>
                <div className="bg-white/80 p-2 rounded-lg border border-emerald-200/40">
                  <span className="text-[10px] uppercase font-bold text-emerald-700">Temperature</span>
                  <p className="font-bold text-emerald-950">{result.features.temperature}°C</p>
                </div>
                <div className="bg-white/80 p-2 rounded-lg border border-emerald-200/40">
                  <span className="text-[10px] uppercase font-bold text-emerald-700">Humidity</span>
                  <p className="font-bold text-emerald-950">{result.features.humidity}%</p>
                </div>
                <div className="bg-white/80 p-2 rounded-lg border border-emerald-200/40">
                  <span className="text-[10px] uppercase font-bold text-emerald-700">Soil pH</span>
                  <p className="font-bold text-emerald-950">{result.features.ph}</p>
                </div>
                <div className="bg-white/80 p-2 rounded-lg border border-emerald-200/40 sm:col-span-2">
                  <span className="text-[10px] uppercase font-bold text-emerald-700">Rainfall</span>
                  <p className="font-bold text-emerald-950">{result.features.rainfall} mm</p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
