import React, { useState } from 'react';
import { apiService } from '../services/api';
import type { CropRecommendationInput, CropRecommendationResult } from '../types';

interface CropRecommendationCardProps {
  selectedField?: any;
  onOpenModal?: () => void;
}

export const CropRecommendationCard: React.FC<CropRecommendationCardProps> = ({
  selectedField,
  onOpenModal
}) => {
  const [N, setN] = useState<number>(45);
  const [P, setP] = useState<number>(30);
  const [K, setK] = useState<number>(35);
  const [temperature, setTemperature] = useState<number>(selectedField?.temperature ?? 28);
  const [humidity, setHumidity] = useState<number>(selectedField?.humidity ?? 70);
  const [ph, setPh] = useState<number>(selectedField?.soilPH ?? 6.5);
  const [rainfall, setRainfall] = useState<number>(selectedField?.rainfall ?? 200);

  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<CropRecommendationResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const inputData: CropRecommendationInput = {
        N: Number(N),
        P: Number(P),
        K: Number(K),
        temperature: Number(temperature),
        humidity: Number(humidity),
        ph: Number(ph),
        rainfall: Number(rainfall)
      };
      const res = await apiService.recommendCrop(inputData);
      if (res.success) {
        setResult(res);
      } else {
        setError(res.error || 'Failed to generate crop recommendation.');
      }
    } catch {
      setError('Unable to connect to crop recommendation model.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-surface-container-lowest border border-outline-variant/40 rounded-2xl p-5 shadow-xs flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-primary/10 text-primary flex items-center justify-center font-bold">
            <span className="material-symbols-outlined text-[24px]">psychology</span>
          </div>
          <div>
            <h3 className="font-bold text-base text-on-surface">Decision Tree Crop Recommendation</h3>
            <p className="text-xs text-on-surface-variant">Real-time ML Inference Model (98.86% Accuracy)</p>
          </div>
        </div>
        {onOpenModal && (
          <button
            type="button"
            onClick={onOpenModal}
            className="px-3 py-1.5 bg-secondary/10 text-secondary hover:bg-secondary/20 rounded-xl font-bold text-xs flex items-center gap-1 cursor-pointer transition-colors"
          >
            <span className="material-symbols-outlined text-[16px]">open_in_full</span>
            Full Input Workspace
          </button>
        )}
      </div>

      {/* Quick Input Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2 text-xs">
        <div className="flex flex-col gap-1 bg-surface-container-low p-2 rounded-xl border border-outline-variant/30">
          <span className="font-semibold text-outline text-[10px]">N (Nitrogen)</span>
          <input
            type="number"
            value={N}
            onChange={(e) => setN(Number(e.target.value))}
            className="bg-transparent font-bold text-on-surface focus:outline-none"
          />
        </div>
        <div className="flex flex-col gap-1 bg-surface-container-low p-2 rounded-xl border border-outline-variant/30">
          <span className="font-semibold text-outline text-[10px]">P (Phosphorus)</span>
          <input
            type="number"
            value={P}
            onChange={(e) => setP(Number(e.target.value))}
            className="bg-transparent font-bold text-on-surface focus:outline-none"
          />
        </div>
        <div className="flex flex-col gap-1 bg-surface-container-low p-2 rounded-xl border border-outline-variant/30">
          <span className="font-semibold text-outline text-[10px]">K (Potassium)</span>
          <input
            type="number"
            value={K}
            onChange={(e) => setK(Number(e.target.value))}
            className="bg-transparent font-bold text-on-surface focus:outline-none"
          />
        </div>
        <div className="flex flex-col gap-1 bg-surface-container-low p-2 rounded-xl border border-outline-variant/30">
          <span className="font-semibold text-outline text-[10px]">Temp (°C)</span>
          <input
            type="number"
            value={temperature}
            onChange={(e) => setTemperature(Number(e.target.value))}
            className="bg-transparent font-bold text-on-surface focus:outline-none"
          />
        </div>
        <div className="flex flex-col gap-1 bg-surface-container-low p-2 rounded-xl border border-outline-variant/30">
          <span className="font-semibold text-outline text-[10px]">Humidity (%)</span>
          <input
            type="number"
            value={humidity}
            onChange={(e) => setHumidity(Number(e.target.value))}
            className="bg-transparent font-bold text-on-surface focus:outline-none"
          />
        </div>
        <div className="flex flex-col gap-1 bg-surface-container-low p-2 rounded-xl border border-outline-variant/30">
          <span className="font-semibold text-outline text-[10px]">Soil pH</span>
          <input
            type="number"
            step="0.1"
            value={ph}
            onChange={(e) => setPh(Number(e.target.value))}
            className="bg-transparent font-bold text-on-surface focus:outline-none"
          />
        </div>
        <div className="flex flex-col gap-1 bg-surface-container-low p-2 rounded-xl border border-outline-variant/30 col-span-2 sm:col-span-1">
          <span className="font-semibold text-outline text-[10px]">Rainfall (mm)</span>
          <input
            type="number"
            value={rainfall}
            onChange={(e) => setRainfall(Number(e.target.value))}
            className="bg-transparent font-bold text-on-surface focus:outline-none"
          />
        </div>
      </div>

      <div className="flex items-center justify-between gap-3 pt-1">
        <button
          type="button"
          onClick={() => handlePredict()}
          disabled={loading}
          className="px-5 py-2.5 bg-primary hover:bg-primary/90 text-on-primary font-bold text-xs rounded-xl shadow-xs transition-all flex items-center gap-2 cursor-pointer disabled:opacity-50"
        >
          {loading ? (
            <>
              <span className="w-3.5 h-3.5 border-2 border-on-primary border-t-transparent rounded-full animate-spin"></span>
              <span>Analyzing field conditions...</span>
            </>
          ) : (
            <>
              <span className="material-symbols-outlined text-[18px]">psychology</span>
              <span>Get Crop Recommendation</span>
            </>
          )}
        </button>
      </div>

      {error && (
        <div className="p-3 bg-rose-50 text-rose-800 rounded-xl text-xs font-medium border border-rose-200">
          {error}
        </div>
      )}

      {/* Result Display */}
      {result && result.success && (
        <div className="p-4 bg-emerald-500/10 border border-emerald-500/30 rounded-xl flex flex-col gap-2 animate-in fade-in">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-bold uppercase tracking-wider text-emerald-800 dark:text-emerald-300">
              Recommended Crop
            </span>
            <span className="px-2.5 py-0.5 rounded-full bg-emerald-600 text-white text-[11px] font-bold">
              Confidence: {result.confidence}%
            </span>
          </div>
          <div className="text-2xl font-extrabold text-emerald-950 dark:text-emerald-100 flex items-center gap-2">
            {result.recommendedCrop}
          </div>
          <p className="text-[11px] text-emerald-800/80 dark:text-emerald-300/80">
            Based on: N: {result.features?.N}, P: {result.features?.P}, K: {result.features?.K}, Temp: {result.features?.temperature}°C, Humidity: {result.features?.humidity}%, pH: {result.features?.ph}, Rainfall: {result.features?.rainfall}mm
          </p>
        </div>
      )}
    </div>
  );
};
