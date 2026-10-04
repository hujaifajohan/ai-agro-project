export type FieldStatus = 'Healthy' | 'Moderate' | 'Dry' | 'Critical';

export interface Field {
  id: string;
  name: string;
  crop: string;
  soilMoisture?: number; // in %
  soilPH?: number;
  temperature?: number; // in °C
  humidity?: number; // in %
  rainfall?: number; // in mm
  status: FieldStatus;
  waterRequirement?: string; // e.g. "Low", "Moderate", "High"
  assignedFarmerId?: string | null;
  assignedFarmerName?: string | null;
  farmerId?: string | null;
  areaAcres?: number;
  soilType?: string;
  latitude?: number;
  longitude?: number;
  farmId?: string;
  ownerId?: string;
  fieldId?: string;
  docId?: string;
  boundary?: any;
  path?: any;
  createdAt?: string;
  updatedAt?: string;
}

export type ModulePhaseStatus = 'Foundation Ready' | 'Coming in Week 2' | 'Research Ready';

export interface AIModuleInfo {
  id: string;
  name: string;
  category: 'Machine Learning' | 'Deep Learning' | 'Constraint Satisfaction' | 'Search Algorithm' | 'Optimization';
  status: ModulePhaseStatus;
  week: 1 | 2;
  description: string;
  route: string;
}

export interface SystemMetrics {
  totalFields: number;
  healthyFields: number;
  dryFields: number;
  criticalFields: number;
  waterAvailability: number; // in Liters or %
}

export interface CropRecommendationInput {
  N: number;
  P: number;
  K: number;
  temperature: number;
  humidity: number;
  ph: number;
  rainfall: number;
}

export interface CropRecommendationResult {
  success: boolean;
  recommendedCrop?: string;
  rawCrop?: string;
  confidence?: number;
  model?: string;
  modelAccuracy?: number;
  features?: CropRecommendationInput;
  featureImportances?: Record<string, number>;
  error?: string;
}

export interface FieldDecisionInput {
  crop?: string;
  N?: number;
  P?: number;
  K?: number;
  temperature?: number;
  humidity?: number;
  ph?: number;
  rainfall?: number;
  soil_moisture?: number;
  soilMoisture?: number;
  soilPH?: number;
}

export interface FieldDecisionV2Result {
  success: boolean;
  status: 'Healthy' | 'Attention' | 'Critical' | string;
  water_need: 'Low' | 'Moderate' | 'High' | 'Urgent' | string;
  action: 'Pathogen AI' | 'Queue Run' | 'Inspect' | string;
  action_route: string;
  status_confidence: number;
  water_need_confidence: number;
  status_probabilities?: Record<string, number>;
  water_need_probabilities?: Record<string, number>;
  is_low_confidence?: boolean;
  reasons?: string[];
  model_version?: string;
  soil_moisture_provenance?: string;
  error?: string;
}

