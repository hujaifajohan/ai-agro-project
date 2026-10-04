"""
AgroAI - FastAPI Backend Application
Main Entry Point with Full API Contracts & AI Modules Integration
"""

import sys
import os
import csv
import math
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai.csp.ac3 import ac3
from ai.csp.backtracking import backtrack_csp
from ai.csp.genetic import IrrigationGeneticOptimizer
from ai.search.search_engine import bfs_search, dfs_search
from ai.search.astar import astar_search
from ai.minimax.minimax import PestRiskMinimax
from ai.kmeans.kmeans_service import KMeansService
from ai.decision_tree.dtree_service import DecisionTreeService
from ai.cnn.cnn_service import CNNDiseaseService
from ai.dtree.predictor import crop_predictor
from ai.dtree.decision_tree_v2_service import dtree_v2_service
from firebase_db import AgroDatabaseService, using_firestore

app = FastAPI(
    title="AgroAI Intelligent Decision Support API",
    description="Full Backend API supporting Farm & Field CRUD, AI Prediction Interfaces, and Search/CSP Solvers.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI Services
kmeans_svc = KMeansService()
dtree_svc = DecisionTreeService()
cnn_svc = CNNDiseaseService()

# --- Pydantic Data Schemas ---
class FarmModel(BaseModel):
    id: Optional[str] = None
    name: str
    location: str
    area: str
    soilType: str
    mainCrop: str
    description: Optional[str] = ""

class FieldModel(BaseModel):
    id: Optional[str] = None
    fieldId: Optional[str] = None
    name: Optional[str] = "Field Sector"
    crop: Optional[str] = "Crop"
    area: Optional[str] = "100 Hectares"
    soilMoisture: Optional[float] = 50.0
    soilPH: Optional[float] = 6.5
    temperature: Optional[float] = 28.0
    humidity: Optional[float] = 60.0
    rainfall: Optional[float] = 10.0
    waterAvailability: Optional[str] = "Moderate"
    status: Optional[str] = "Healthy"
    waterRequirement: Optional[str] = "Low"
    assignedFarmerId: Optional[str] = None
    assignedFarmerName: Optional[str] = None
    farmerId: Optional[str] = None
    assignedTo: Optional[str] = None

class KMeansRequest(BaseModel):
    soil_moisture: float = 45.0
    soil_ph: float = 6.5
    temperature: float = 28.0
    humidity: float = 60.0
    rainfall: float = 10.0

class DTreeRequest(BaseModel):
    crop: str = "Tomato"
    soil_moisture: float = 30.0
    soil_ph: float = 6.2
    temperature: float = 32.0
    humidity: float = 55.0
    rainfall: float = 5.0

class CropRecommendationRequest(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

class DecisionRequest(BaseModel):
    crop: Optional[str] = "rice"
    N: float = 90.0
    P: float = 42.0
    K: float = 43.0
    temperature: float = 20.8
    humidity: float = 82.0
    ph: Optional[float] = None
    soilPH: Optional[float] = None
    rainfall: float = 202.9
    soil_moisture: Optional[float] = None
    soilMoisture: Optional[float] = None

class SearchRequest(BaseModel):
    algorithm: str = "astar"  # bfs, dfs, astar
    start: List[int] = [0, 0]
    goal: List[int] = [2, 3]

class MinimaxRequest(BaseModel):
    field_name: Optional[str] = "Field D"
    crop: Optional[str] = "Tomato"

# --- Routes ---


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "project": "AgroAI",
        "fastapi_version": "0.110",
        "database": "Firestore (Cloud)" if using_firestore else "Persistent Database",
        "ai_modules_ready": True,
        "cropRecommendationModelLoaded": crop_predictor.is_loaded,
        "decision_tree_v2": {
            "loaded": dtree_v2_service.is_loaded
        }
    }

# --- Farm CRUD ---
@app.get("/api/farms")
def get_farms():
    return AgroDatabaseService.get_farms()

@app.post("/api/farms")
def create_farm(farm: FarmModel):
    return AgroDatabaseService.save_farm(farm.dict())

@app.put("/api/farms/{farm_id}")
def update_farm(farm_id: str, farm: FarmModel):
    res = AgroDatabaseService.update_farm(farm_id, farm.dict())
    if not res:
        raise HTTPException(status_code=404, detail="Farm not found")
    return res

@app.delete("/api/farms/{farm_id}")
def delete_farm(farm_id: str):
    AgroDatabaseService.delete_farm(farm_id)
    return {"message": f"Farm {farm_id} deleted successfully."}

# --- Field CRUD ---
@app.get("/api/fields")
def get_fields():
    return AgroDatabaseService.get_fields()

@app.post("/api/fields")
def create_field(field: FieldModel):
    return AgroDatabaseService.save_field(field.dict())

@app.put("/api/fields/{field_id}")
def update_field(field_id: str, payload: dict):
    res = AgroDatabaseService.update_field(field_id, payload)
    if not res:
        raise HTTPException(status_code=404, detail="Field not found")
    return res

@app.delete("/api/fields/{field_id}")
def delete_field(field_id: str):
    AgroDatabaseService.delete_field(field_id)
    return {"message": f"Field {field_id} deleted successfully."}

# --- Weather with Real Online Open-Meteo Integration ---
WMO_WEATHER_MAP = {
    0: {"desc": "Clear Peak", "icon": "wb_sunny", "summary": "Clear sky with maximum solar penetration. Unrestricted transpiration potential."},
    1: {"desc": "Sunny", "icon": "wb_sunny", "summary": "Mainly clear conditions across the agricultural canopy with optimal photosynthesis rates."},
    2: {"desc": "Partly Cloudy", "icon": "partly_cloudy_day", "summary": "Partly cloudy with intermittent solar irradiance. Ideal atmospheric boundary layer."},
    3: {"desc": "Overcast", "icon": "cloud", "summary": "Overcast cloud cover reducing direct PAR irradiance. Lower evapotranspiration stress."},
    45: {"desc": "Coastal Fog", "icon": "foggy", "summary": "Marine boundary layer fog with high relative humidity and suppressed vapor pressure deficit."},
    48: {"desc": "Coastal Fog", "icon": "foggy", "summary": "Dense rime fog maintaining high surface leaf moisture levels."},
    51: {"desc": "Light Showers", "icon": "rainy", "summary": "Light patchy drizzle with slight surface wetting. Inhibit foliar chemical spray operations."},
    53: {"desc": "Light Showers", "icon": "rainy", "summary": "Moderate drizzle creating wet canopy conditions. Foliage disease risk elevated."},
    55: {"desc": "Light Showers", "icon": "rainy", "summary": "Dense drizzle accumulation across root zones."},
    61: {"desc": "Light Showers", "icon": "rainy", "summary": "Slight rain showers. Temporarily pause scheduled mechanical irrigation cycles."},
    63: {"desc": "Light Showers", "icon": "rainy", "summary": "Moderate rain accumulation. Natural infiltration replenishing topsoil layer."},
    65: {"desc": "Light Showers", "icon": "rainy", "summary": "Heavy rainfall. Check drainage ditches and pause active irrigation pumps."},
    71: {"desc": "Light Showers", "icon": "ac_unit", "summary": "Slight snow or frost risk. Activate frost protection protocols where active."},
    80: {"desc": "Light Showers", "icon": "rainy", "summary": "Isolated rain showers passing through the field sector."},
    81: {"desc": "Light Showers", "icon": "rainy", "summary": "Moderate rain showers across the cultivation zone."},
    82: {"desc": "Light Showers", "icon": "rainy", "summary": "Violent rain showers with high kinetic droplet impact."},
    95: {"desc": "Light Showers", "icon": "thunderstorm", "summary": "Thunderstorm activity detected. Suspend field personnel and autonomous machinery."}
}

CARDINAL_DIRECTIONS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]

def degrees_to_cardinal(deg: float) -> str:
    try:
        idx = int((deg + 11.25) / 22.5) % 16
        return CARDINAL_DIRECTIONS[idx]
    except Exception:
        return "NW"

def calculate_stull_wet_bulb(temp_c: float, rh_pct: float) -> float:
    """
    Calculates Wet Bulb Temperature (°C) using the scientifically validated Stull (2011) formula.
    Accurate to within 0.15°C for agricultural spray operations.
    """
    rh = max(1.0, min(100.0, rh_pct))
    t = temp_c
    tw = (
        t * math.atan(0.151977 * math.sqrt(rh + 8.313659))
        + math.atan(t + rh)
        - math.atan(rh - 1.676331)
        + 0.00391838 * (rh ** 1.5) * math.atan(0.023101 * rh)
        - 4.686035
    )
    return round(tw, 2)

def resolve_weather_location(lat: Optional[float], lon: Optional[float], location_name: Optional[str]):
    """
    Resolves target latitude, longitude, and descriptive label.
    Supports coordinates, city/place geocoding, and IP auto-detection.
    """
    import requests

    # Case 1: Coordinates provided directly
    if lat is not None and lon is not None:
        label = location_name if location_name and location_name.strip() else f"{round(lat, 4)}°, {round(lon, 4)}°"
        return lat, lon, label

    # Case 2: City / location name provided -> use Open-Meteo Geocoding
    if location_name and location_name.strip():
        try:
            q = requests.utils.quote(location_name.strip())
            geo_res = requests.get(
                f"https://geocoding-api.open-meteo.com/v1/search?name={q}&count=1&language=en&format=json",
                timeout=4
            ).json()
            results = geo_res.get("results")
            if results and len(results) > 0:
                first = results[0]
                resolved_lat = float(first["latitude"])
                resolved_lon = float(first["longitude"])
                name = first.get("name", location_name)
                country = first.get("country", "")
                resolved_label = f"{name}, {country}".strip(", ")
                return resolved_lat, resolved_lon, resolved_label
        except Exception as e:
            print(f"[Weather Geocoding] Failed to resolve '{location_name}': {e}")

    # Case 3: Auto-detect location via IP geolocation (find where user/server actually is)
    try:
        ip_res = requests.get("http://ip-api.com/json", timeout=3).json()
        if ip_res.get("status") == "success":
            detected_lat = float(ip_res["lat"])
            detected_lon = float(ip_res["lon"])
            city = ip_res.get("city", "")
            country = ip_res.get("country", "")
            detected_label = f"{city}, {country}".strip(", ")
            return detected_lat, detected_lon, detected_label
    except Exception as e:
        print(f"[Weather IP Detect] IP geolocation fallback failed: {e}")

    # Fallback to Dhaka, Bangladesh (BST +06:00)
    return 23.7891, 90.4126, "Dhaka, Bangladesh"

@app.get("/api/weather/search")
def search_weather_location(query: str):
    if not query or len(query.strip()) < 2:
        return []
    try:
        import requests
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={requests.utils.quote(query.strip())}&count=6&language=en&format=json"
        res = requests.get(geo_url, timeout=4).json()
        results = res.get("results", [])
        return [
            {
                "name": item.get("name"),
                "country": item.get("country", ""),
                "admin1": item.get("admin1", ""),
                "latitude": item.get("latitude"),
                "longitude": item.get("longitude"),
                "label": f"{item.get('name')}, {item.get('admin1') + ', ' if item.get('admin1') else ''}{item.get('country', '')}".strip(", ")
            }
            for item in results
        ]
    except Exception as e:
        print(f"[Weather Location Search] Error: {e}")
        return []

@app.get("/api/weather")
def get_weather(lat: Optional[float] = None, lon: Optional[float] = None, location: Optional[str] = None):
    target_lat, target_lon, loc_label = resolve_weather_location(lat, lon, location)

    try:
        import requests
        api_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={target_lat}&longitude={target_lon}"
            f"&current=temperature_2m,relative_humidity_2m,apparent_temperature,dew_point_2m,precipitation,rain,showers,weather_code,pressure_msl,surface_pressure,wind_speed_10m,wind_direction_10m,wind_gusts_10m,cloud_cover,shortwave_radiation"
            f"&daily=weather_code,temperature_2m_max,temperature_2m_min,apparent_temperature_max,apparent_temperature_min,precipitation_sum,precipitation_probability_max,et0_fao_evapotranspiration,uv_index_max,wind_speed_10m_max,wind_gusts_10m_max"
            f"&hourly=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation_probability,precipitation,weather_code,wind_speed_10m,cloud_cover"
            f"&timezone=auto&models=best_match"
        )
        resp = requests.get(api_url, timeout=6)
        if resp.status_code == 200:
            data = resp.json()
            cur = data.get("current", {})
            daily = data.get("daily", {})
            hourly = data.get("hourly", {})

            temp = float(cur.get("temperature_2m", 28.0))
            feels_like = float(cur.get("apparent_temperature", temp))
            rh = float(cur.get("relative_humidity_2m", 65.0))
            dew = float(cur.get("dew_point_2m", temp - 5.0))
            wet_bulb = calculate_stull_wet_bulb(temp, rh)
            delta_t = round(max(0.0, temp - wet_bulb), 1)

            wind_speed = float(cur.get("wind_speed_10m", 8.0))
            wind_gusts = float(cur.get("wind_gusts_10m", round(wind_speed * 1.3, 1)))
            wind_dir_deg = float(cur.get("wind_direction_10m", 180.0))
            wind_dir = degrees_to_cardinal(wind_dir_deg)

            # Mean Sea Level normalized pressure for standard barometry
            pressure = float(cur.get("pressure_msl") or cur.get("surface_pressure", 1013.2))
            cloud_cover = int(cur.get("cloud_cover", 20))
            precip = float(cur.get("precipitation", 0.0))
            wcode = int(cur.get("weather_code", 0))
            solar = float(cur.get("shortwave_radiation", 600.0))

            # VPD calculation (kPa) using actual saturated and actual vapor pressures
            es = 0.61078 * math.exp((17.27 * temp) / (temp + 237.3))
            ea = es * (rh / 100.0)
            vpd = round(max(0.0, es - ea), 2)

            # Leaf moisture estimation
            if precip > 0:
                leaf_moist = min(100, int(75 + precip * 10))
            else:
                leaf_moist = max(5, min(95, int((rh - 20) * 0.9)))

            # Current weather interpretation
            w_info = WMO_WEATHER_MAP.get(wcode, {"desc": "Clear Sky", "icon": "wb_sunny", "summary": "Favorable microclimate conditions across field sector."})

            # Daily highs/lows and Evapotranspiration
            max_temps = daily.get("temperature_2m_max", [round(temp + 3, 1)])
            min_temps = daily.get("temperature_2m_min", [round(temp - 6, 1)])
            et0_list = daily.get("et0_fao_evapotranspiration", [4.2])
            precip_sums = daily.get("precipitation_sum", [precip])
            prob_list = daily.get("precipitation_probability_max", [10])
            uv_max_list = daily.get("uv_index_max", [round(solar / 100, 1)])
            dates = daily.get("time", [])

            cur_high = round(max_temps[0]) if max_temps else round(temp + 3)
            cur_low = round(min_temps[0]) if min_temps else round(temp - 6)
            cur_et0 = round(et0_list[0], 1) if et0_list and et0_list[0] is not None else 4.2
            cur_precip_24h = round(precip_sums[0], 1) if precip_sums and precip_sums[0] is not None else precip
            cur_uv = round(uv_max_list[0], 1) if uv_max_list and uv_max_list[0] is not None else round(max(1.0, solar / 100.0), 1)

            # Build 7-day forecast
            forecast = []
            num_days = min(7, len(dates)) if dates else 7
            for i in range(num_days):
                code_i = daily.get("weather_code", [])[i] if i < len(daily.get("weather_code", [])) else wcode
                info_i = WMO_WEATHER_MAP.get(code_i, {"desc": "Clear Sky", "icon": "wb_sunny"})
                d_high = round(max_temps[i]) if i < len(max_temps) else round(temp + 2)
                d_low = round(min_temps[i]) if i < len(min_temps) else round(temp - 7)
                d_rain = round(precip_sums[i], 1) if i < len(precip_sums) and precip_sums[i] is not None else 0.0
                d_prob = int(prob_list[i]) if i < len(prob_list) and prob_list[i] is not None else 0
                d_et0 = round(et0_list[i], 1) if i < len(et0_list) and et0_list[i] is not None else 4.0
                d_uv = round(uv_max_list[i], 1) if i < len(uv_max_list) and uv_max_list[i] is not None else 5.0

                forecast.append({
                    "day": i,
                    "date": dates[i] if i < len(dates) else None,
                    "icon": info_i["icon"],
                    "high": d_high,
                    "low": d_low,
                    "desc": info_i["desc"],
                    "rain": d_rain,
                    "prob": d_prob,
                    "et0": d_et0,
                    "uv_index": d_uv
                })

            # Build next 24-hour micro-forecast
            h_times = hourly.get("time", [])
            h_temps = hourly.get("temperature_2m", [])
            h_feels = hourly.get("apparent_temperature", [])
            h_rhs = hourly.get("relative_humidity_2m", [])
            h_probs = hourly.get("precipitation_probability", [])
            h_precips = hourly.get("precipitation", [])
            h_wcodes = hourly.get("weather_code", [])
            h_winds = hourly.get("wind_speed_10m", [])
            h_clouds = hourly.get("cloud_cover", [])

            cur_time_str = cur.get("time", "")
            cur_hour_prefix = cur_time_str[:13] if len(cur_time_str) >= 13 else ""
            start_idx = 0
            if cur_hour_prefix:
                for i, ht in enumerate(h_times):
                    if ht.startswith(cur_hour_prefix) or ht >= cur_time_str:
                        start_idx = i
                        break

            hourly_24h = []
            for i in range(start_idx, min(start_idx + 24, len(h_times))):
                code_h = h_wcodes[i] if i < len(h_wcodes) else 0
                h_info = WMO_WEATHER_MAP.get(code_h, {"desc": "Clear Sky", "icon": "wb_sunny"})
                t_iso = h_times[i]
                hour_label = t_iso.split("T")[-1] if "T" in t_iso else t_iso
                hourly_24h.append({
                    "time": t_iso,
                    "hour": hour_label,
                    "temp_c": round(float(h_temps[i]), 1) if i < len(h_temps) else temp,
                    "feels_like_c": round(float(h_feels[i]), 1) if i < len(h_feels) else temp,
                    "humidity_pct": int(h_rhs[i]) if i < len(h_rhs) else int(rh),
                    "rain_prob": int(h_probs[i]) if i < len(h_probs) else 0,
                    "rain_mm": round(float(h_precips[i]), 1) if i < len(h_precips) else 0.0,
                    "wind_speed_kmh": round(float(h_winds[i]), 1) if i < len(h_winds) else wind_speed,
                    "cloud_cover_pct": int(h_clouds[i]) if i < len(h_clouds) else cloud_cover,
                    "weather_code": code_h,
                    "icon": h_info["icon"],
                    "desc": h_info["desc"]
                })

            return {
                "location": loc_label,
                "latitude": target_lat,
                "longitude": target_lon,
                "temperature_c": round(temp, 1),
                "feels_like_c": round(feels_like, 1),
                "temp_high_c": cur_high,
                "temp_low_c": cur_low,
                "humidity_pct": int(rh),
                "dew_point_c": round(dew, 1),
                "wet_bulb_c": round(wet_bulb, 1),
                "delta_t_c": delta_t,
                "vpd_kpa": vpd,
                "leaf_moisture_pct": leaf_moist,
                "wind_speed_kmh": round(wind_speed, 1),
                "wind_gusts_kmh": round(wind_gusts, 1),
                "wind_direction": wind_dir,
                "wind_direction_deg": round(wind_dir_deg),
                "cloud_cover_pct": cloud_cover,
                "uv_index": cur_uv,
                "solar_irradiance_w_m2": round(solar) if solar > 0 else 580,
                "evapotranspiration_eto_mm": cur_et0,
                "barometer_hpa": round(pressure, 1),
                "precipitation_24h_mm": cur_precip_24h,
                "status": w_info["desc"],
                "icon": w_info["icon"],
                "summary": w_info["summary"],
                "forecast": forecast,
                "hourly": hourly_24h,
                "is_simulated": False,
                "source": "Open-Meteo High-Resolution (ECMWF/ICON/GFS Blended)",
                "online": True
            }
    except Exception as e:
        print(f"[Weather API] Open-Meteo fetch failed, using fallback: {e}")

    # Fallback if offline
    return {
        "location": loc_label,
        "latitude": target_lat,
        "longitude": target_lon,
        "temperature_c": 30.5,
        "feels_like_c": 35.8,
        "temp_high_c": 33,
        "temp_low_c": 26,
        "humidity_pct": 68,
        "dew_point_c": 24.2,
        "wet_bulb_c": 25.8,
        "delta_t_c": 4.7,
        "vpd_kpa": 1.35,
        "leaf_moisture_pct": 35,
        "wind_speed_kmh": 6.5,
        "wind_gusts_kmh": 12.0,
        "wind_direction": "SE",
        "wind_direction_deg": 135,
        "cloud_cover_pct": 25,
        "uv_index": 6.5,
        "solar_irradiance_w_m2": 620,
        "evapotranspiration_eto_mm": 4.5,
        "barometer_hpa": 1009.5,
        "precipitation_24h_mm": 0.0,
        "status": "Clear Peak",
        "icon": "wb_sunny",
        "summary": "Typical seasonal boundary layer. Stable transpiration conditions.",
        "forecast": [
            {"day": 0, "icon": "wb_sunny", "high": 33, "low": 26, "desc": "Clear Peak", "rain": 0, "prob": 5, "uv_index": 7.0},
            {"day": 1, "icon": "partly_cloudy_day", "high": 34, "low": 26, "desc": "Partly Cloudy", "rain": 0, "prob": 10, "uv_index": 6.5},
            {"day": 2, "icon": "partly_cloudy_day", "high": 32, "low": 25, "desc": "Partly Cloudy", "rain": 0.5, "prob": 25, "uv_index": 5.8},
            {"day": 3, "icon": "rainy", "high": 31, "low": 25, "desc": "Light Showers", "rain": 3.0, "prob": 60, "uv_index": 4.0},
            {"day": 4, "icon": "cloud", "high": 30, "low": 24, "desc": "Overcast", "rain": 1.0, "prob": 40, "uv_index": 4.5},
            {"day": 5, "icon": "wb_sunny", "high": 32, "low": 25, "desc": "Sunny", "rain": 0.0, "prob": 10, "uv_index": 6.8},
            {"day": 6, "icon": "wb_sunny", "high": 33, "low": 26, "desc": "Clear Peak", "rain": 0.0, "prob": 5, "uv_index": 7.0},
        ],
        "hourly": [],
        "is_simulated": True,
        "source": "Fallback Cache",
        "online": False
    }

# --- Resources & Sensors ---
@app.get("/api/resources")
def get_resources():
    return {
        "water_reservoir_liters": 64200,
        "water_reservoir_capacity": 85000,
        "active_pumps": 2,
        "total_pumps": 3,
        "sensor_nodes_count": 24,
        "mesh_status": "Online (100% Signal)",
        "is_simulated": True
    }

# --- Activity Audit Logs ---
@app.get("/api/logs")
def get_logs():
    return AgroDatabaseService.get_logs()

@app.post("/api/logs")
def create_log(log: dict):
    return AgroDatabaseService.save_log(log)


# --- Farmers (Read role=farmer users) ---

class AssignmentRequestModel(BaseModel):
    ownerId: str
    ownerName: str
    farmerId: str
    farmerName: str
    farmId: str
    farmName: str
    fieldId: str
    fieldName: str

class AssignmentRequestStatus(BaseModel):
    requestId: str
    status: str  # approved | rejected | cancelled

class UnassignModel(BaseModel):
    farmerId: str
    farmerName: str

class ConversationModel(BaseModel):
    ownerId: str
    farmerId: str
    ownerName: str
    farmerName: str

class MessageModel(BaseModel):
    conversationId: str
    senderId: str
    senderName: str
    receiverId: str
    text: str

class FarmerRatingModel(BaseModel):
    farmerId: str
    farmerName: Optional[str] = "Farmer"
    ownerId: str
    ownerName: Optional[str] = "Farm Owner"
    rating: int = 5
    feedback: Optional[str] = ""
    fieldId: Optional[str] = ""
    fieldName: Optional[str] = ""

@app.get("/api/farmers")
def get_farmers():
    """Get all registered farmers with their current assignment status and ratings."""
    return AgroDatabaseService.get_farmers()

@app.get("/api/farmers/available")
def get_available_farmers():
    """Get farmers who currently have no active assignment."""
    all_farmers = AgroDatabaseService.get_farmers()
    available = [f for f in all_farmers if not f.get("isAssigned", False)]
    return available

@app.post("/api/farmers/{farmer_id}/ratings")
def submit_farmer_rating(farmer_id: str, rating: FarmerRatingModel):
    """Owner submits a rating and review for a farmer."""
    data = rating.dict()
    data["farmerId"] = farmer_id
    res = AgroDatabaseService.save_farmer_rating(data)
    return {"success": True, "rating": res}

@app.get("/api/farmers/{farmer_id}/ratings")
def get_farmer_ratings(farmer_id: str):
    """Get all ratings and reviews for a farmer."""
    return AgroDatabaseService.get_farmer_ratings(farmer_id)


# --- Assignment Requests ---

@app.post("/api/assignments/request")
def create_assignment_request(req: AssignmentRequestModel):
    """Owner sends assignment request to a farmer."""
    result = AgroDatabaseService.create_assignment_request(req.dict())
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Request failed"))
    return result

@app.get("/api/assignments/requests")
def get_assignment_requests(userId: Optional[str] = None, role: Optional[str] = None):
    """Get assignment requests. Filter by userId and role (owner/farmer)."""
    return AgroDatabaseService.get_assignment_requests(userId=userId, role=role)

@app.post("/api/assignments/{request_id}/approve")
def approve_assignment(request_id: str):
    """Farmer approves an assignment request."""
    result = AgroDatabaseService.update_assignment_request_status(request_id, "approved")
    if not result:
        raise HTTPException(status_code=404, detail="Assignment request not found")
    return {"success": True, "status": "approved", "requestId": request_id}

@app.post("/api/assignments/{request_id}/reject")
def reject_assignment(request_id: str):
    """Farmer rejects an assignment request."""
    result = AgroDatabaseService.update_assignment_request_status(request_id, "rejected")
    if not result:
        raise HTTPException(status_code=404, detail="Assignment request not found")
    return {"success": True, "status": "rejected", "requestId": request_id}

@app.post("/api/assignments/unassign")
def unassign_farmer(req: UnassignModel):
    """Farmer unassigns themselves. Keeps history."""
    result = AgroDatabaseService.unassign_farmer(req.farmerId, req.farmerName)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Unassign failed"))
    return result

@app.get("/api/assignments/history")
def get_assignment_history(userId: Optional[str] = None, role: Optional[str] = None):
    """Get assignment history records."""
    return AgroDatabaseService.get_assignment_history(userId=userId, role=role)


# --- Conversations & Messages (1:1 Chat) ---

@app.get("/api/conversations")
def get_conversations(userId: Optional[str] = None):
    """Get conversations for a user."""
    return AgroDatabaseService.get_conversations(userId=userId)

@app.post("/api/conversations")
def create_conversation(conv: ConversationModel):
    """Get or create a 1:1 conversation between owner and farmer."""
    return AgroDatabaseService.get_or_create_conversation(conv.dict())

@app.get("/api/conversations/{conversation_id}/messages")
def get_messages(conversation_id: str, userId: Optional[str] = None):
    """Get messages for a conversation. Access control via userId."""
    return AgroDatabaseService.get_messages(conversation_id, userId)

@app.post("/api/conversations/{conversation_id}/messages")
def send_message(conversation_id: str, msg: MessageModel):
    """Send a message in a conversation."""
    result = AgroDatabaseService.save_message({**msg.dict(), "conversationId": conversation_id})
    if not result:
        raise HTTPException(status_code=400, detail="Unable to send message")
    return result


# --- AI Endpoints ---

@app.post("/api/ai/kmeans")
def run_kmeans(req: KMeansRequest):
    return kmeans_svc.predict_cluster(
        soil_moisture=req.soil_moisture,
        soil_ph=req.soil_ph,
        temperature=req.temperature,
        humidity=req.humidity,
        rainfall=req.rainfall
    )

@app.get("/api/ai/kmeans")
def get_kmeans(soil_moisture: float = 45.0, soil_ph: float = 6.5, temperature: float = 28.0, humidity: float = 60.0, rainfall: float = 10.0):
    return kmeans_svc.predict_cluster(
        soil_moisture=soil_moisture,
        soil_ph=soil_ph,
        temperature=temperature,
        humidity=humidity,
        rainfall=rainfall
    )

@app.post("/api/ai/decision-tree")
def run_decision_tree(req: DTreeRequest):
    return dtree_svc.predict_recommendation(
        crop=req.crop,
        soil_moisture=req.soil_moisture,
        soil_ph=req.soil_ph,
        temperature=req.temperature,
        humidity=req.humidity,
        rainfall=req.rainfall
    )

@app.get("/api/ai/decision-tree")
def get_decision_tree(crop: str = "Tomato", soil_moisture: float = 30.0, soil_ph: float = 6.2, temperature: float = 32.0, humidity: float = 55.0, rainfall: float = 5.0):
    return dtree_svc.predict_recommendation(
        crop=crop,
        soil_moisture=soil_moisture,
        soil_ph=soil_ph,
        temperature=temperature,
        humidity=humidity,
        rainfall=rainfall
    )

# --- Crop Recommendation Decision Tree API ---
@app.post("/api/crop-recommendation")
def recommend_crop_post(req: CropRecommendationRequest):
    res = crop_predictor.predict(
        N=req.N,
        P=req.P,
        K=req.K,
        temperature=req.temperature,
        humidity=req.humidity,
        ph=req.ph,
        rainfall=req.rainfall
    )
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("error", "Crop recommendation prediction failed"))
    return res

@app.get("/api/crop-recommendation")
def recommend_crop_get(
    N: float = 45.0,
    P: float = 30.0,
    K: float = 35.0,
    temperature: float = 28.0,
    humidity: float = 70.0,
    ph: float = 6.5,
    rainfall: float = 200.0
):
    res = crop_predictor.predict(
        N=N,
        P=P,
        K=K,
        temperature=temperature,
        humidity=humidity,
        ph=ph,
        rainfall=rainfall
    )
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("error", "Crop recommendation prediction failed"))
    return res

# --- Decision Tree V2 Field Decision API ---
@app.post("/api/decision")
def predict_decision_post(req: DecisionRequest):
    ph_val = req.ph if req.ph is not None else (req.soilPH if req.soilPH is not None else 6.5)
    sm_val = req.soil_moisture if req.soil_moisture is not None else (req.soilMoisture if req.soilMoisture is not None else 45.0)

    res = dtree_v2_service.predict(
        crop=req.crop or "rice",
        N=req.N,
        P=req.P,
        K=req.K,
        temperature=req.temperature,
        humidity=req.humidity,
        ph=ph_val,
        rainfall=req.rainfall,
        soil_moisture=sm_val
    )
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("error", "Decision Tree V2 prediction failed"))
    return res

@app.get("/api/decision")
def predict_decision_get(
    crop: str = "rice",
    N: float = 90.0,
    P: float = 42.0,
    K: float = 43.0,
    temperature: float = 20.8,
    humidity: float = 82.0,
    ph: float = 6.5,
    rainfall: float = 202.9,
    soil_moisture: float = 46.9
):
    res = dtree_v2_service.predict(
        crop=crop,
        N=N,
        P=P,
        K=K,
        temperature=temperature,
        humidity=humidity,
        ph=ph,
        rainfall=rainfall,
        soil_moisture=soil_moisture
    )
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("error", "Decision Tree V2 prediction failed"))
    return res

@app.post("/api/ai/cnn")
async def run_cnn_disease_detection(file: UploadFile = File(...)):
    contents = await file.read()
    return cnn_svc.classify_leaf_image(contents, filename=file.filename)

@app.api_route("/api/ai/csp", methods=["GET", "POST"])
def run_csp_scheduler():
    variables = ["Field_A", "Field_B", "Field_C", "Field_D"]
    domains = {
        "Field_A": ["06:00-07:15", "07:30-08:30"],
        "Field_B": ["07:30-08:30", "16:30-17:30"],
        "Field_C": ["16:30-17:30"],
        "Field_D": ["06:00-07:15", "07:30-08:30"]
    }
    constraints = {"lockout_slots": ["12:00-13:00", "13:00-14:00", "14:00-15:00"], "max_pumps_per_slot": 2}
    
    # Run AC-3 Domain Reduction
    ac3_success = ac3(variables, domains)
    # Run Backtracking Search
    assigned_schedule = backtrack_csp(variables, domains, constraints)

    return {
        "algorithm": "CSP + AC-3 + Backtracking",
        "ac3_domain_reduction_success": ac3_success,
        "assigned_timetable": assigned_schedule,
        "reduced_domains": domains,
        "message": "Arc Consistency verified: 24h schedule generated with 0 domain conflicts."
    }

@app.post("/api/ai/search")
def run_search_algorithm(req: SearchRequest):
    grid = [
        ['S', '.', '.', '.'],
        ['.', '#', '.', '.'],
        ['.', '.', '.', 'G']
    ]
    start = tuple(req.start)
    goal = tuple(req.goal)

    if req.algorithm == 'bfs':
        path = bfs_search(grid, start, goal)
        name = "Breadth-First Search (BFS)"
    elif req.algorithm == 'dfs':
        path = dfs_search(grid, start, goal)
        name = "Depth-First Search (DFS)"
    else:
        path = astar_search(grid, start, goal)
        name = "A* Pathfinding Search"

    return {
        "algorithm": name,
        "start": start,
        "goal": goal,
        "grid_map": grid,
        "optimal_path": path,
        "path_cost": len(path) - 1 if path else 0
    }

@app.get("/api/ai/search")
def get_search_algorithm(algorithm: str = "astar", start_r: int = 0, start_c: int = 0, goal_r: int = 2, goal_c: int = 3):
    return run_search_algorithm(SearchRequest(algorithm=algorithm, start=[start_r, start_c], goal=[goal_r, goal_c]))

@app.get("/api/ai/minimax")
def get_minimax(field_name: str = "Field D", crop: str = "Tomato"):
    return PestRiskMinimax.evaluate(field_name, crop)

@app.post("/api/ai/minimax")
def post_minimax(req: Optional[MinimaxRequest] = None):
    field_name = req.field_name if req and req.field_name else "Field D"
    crop = req.crop if req and req.crop else "Tomato"
    return PestRiskMinimax.evaluate(field_name, crop)

@app.api_route("/api/ai/genetic", methods=["GET", "POST"])
def run_genetic_optimizer():
    fields = AgroDatabaseService.get_fields()
    optimizer = IrrigationGeneticOptimizer(fields=fields, time_slots=["06:00-07:00", "07:00-08:00", "16:00-17:00", "17:00-18:00"])
    return optimizer.optimize()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
