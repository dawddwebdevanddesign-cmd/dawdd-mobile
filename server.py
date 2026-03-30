import os
import logging
import uuid
import asyncio
import httpx
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, APIRouter, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# ==================== CONFIGURATION ====================

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Critical Environment Check
try:
    MONGO_URL = os.environ['MONGO_URL']
    # Defaulting DB_NAME to 'dawdd_db' if not in .env to match your Atlas string
    DB_NAME = os.environ.get('DB_NAME', 'dawdd_db')
    ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', '*').split(',')
except KeyError as e:
    raise RuntimeError(f"Missing required environment variable: {e}")

# MongoDB Connection
client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]

# Emergent LLM Key
EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY', '')

# Main App Initialization
app = FastAPI(title="D.A.W.D.D. API", version="1.0.0")
api_router = APIRouter(prefix="/api")

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== ENUMS ====================

class ProjectStage(str, Enum):
    LEAD = "lead"
    DISCOVERY = "discovery"
    DESIGN = "design"
    DEVELOPMENT = "development"
    REVIEW = "review"
    DEPLOY = "deploy"
    MAINTENANCE = "maintenance"

class ServiceType(str, Enum):
    CLOUDFLARE = "cloudflare"
    NETLIFY = "netlify"
    GITHUB = "github"
    ZOHO = "zoho"
    IMPROVMX = "improvmx"

# ==================== MODELS ====================

class APIKeyCreate(BaseModel):
    service: ServiceType
    api_key: str
    email: Optional[str] = None
    account_id: Optional[str] = None

class APIKeyResponse(BaseModel):
    id: str
    service: ServiceType
    email: Optional[str] = None
    account_id: Optional[str] = None
    is_valid: bool = False
    created_at: datetime

class ProjectCreate(BaseModel):
    name: str
    client_name: str
    domain: Optional[str] = None
    description: Optional[str] = None

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    client_name: Optional[str] = None
    domain: Optional[str] = None
    description: Optional[str] = None
    stage: Optional[ProjectStage] = None

class Project(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    client_name: str
    domain: Optional[str] = None
    description: Optional[str] = None
    stage: ProjectStage = ProjectStage.LEAD
    github_repo: Optional[str] = None
    netlify_site_id: Optional[str] = None
    cloudflare_zone_id: Optional[str] = None
    email_configured: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class DNSRecord(BaseModel):
    type: str
    name: str
    content: str
    ttl: int = 3600
    priority: Optional[int] = None

class AIAssistRequest(BaseModel):
    project_id: Optional[str] = None
    query: str
    context: Optional[str] = None

# ==================== HELPER FUNCTIONS ====================

async def get_api_key(service: ServiceType) -> Optional[Dict]:
    return await db.api_keys.find_one({"service": service})

async def validate_cloudflare_key(api_key: str, account_id: str = None) -> bool:
    try:
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            response = await client.get("https://api.cloudflare.com/client/v4/user/tokens/verify", headers=headers)
            return response.status_code == 200 and response.json().get("success", False)
    except Exception as e:
        logger.error(f"Cloudflare validation error: {e}")
        return False

# ==================== API KEY ROUTES ====================

@api_router.post("/api-keys", response_model=APIKeyResponse)
async def store_api_key(key_data: APIKeyCreate):
    is_valid = False
    if key_data.service == ServiceType.CLOUDFLARE:
        is_valid = await validate_cloudflare_key(key_data.api_key, key_data.account_id)
    # (Add other validation calls here as needed)
    
    key_doc = {
        "id": str(uuid.uuid4()),
        "service": key_data.service,
        "api_key": key_data.api_key,
        "email": key_data.email,
        "account_id": key_data.account_id,
        "is_valid": is_valid,
        "created_at": datetime.utcnow()
    }
    await db.api_keys.update_one({"service": key_data.service}, {"$set": key_doc}, upsert=True)
    return APIKeyResponse(**{k: v for k, v in key_doc.items() if k != "api_key"})

@api_router.get("/api-keys", response_model=List[APIKeyResponse])
async def get_api_keys():
    keys = await db.api_keys.find().to_list(100)
    return [APIKeyResponse(**{k: v for k, v in key.items() if k not in ["api_key", "_id"]}) for key in keys]

# ==================== PROJECT ROUTES ====================

@api_router.post("/projects", response_model=Project)
async def create_project(project_data: ProjectCreate):
    project = Project(**project_data.dict())
    await db.projects.insert_one(project.dict())
    return project

@api_router.get("/projects", response_model=List[Project])
async def get_projects(
    stage: Optional[ProjectStage] = None,
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0)
):
    query = {"stage": stage} if stage else {}
    projects = await db.projects.find(query).sort("created_at", -1).skip(offset).limit(limit).to_list(limit)
    return [Project(**p) for p in projects]

@api_router.get("/projects/{project_id}", response_model=Project)
async def get_project(project_id: str):
    project = await db.projects.find_one({"id": project_id})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return Project(**project)

# ==================== DASHBOARD STATS (STEP 2 OPTIMIZED) ====================

@api_router.get("/dashboard/stats")
async def get_dashboard_stats():
    """Get optimized dashboard statistics using a single MongoDB Aggregation"""
    pipeline = [
        {
            "$facet": {
                "stage_counts": [
                    {"$group": {"_id": "$stage", "count": {"$sum": 1}}}
                ],
                "total_projects": [
                    {"$count": "total"}
                ],
                "recent_projects": [
                    {"$sort": {"updated_at": -1}},
                    {"$limit": 5}
                ]
            }
        }
    ]
    
    result = await db.projects.aggregate(pipeline).to_list(1)
    stats = result[0] if result else {}

    # Format stage counts
    formatted_stages = {stage.value: 0 for stage in ProjectStage}
    for item in stats.get("stage_counts", []):
        if item["_id"]:
            formatted_stages[item["_id"]] = item["count"]

    # API configuration status
    api_keys = await db.api_keys.find().to_list(100)
    services_configured = len([k for k in api_keys if k.get("is_valid", False)])

    return {
        "total_projects": stats.get("total_projects", [{"total": 0}])[0]["total"],
        "stage_counts": formatted_stages,
        "services_configured": services_configured,
        "total_services": 5,
        "recent_projects": stats.get("recent_projects", [])
    }

# ==================== CORE ROUTES ====================

@api_router.get("/")
async def root():
    return {"message": "D.A.W.D.D. API is running", "version": "1.0.0"}

@api_router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# Include router and middleware
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
