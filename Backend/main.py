from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import datetime, timedelta
import numpy as np
import pickle
from pydantic import BaseModel, EmailStr
from sklearn.preprocessing import StandardScaler
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import logging
from passlib.context import CryptContext
import secrets
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import Optional
from retrive import predict_risk, generate_recommendation
from motor.motor_asyncio import AsyncIOMotorClient
import traceback  # Add this import at the top of the file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Load the trained models
try:
    kmeans_model = pickle.load(open("models/kmeans_model.pkl", "rb"))
    scaler = pickle.load(open("models/scaler.pkl", "rb"))
    logger.info("ML models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise

# MongoDB connection
MONGO_DETAILS = "mongodb+srv://prajwal2403:Mysql321@prajwal2403.s8a1j.mongodb.net/risk_analysis?retryWrites=true&w=majority"

try:
    # Initialize MongoDB client with timeout and SSL
    client = AsyncIOMotorClient(
        MONGO_DETAILS,
        serverSelectionTimeoutMS=5000,
        ssl=True,
        tlsAllowInvalidCertificates=False
    )
    # Test the connection
    client.admin.command('ping')
    logger.info("Connected to MongoDB successfully!")
    
    db = client.risk_analysis
    users_collection = db.user  # Consistent collection naming
    
except ConnectionFailure as e:
    logger.error(f"Could not connect to MongoDB: {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected MongoDB error: {e}")
    raise

# FastAPI instance
app = FastAPI(title="Health Risk Assessment API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Pydantic models
class PatientData(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigree: float
    Age: int

class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserInDB(BaseModel):
    id: str
    name: str
    email: EmailStr
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

class UserResponse(BaseModel):
    id: str
    name: str
    email: EmailStr

# Helper functions
def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception
    
    user = await users_collection.find_one({"email": token_data.email})
    if user is None:
        raise credentials_exception
    return user

# API endpoints
@app.post("/predict", response_model=dict)
async def predict_cluster(data: PatientData, current_user: dict = Depends(get_current_user)):
    print(f"Received data: {data}")
    try:
        # Convert user input into a NumPy array
        patient_dict = data.dict()
        user_input = np.array([[patient_dict[feature] for feature in [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigree', 'Age'
        ]]])

        print(f"User input: {user_input}")

        predicted_risk = predict_risk(patient_dict)  # now pass dict
        recommendation = generate_recommendation(predicted_risk, patient_dict)

        return {
            "risk": predicted_risk,
            "recommendation": recommendation,
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/signup/", response_model=UserResponse)
async def create_user(user: UserCreate):
    try:
        # Check if user already exists
        logger.info(f"Checking if user with email {user.email} already exists.")
        existing_user = await users_collection.find_one({"email": user.email})
        if existing_user:
            logger.warning(f"User with email {user.email} already exists.")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

        # Create new user
        logger.info(f"Creating new user with email {user.email}.")
        user_dict = user.dict()
        user_dict["hashed_password"] = get_password_hash(user.password)
        del user_dict["password"]  # Remove plaintext password
        
        # Insert new user
        result = await users_collection.insert_one(user_dict)
        if not result.inserted_id:
            logger.error("Failed to insert new user into the database.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="User creation failed"
            )

        # Return the created user (without password)
        logger.info(f"Retrieving created user with ID {result.inserted_id}.")
        created_user = await users_collection.find_one({"_id": result.inserted_id})
        if not created_user:
            logger.error("Failed to retrieve the created user from the database.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not retrieve created user"
            )

        logger.info(f"User with email {user.email} created successfully.")
        return {
            "id": str(created_user["_id"]),
            "name": created_user["name"],
            "email": created_user["email"]
        }
    except Exception as e:
        logger.error(f"Signup error: {e}")
        logger.error(traceback.format_exc())  # Log the full traceback
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    try:
        user = await users_collection.find_one({"email": form_data.username})
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )

        if not verify_password(form_data.password, user["hashed_password"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )

        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user["email"]},
            expires_delta=access_token_expires
        )

        return {"access_token": access_token, "token_type": "bearer"}
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.get("/users/me/", response_model=UserInDB)
async def read_users_me(current_user: dict = Depends(get_current_user)):
    return {
        "id": str(current_user["_id"]),
        "name": current_user["name"],
        "email": current_user["email"]
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    try:
        # Check MongoDB connection
        client.admin.command('ping')
        return {
            "status": "healthy",
            "database": "connected",
            "models_loaded": True
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unavailable: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)