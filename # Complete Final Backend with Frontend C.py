# Complete Final Backend with Frontend Compatibility and Full Synchronization
# Combines Option 1 (frontend compatibility) with comprehensive improvements

import asyncio
import json
import uvicorn
import sys
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Depends, Security, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel , Field
import logging
import hashlib
import secrets
import bcrypt
from collections import defaultdict
import random
import os

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quantum_backend.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("quantum_cybersecurity")

# Database Setup
DATABASE_URL = "sqlite:///quantum_cybersecurity.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class LoginRequest(BaseModel):
    username: str
    password: str

class DBLogEntry(Base):
    __tablename__ = "log_entries"
    
    id = Column(Integer, primary_key=True, index=True)
    event = Column(String, index=True)
    details = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    hash_prev = Column(String)
    hash_current = Column(String)

class DBGPSData(Base):
    __tablename__ = "gps_data"
    
    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    encrypted_payload = Column(Text)
    quantum_token_id = Column(String)
    quantum_token_theta = Column(Float)
    quantum_token_phi = Column(Float)

class DBUser(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)
    role = Column(String, default="viewer")
    created_at = Column(DateTime, default=datetime.utcnow)

class DBDeviceStatus(Base):
    __tablename__ = "device_status"
    
    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String, index=True)
    status = Column(String, default="active")
    last_seen = Column(DateTime, default=datetime.utcnow)
    latitude = Column(Float)
    longitude = Column(Float)

def create_db_tables():
    """Create database tables"""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Database dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic Models
class GPSData(BaseModel):
    device_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    accuracy: Optional[float] = None
    encrypted_payload: Optional[str] = None

class QuantumToken(BaseModel):
    token_id: str
    device_id: str
    q_state: str = "GENERATED"
    theta: float
    phi: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class LogEntry(BaseModel):
    id: int
    event: str
    details: Dict[str, Any]
    timestamp: datetime
    hash_prev: str
    hash_current: str

class SealEvent(BaseModel):
    vehicle_id: str
    event_type: str
    location: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    user_token: str

class SecurityEvent(BaseModel):
    event_id: str
    timestamp: datetime
    event_type: str
    severity: str
    source_ip: str
    user_agent: Optional[str] = None
    endpoint: Optional[str] = None
    details: Dict[str, Any] = {}
    action_taken: str = "LOGGED"

# Core Components
class PostQuantumCrypto:
    """Simulated Post-Quantum Cryptography implementation"""
    
    def __init__(self):
        self.algorithm = "CRYSTALS-Kyber"
        logger.info("PostQuantumCrypto initialized with CRYSTALS-Kyber simulation")
    
    def encrypt(self, data: str, quantum_token: QuantumToken) -> str:
        """Encrypt data using quantum token parameters"""
        try:
            key_material = f"{quantum_token.theta}:{quantum_token.phi}:{quantum_token.token_id}"
            key_hash = hashlib.sha256(key_material.encode()).hexdigest()
            encrypted = hashlib.sha256(f"{data}:{key_hash}".encode()).hexdigest()
            return encrypted
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt(self, encrypted_data: str, quantum_token: QuantumToken) -> str:
        """Decrypt data using quantum token parameters"""
        try:
            return f"DECRYPTED_DATA_FOR_{quantum_token.device_id}"
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise

class QuantumTokenSimulator:
    """Quantum token simulation system"""
    
    def __init__(self):
        self.active_tokens = {}
        logger.info("QuantumTokenSimulator initialized")
    
    def generate_quantum_token(self, device_id: str) -> QuantumToken:
        """Generate a quantum authentication token"""
        token = QuantumToken(
            token_id=f"QT_{secrets.token_hex(8)}",
            device_id=device_id,
            theta=random.uniform(0, 360),
            phi=random.uniform(0, 180),
        )
        
        self.active_tokens[token.token_id] = token
        logger.info(f"Generated quantum token {token.token_id} for device {device_id}")
        return token
    
    def verify_quantum_token(self, token: QuantumToken) -> bool:
        """Verify quantum token authenticity"""
        stored_token = self.active_tokens.get(token.token_id)
        if not stored_token:
            return False
        
        return (stored_token.device_id == token.device_id and
                stored_token.theta == token.theta and
                stored_token.phi == token.phi)
    
    def simulate_qkd_token_exchange(self, sender_id: str, receiver_id: str) -> Optional[QuantumToken]:
        """Simulate QKD token exchange between devices"""
        if random.random() < 0.9:  # 90% success rate
            shared_token = QuantumToken(
                token_id=f"QKD_{secrets.token_hex(8)}",
                device_id=f"{sender_id}_{receiver_id}",
                theta=random.uniform(0, 360),
                phi=random.uniform(0, 180)
            )
            self.active_tokens[shared_token.token_id] = shared_token
            logger.info(f"QKD exchange successful between {sender_id} and {receiver_id}")
            return shared_token
        else:
            logger.warning(f"QKD exchange failed between {sender_id} and {receiver_id}")
            return None

class TamperEvidentLogger:
    """Tamper-evident logging system using blockchain-like hash chains"""
    
    def __init__(self):
        self.genesis_hash = "0" * 64
        logger.info("TamperEvidentLogger initialized")
    
    def _calculate_hash(self, event: str, details: Dict[str, Any], timestamp: datetime, prev_hash: str) -> str:
        """Calculate hash for log entry"""
        data = f"{event}:{json.dumps(details, sort_keys=True)}:{timestamp.isoformat()}:{prev_hash}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def add_log_entry(self, event: str, details: Dict[str, Any], db: Session):
        """Add tamper-evident log entry to database"""
        try:
            last_entry = db.query(DBLogEntry).order_by(DBLogEntry.id.desc()).first()
            prev_hash = last_entry.hash_current if last_entry else self.genesis_hash
            
            timestamp = datetime.now(timezone.utc)
            current_hash = self._calculate_hash(event, details, timestamp, prev_hash)
            
            log_entry = DBLogEntry(
                event=event,
                details=json.dumps(details),
                timestamp=timestamp,
                hash_prev=prev_hash,
                hash_current=current_hash
            )
            
            db.add(log_entry)
            db.commit()
            
            logger.info(f"Log entry added: {event}")
            return log_entry.id
            
        except Exception as e:
            logger.error(f"Failed to add log entry: {e}")
            db.rollback()
            raise
    
    def verify_chain_integrity(self, db: Session) -> bool:
        """Verify the integrity of the hash chain"""
        try:
            entries = db.query(DBLogEntry).order_by(DBLogEntry.id).all()
            
            if not entries:
                return True
            
            # Check first entry
            first_entry = entries[0]
            expected_hash = self._calculate_hash(
                first_entry.event,
                json.loads(first_entry.details),
                first_entry.timestamp,
                first_entry.hash_prev
            )
            
            if first_entry.hash_current != expected_hash:
                logger.error("First entry hash mismatch")
                return False
            
            # Check subsequent entries
            for i in range(1, len(entries)):
                current_entry = entries[i]
                prev_entry = entries[i-1]
                
                if current_entry.hash_prev != prev_entry.hash_current:
                    logger.error(f"Hash chain broken at entry {current_entry.id}")
                    return False
                
                expected_hash = self._calculate_hash(
                    current_entry.event,
                    json.loads(current_entry.details),
                    current_entry.timestamp,
                    current_entry.hash_prev
                )
                
                if current_entry.hash_current != expected_hash:
                    logger.error(f"Hash mismatch at entry {current_entry.id}")
                    return False
            
            logger.info("Hash chain integrity verified")
            return True
            
        except Exception as e:
            logger.error(f"Chain integrity verification failed: {e}")
            return False

class GPSDataManager:
    """GPS data management with encryption"""
    
    def __init__(self):
        logger.info("GPSDataManager initialized")
    
    def store_gps_data(self, gps_data: GPSData, quantum_token: QuantumToken, db: Session) -> int:
        """Store encrypted GPS data"""
        try:
            db_gps = DBGPSData(
                device_id=gps_data.device_id,
                timestamp=gps_data.timestamp,
                latitude=gps_data.latitude,
                longitude=gps_data.longitude,
                encrypted_payload=gps_data.encrypted_payload,
                quantum_token_id=quantum_token.token_id,
                quantum_token_theta=quantum_token.theta,
                quantum_token_phi=quantum_token.phi
            )
            
            db.add(db_gps)
            db.commit()
            
            logger.info(f"GPS data stored for device {gps_data.device_id}")
            return db_gps.id
            
        except Exception as e:
            logger.error(f"Failed to store GPS data: {e}")
            db.rollback()
            raise
    
    def get_gps_data_for_device(self, device_id: str, db: Session) -> List[Dict[str, Any]]:
        """Retrieve GPS data for a device"""
        try:
            records = db.query(DBGPSData).filter(DBGPSData.device_id == device_id).all()
            
            result = []
            for record in records:
                result.append({
                    "id": record.id,
                    "device_id": record.device_id,
                    "timestamp": record.timestamp,
                    "latitude": record.latitude,
                    "longitude": record.longitude,
                    "encrypted_payload": record.encrypted_payload,
                    "quantum_token_id": record.quantum_token_id,
                    "quantum_token_theta": record.quantum_token_theta,
                    "quantum_token_phi": record.quantum_token_phi
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to retrieve GPS data: {e}")
            raise

class AuthManager:
    """Authentication and authorization manager"""
    
    def __init__(self):
        self.user_roles = {
            "admin": ["read", "write", "audit", "admin"],
            "operator": ["read", "write"],
            "auditor": ["read", "audit"],
            "viewer": ["read"]
        }
        self.active_sessions = {}
        logger.info("AuthManager initialized")
    
    def initialize_default_users(self, db: Session):
        """Initialize default users"""
        try:
            default_users = [
                ("admin", "admin123", "admin"),
                ("operator", "operator123", "operator"),
                ("auditor", "auditor123", "auditor"),
                ("viewer", "viewer123", "viewer")
            ]
            
            for username, password, role in default_users:
                existing_user = db.query(DBUser).filter(DBUser.username == username).first()
                if not existing_user:
                    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                    user = DBUser(
                        username=username,
                        password_hash=password_hash,
                        role=role
                    )
                    db.add(user)
            
            db.commit()
            logger.info("Default users initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize default users: {e}")
            db.rollback()
    
    def authenticate_user(self, username: str, password: str, db: Session) -> Optional[str]:
        """Authenticate user and return token"""
        try:
            user = db.query(DBUser).filter(DBUser.username == username).first()
            if user and bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
                token = secrets.token_urlsafe(32)
                self.active_sessions[token] = {
                    "username": username,
                    "role": user.role,
                    "token": token,
                    "created_at": datetime.now(timezone.utc)
                }
                logger.info(f"User {username} authenticated successfully")
                return token
            else:
                logger.warning(f"Authentication failed for user {username}")
                return None
                
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify authentication token"""
        session = self.active_sessions.get(token)
        if session:
            # Check if token is expired (24 hours)
            if (datetime.now(timezone.utc) - session["created_at"]).total_seconds() > 86400:
                del self.active_sessions[token]
                return None
            return session
        return None
    
    def check_permission(self, token: str, required_permission: str) -> bool:
        """Check if user has required permission"""
        session = self.verify_token(token)
        if not session:
            return False
        
        user_role = session.get("role", "viewer")
        user_permissions = self.user_roles.get(user_role, [])
        return required_permission in user_permissions

class SecurityMonitor:
    """Security event monitoring"""
    
    def __init__(self):
        self.security_events = []
        self.blocked_ips = set()
        self.rate_limits = defaultdict(list)
        logger.info("SecurityMonitor initialized")
    
    def log_security_event(self, event: Dict[str, Any]):
        """Log security event"""
        self.security_events.append(event)
        logger.warning(f"Security event: {event['event_type']} from {event.get('source_ip', 'unknown')}")
    
    def is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is blocked"""
        return ip in self.blocked_ips
    
    def block_ip(self, ip: str, reason: str):
        """Block IP address"""
        self.blocked_ips.add(ip)
        self.log_security_event({
            "event_id": f"IP_BLOCK_{int(datetime.now().timestamp())}",
            "timestamp": datetime.now(timezone.utc),
            "event_type": "IP_BLOCKED",
            "severity": "HIGH",
            "source_ip": ip,
            "details": {"reason": reason},
            "action_taken": "BLOCKED"
        })
    
    def unblock_ip(self, ip: str):
        """Unblock IP address"""
        self.blocked_ips.discard(ip)
        self.log_security_event({
            "event_id": f"IP_UNBLOCK_{int(datetime.now().timestamp())}",
            "timestamp": datetime.now(timezone.utc),
            "event_type": "IP_UNBLOCKED",
            "severity": "INFO",
            "source_ip": ip,
            "action_taken": "UNBLOCKED"
        })

class AdvancedRateLimiter:
    """Advanced rate limiting system"""
    
    def __init__(self):
        self.rate_limits = defaultdict(list)
        self.limits = {
            "default": (1000, 3600),  # 1000 requests per hour (increased)
            "auth": (100, 300),       # 100 auth attempts per 5 minutes (increased)
            "admin": (500, 3600)      # 500 admin operations per hour (increased)
        }
    
    def is_rate_limited(self, identifier: str, limit_type: str = "default") -> bool:
        """Check if identifier is rate limited"""
        max_requests, window_seconds = self.limits.get(limit_type, self.limits["default"])
        now = datetime.now()
        
        # Clean old entries
        cutoff = now - timedelta(seconds=window_seconds)
        self.rate_limits[identifier] = [
            timestamp for timestamp in self.rate_limits[identifier]
            if timestamp > cutoff
        ]
        
        # Check limit
        if len(self.rate_limits[identifier]) >= max_requests:
            return True
        
        # Add current request
        self.rate_limits[identifier].append(now)
        return False

class GPSSimulator:
    """GPS data simulator for realistic device tracking"""
    
    def __init__(self):
        # NYC area coordinates for simulation
        self.base_locations = {
            'QV001': {'lat': 40.7128, 'lng': -74.0060, 'name': 'Manhattan'},
            'QV002': {'lat': 40.7589, 'lng': -73.9851, 'name': 'Central Park'},
            'QV003': {'lat': 40.6782, 'lng': -73.9442, 'name': 'Brooklyn'},
            'QV004': {'lat': 40.7831, 'lng': -73.9712, 'name': 'Bronx'},
            'QV005': {'lat': 40.7282, 'lng': -73.7949, 'name': 'Queens'}
        }
        logger.info("GPS Simulator initialized")
    
    def generate_gps_data(self, device_id: str, num_points: int = 5) -> List[Dict]:
        """Generate realistic GPS tracking data for a device"""
        if device_id not in self.base_locations:
            base_loc = self.base_locations['QV001']
        else:
            base_loc = self.base_locations[device_id]
        
        gps_points = []
        current_time = datetime.now(timezone.utc)
        
        for i in range(num_points):
            # Add some random movement around base location
            lat_offset = random.uniform(-0.01, 0.01)
            lng_offset = random.uniform(-0.01, 0.01)
            
            gps_point = {
                'device_id': device_id,
                'latitude': base_loc['lat'] + lat_offset,
                'longitude': base_loc['lng'] + lng_offset,
                'timestamp': current_time - timedelta(hours=i),
                'accuracy': random.uniform(3.0, 15.0),
                'altitude': random.uniform(0, 100)
            }
            gps_points.append(gps_point)
        
        return gps_points

# Main Application Class
class QuantumCybersecuritySystem:
    """Complete quantum cybersecurity system integrating all checkpoints"""
    
    def __init__(self):
        # Core components
        self.pqc = PostQuantumCrypto()
        self.quantum_simulator = QuantumTokenSimulator()
        self.gps_manager = GPSDataManager()
        self.auth_manager = AuthManager()
        self.tamper_logger = TamperEvidentLogger()
        self.security_monitor = SecurityMonitor()
        self.rate_limiter = AdvancedRateLimiter()
        self.gps_simulator = GPSSimulator()
        
        logger.info("Quantum Cybersecurity System initialized with all checkpoints")
    
    async def startup(self, db: Session):
        """System startup procedures"""
        logger.info("Starting Quantum Cybersecurity System...")
        
        # Verify system integrity
        integrity_check = self.tamper_logger.verify_chain_integrity(db)
        if not integrity_check:
            logger.warning("Hash chain integrity check failed - may be first startup")
        
        # Initialize demo devices
        self.initialize_demo_devices(db)
        
        # Start background tasks
        asyncio.create_task(self._background_monitoring())
        
        logger.info("System startup complete - All checkpoints operational")
    
    def initialize_demo_devices(self, db: Session):
        """Initialize demo devices with GPS data"""
        try:
            demo_devices = ['QV001', 'QV002', 'QV003', 'QV004', 'QV005']
            
            for device_id in demo_devices:
                # Check if device already has data
                existing = db.query(DBGPSData).filter(DBGPSData.device_id == device_id).first()
                if not existing:
                    # Generate GPS history for device
                    gps_points = self.gps_simulator.generate_gps_data(device_id, 10)
                    
                    for point in gps_points:
                        # Generate quantum token for each GPS point
                        token = self.quantum_simulator.generate_quantum_token(device_id)
                        
                        # Create GPS record
                        gps_record = DBGPSData(
                            device_id=point['device_id'],
                            timestamp=point['timestamp'],
                            latitude=point['latitude'],
                            longitude=point['longitude'],
                            encrypted_payload=f"encrypted_gps_data_{secrets.token_hex(8)}",
                            quantum_token_id=token.token_id,
                            quantum_token_theta=token.theta,
                            quantum_token_phi=token.phi
                        )
                        db.add(gps_record)
                    
                    # Add device status
                    device_status = DBDeviceStatus(
                        device_id=device_id,
                        status='active',
                        last_seen=datetime.now(timezone.utc),
                        latitude=gps_points[0]['latitude'],
                        longitude=gps_points[0]['longitude']
                    )
                    db.add(device_status)
            
            db.commit()
            logger.info("Demo devices initialized with GPS data")
            
        except Exception as e:
            logger.error(f"Failed to initialize demo devices: {e}")
            db.rollback()
    
    async def shutdown(self):
        """Graceful system shutdown"""
        logger.info("Shutting down Quantum Cybersecurity System...")
        logger.info("System shutdown complete")
    
    async def _background_monitoring(self):
        """Background security monitoring task"""
        while True:
            try:
                # Clean up old security events every 5 minutes
                cutoff = datetime.now(timezone.utc) - timedelta(days=7)
                self.security_monitor.security_events = [
                    e for e in self.security_monitor.security_events
                    if e["timestamp"] > cutoff
                ]
                
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Background monitoring error: {e}")
                await asyncio.sleep(60)  # Retry in 1 minute

# FastAPI Application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Create database tables on startup
    create_db_tables()
    logger.info("Database tables checked/created.")
    
    # Initialize system
    db = SessionLocal()
    try:
        system = app.state.quantum_system
        system.auth_manager.initialize_default_users(db)
        await system.startup(db)
    finally:
        db.close()
    
    yield
    
    # Shutdown
    await app.state.quantum_system.shutdown()

def create_production_app():
    """Create production-ready FastAPI application"""
    
    app = FastAPI(
        title="Quantum Cybersecurity Backend - Complete",
        description="Complete quantum-safe cybersecurity system with frontend compatibility",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc"
    )
    
    # Initialize system
    quantum_system = QuantumCybersecuritySystem()
    app.state.quantum_system = quantum_system
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    
    # Security middleware
    @app.middleware("http")
    async def security_middleware(request: Request, call_next):
        client_ip = request.client.host
        
        # Check if IP is blocked
        if quantum_system.security_monitor.is_ip_blocked(client_ip):
            from starlette.responses import Response
            return Response("Forbidden", status_code=403)
        
        # Check rate limiting
        endpoint = str(request.url.path)
        limit_type = "auth" if "/auth/" in endpoint else "default"
        
        if quantum_system.rate_limiter.is_rate_limited(client_ip, limit_type):
            quantum_system.security_monitor.log_security_event({
                "event_id": f"RATE_LIMIT_{int(datetime.now().timestamp())}",
                "timestamp": datetime.now(timezone.utc),
                "event_type": "RATE_LIMIT_EXCEEDED",
                "severity": "MEDIUM",
                "source_ip": client_ip,
                "endpoint": endpoint,
                "details": {"limit_type": limit_type},
                "action_taken": "BLOCKED"
            })
            from starlette.responses import Response
            return Response("Rate limit exceeded", status_code=429)
        
        response = await call_next(request)
        return response
    
    # Security dependency
    security = HTTPBearer()
    
    def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
        token_info = quantum_system.auth_manager.verify_token(credentials.credentials)
        if not token_info:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication"
            )
        return token_info

    # ==================== API ENDPOINTS ====================
    

    

    @app.get("/", tags=["System Management"])
    async def root():
        """Root endpoint with system information"""
        return {
            "system": "Quantum Cybersecurity Backend",
            "version": "1.0.0",
            "description": "Production-ready quantum-safe cybersecurity system",
            "features": [
                "Post-Quantum Cryptography",
                "Quantum Token Authentication",
                "Tamper-Evident Logging",
                "Advanced Threat Detection",
                "Real-time Security Monitoring",
                "Comprehensive Auditing",
                "GPS Device Management",
                "Frontend Integration"
            ],
            "endpoints": {
                "api_docs": "/api/docs",
                "status": "/api/v1/system/status",
                "health": "/api/v1/system/health",
                "login": "/api/v1/auth/login"
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    # Authentication Endpoints
    @app.post("/api/v1/auth/login", tags=["Authentication"])
    async def login(login_req: LoginRequest, request: Request, db: Session = Depends(get_db)):
        username = login_req.username
        password = login_req.password
        client_ip = request.client.host
        
        # Attempt authentication
        token = quantum_system.auth_manager.authenticate_user(username, password, db)
        
        if token:
            quantum_system.tamper_logger.add_log_entry("USER_LOGIN_SUCCESS", {
                "username": username,
                "client_ip": client_ip,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }, db)
            
            return {
                "access_token": token,
                "token_type": "bearer",
                "expires_in": 86400,
                "user_info": {
                    "username": username,
                    "role": quantum_system.auth_manager.active_sessions[token]["role"],
                    "permissions": quantum_system.auth_manager.user_roles.get(
                        quantum_system.auth_manager.active_sessions[token]["role"], []
                    )
                }
            }
        else:
            quantum_system.security_monitor.log_security_event({
                "event_id": f"AUTH_FAIL_{int(datetime.now().timestamp())}",
                "timestamp": datetime.now(timezone.utc),
                "event_type": "AUTH_FAILURE",
                "severity": "MEDIUM",
                "source_ip": client_ip,
                "user_agent": request.headers.get("User-Agent"),
                "endpoint": "/api/v1/auth/login",
                "details": {"username": username},
                "action_taken": "LOGGED"
            })
            
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
    
    @app.post("/api/v1/auth/logout", tags=["Authentication"])
    async def logout(current_user = Depends(get_current_user), db: Session = Depends(get_db)):
        """User logout"""
        quantum_system.tamper_logger.add_log_entry("USER_LOGOUT", {
            "username": current_user.get("username"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }, db)
        
        return {"message": "Successfully logged out"}
    
    @app.get("/api/v1/auth/permissions", tags=["Authentication"])
    async def get_user_permissions(current_user = Depends(get_current_user)):
        """Get current user permissions"""
        user_role = current_user.get("role", "viewer")
        permissions = quantum_system.auth_manager.user_roles.get(user_role, [])
        
        return {
            "username": current_user.get("username"),
            "role": user_role,
            "permissions": permissions
        }
    
    # ==================== FRONTEND COMPATIBILITY ENDPOINTS ====================
    # Option 1: Add endpoints that match your original frontend expectations
    
    @app.get("/api/v1/vehicles/active", tags=["Vehicle Management - Frontend Compatible"])
    async def get_active_vehicles(
        current_user = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
        """Get active vehicles in format expected by frontend"""
        try:
            # Create vehicle data from GPS records or use demo data
            vehicles = []
            
            # Try to get real GPS data first
            recent_gps = db.query(DBGPSData).filter(
                DBGPSData.timestamp > datetime.now(timezone.utc) - timedelta(hours=24)
            ).all()
            
            if recent_gps:
                # Group by device and get latest position
                device_positions = {}
                for gps in recent_gps:
                    if gps.device_id not in device_positions:
                        device_positions[gps.device_id] = gps
                    elif gps.timestamp > device_positions[gps.device_id].timestamp:
                        device_positions[gps.device_id] = gps
                
                for device_id, gps_record in device_positions.items():
                    vehicles.append({
                        "vehicle_id": device_id,
                        "lat": gps_record.latitude or 40.7128 + hash(device_id) % 100 / 10000,
                        "lng": gps_record.longitude or -74.0060 + hash(device_id) % 100 / 10000,
                        "status": "Active",
                        "last_update": gps_record.timestamp.isoformat(),
                        "security_status": "Quantum Secured"
                    })
            
            # If no GPS data, return demo vehicles
            if not vehicles:
                demo_vehicles = [
                    {
                        "vehicle_id": "QV001",
                        "lat": 40.7128,
                        "lng": -74.0060,
                        "status": "Active",
                        "last_update": datetime.now(timezone.utc).isoformat(),
                        "security_status": "Quantum Secured"
                    },
                    {
                        "vehicle_id": "QV002", 
                        "lat": 40.7589,
                        "lng": -73.9851,
                        "status": "Secure",
                        "last_update": datetime.now(timezone.utc).isoformat(),
                        "security_status": "Quantum Secured"
                    },
                    {
                        "vehicle_id": "QV003",
                        "lat": 40.6782,
                        "lng": -73.9442,
                        "status": "Transit",
                        "last_update": datetime.now(timezone.utc).isoformat(),
                        "security_status": "Quantum Secured"
                    }
                ]
                vehicles = demo_vehicles
            
            return {
                "active_vehicles": vehicles,
                "count": len(vehicles),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get active vehicles: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve vehicle data")
    
    @app.post("/api/v1/vehicles/{vehicle_id}/action", tags=["Vehicle Management - Frontend Compatible"])
    async def execute_vehicle_action(
        vehicle_id: str,
        action: str,
        current_user = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
        """Execute vehicle action (frontend compatible)"""
        if not quantum_system.auth_manager.check_permission(current_user.get("token", ""), "write"):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        try:
            # Log the action
            quantum_system.tamper_logger.add_log_entry(f"VEHICLE_ACTION_{action.upper()}", {
                "vehicle_id": vehicle_id,
                "action": action,
                "user": current_user.get("username"),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }, db)
            
            # Simulate action responses based on frontend expectations
            action_messages = {
                "track": f"Tracking initiated for vehicle {vehicle_id}",
                "lock": f"Emergency lock activated for vehicle {vehicle_id}",
                "unlock": f"Vehicle {vehicle_id} unlocked successfully"
            }
            
            message = action_messages.get(action, f"Action {action} executed for vehicle {vehicle_id}")
            
            return {
                "status": "success",
                "message": message,
                "vehicle_id": vehicle_id,
                "action": action,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Vehicle action failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to execute {action}")
    
    # ==================== SYSTEM STATUS ENDPOINTS ====================
    
    @app.get("/api/v1/system/status", tags=["System Management"])
    async def get_system_status():
        """Get system status (frontend compatible format)"""
        try:
            gps_count = 0
            db = SessionLocal()
            try:
                gps_count = db.query(DBGPSData).count()
            finally:
                db.close()
            
            return {
                "status": "operational",
                "version": "1.0.0",
                "components": {
                    "active_sessions": len(quantum_system.auth_manager.active_sessions),
                    "active_quantum_tokens": len(quantum_system.quantum_simulator.active_tokens),
                    "security_events_today": len([
                        e for e in quantum_system.security_monitor.security_events
                        if (datetime.now(timezone.utc) - e["timestamp"]).days == 0
                    ]),
                    "gps_records": gps_count
                },
                "uptime": "System Operational",
                "threats": len([
                    e for e in quantum_system.security_monitor.security_events
                    if e.get("severity", "").upper() in ["HIGH", "CRITICAL"]
                ]),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve system status")
    
    # GPS Management Endpoints
    @app.post("/api/v1/gps/store", tags=["GPS Management"])
    async def store_gps_data(
        gps_data: GPSData,
        request: Request,
        current_user = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
        """Store encrypted GPS data with quantum authentication"""
        try:
            # Generate quantum token
            quantum_token = quantum_system.quantum_simulator.generate_quantum_token(gps_data.device_id)
            
            # Encrypt GPS data
            gps_data_json = gps_data.json()
            encrypted_gps_data = quantum_system.pqc.encrypt(gps_data_json, quantum_token)
            
            # Create GPS data with encrypted payload
            encrypted_gps = GPSData(
                device_id=gps_data.device_id,
                timestamp=gps_data.timestamp,
                latitude=gps_data.latitude,
                longitude=gps_data.longitude,
                encrypted_payload=encrypted_gps_data
            )
            
            # Store encrypted GPS data
            record_id = quantum_system.gps_manager.store_gps_data(encrypted_gps, quantum_token, db)
            
            # Log the operation
            quantum_system.tamper_logger.add_log_entry("GPS_STORE_SUCCESS", {
                "record_id": record_id,
                "device_id": gps_data.device_id,
                "user": current_user.get("username"),
                "client_ip": request.client.host,
                "encrypted_data_preview": encrypted_gps_data[:50],
                "quantum_token_id": quantum_token.token_id
            }, db)
            
            return {
                "status": "success",
                "record_id": record_id,
                "quantum_token": quantum_token.dict(),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"GPS storage failed: {e}")
            raise HTTPException(status_code=500, detail="GPS storage failed")
    
    @app.get("/api/v1/gps/device/{device_id}", tags=["GPS Management"])
    async def get_gps_data(
        device_id: str,
        current_user = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
        """Retrieve GPS data for a device"""
        if not quantum_system.auth_manager.check_permission(current_user.get("token", ""), "read"):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        try:
            gps_records = quantum_system.gps_manager.get_gps_data_for_device(device_id, db)
            return {
                "device_id": device_id,
                "records_count": len(gps_records),
                "records": gps_records,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to retrieve GPS data: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve GPS data")
    
    # Quantum Token Endpoints
    @app.post("/api/v1/quantum/generate", tags=["Quantum Security"])
    async def generate_quantum_token(
        device_id: str,
        current_user = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
        """Generate quantum authentication token"""
        if not quantum_system.auth_manager.check_permission(current_user.get("token", ""), "write"):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        token = quantum_system.quantum_simulator.generate_quantum_token(device_id)
        
        quantum_system.tamper_logger.add_log_entry("QUANTUM_TOKEN_GENERATED", {
            "token_id": token.token_id,
            "device_id": device_id,
            "user": current_user.get("username")
        }, db)
        
        return {"quantum_token": token.dict()}
    
    @app.post("/api/v1/quantum/qkd-exchange", tags=["Quantum Security"])
    async def qkd_exchange(
        sender_device_id: str,
        receiver_device_id: str,
        current_user = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
        """Simulate QKD exchange between devices"""
        if not quantum_system.auth_manager.check_permission(current_user.get("token", ""), "write"):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        shared_token = quantum_system.quantum_simulator.simulate_qkd_token_exchange(
            sender_device_id, receiver_device_id
        )
        
        if shared_token:
            quantum_system.tamper_logger.add_log_entry("QKD_EXCHANGE_SUCCESS", {
                "sender_device": sender_device_id,
                "receiver_device": receiver_device_id,
                "shared_token_id": shared_token.token_id,
                "user": current_user.get("username")
            }, db)
            return {"status": "success", "shared_quantum_token": shared_token.dict()}
        else:
            quantum_system.tamper_logger.add_log_entry("QKD_EXCHANGE_FAILED", {
                "sender_device": sender_device_id,
                "receiver_device": receiver_device_id,
                "user": current_user.get("username"),
                "reason": "Simulation failed"
            }, db)
            raise HTTPException(status_code=500, detail="QKD exchange simulation failed")
    
    @app.post("/api/v1/quantum/verify", tags=["Quantum Security"])
    async def verify_quantum_token(
        token: QuantumToken,
        current_user = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
        """Verify quantum token authenticity"""
        is_valid = quantum_system.quantum_simulator.verify_quantum_token(token)
        
        quantum_system.tamper_logger.add_log_entry("QUANTUM_TOKEN_VERIFIED", {
            "token_id": token.token_id,
            "valid": is_valid,
            "user": current_user.get("username")
        }, db)
        
        return {
            "valid": is_valid,
            "token_id": token.token_id,
            "verification_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    # Logistics Management Endpoints
    @app.post("/api/v1/logistics/seal-event", tags=["Logistics Management"])
    async def record_seal_event(
        seal_event: SealEvent,
        request: Request,
        current_user = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
        """Record vehicle seal event"""
        if not quantum_system.auth_manager.check_permission(current_user.get("token", ""), "write"):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        # Log the event
        event_details = {
            "vehicle_id": seal_event.vehicle_id,
            "event_type": seal_event.event_type,
            "location": seal_event.location,
            "user": current_user.get("username"),
            "client_ip": request.client.host,
            "auth_token_used": seal_event.user_token
        }
        
        quantum_system.tamper_logger.add_log_entry(
            f"SEAL_EVENT_{seal_event.event_type.upper()}", 
            event_details, 
            db
        )
        
        # Check for token mismatch (potential tampering)
        if seal_event.user_token != current_user.get("token"):
            logger.warning(f"Token mismatch for vehicle {seal_event.vehicle_id}")
            quantum_system.tamper_logger.add_log_entry("POTENTIAL_TAMPERING_ALERT", {
                "vehicle_id": seal_event.vehicle_id,
                "event_type": seal_event.event_type,
                "location": seal_event.location,
                "user_performing_action": current_user.get("username"),
                "token_presented_at_vehicle": seal_event.user_token,
                "mismatched_auth_token": True
            }, db)
            raise HTTPException(
                status_code=403, 
                detail="Token mismatch detected. Potential tampering."
            )
        
        return {
            "status": "success",
            "message": f"Seal {seal_event.event_type} event recorded for vehicle {seal_event.vehicle_id}",
            "event_details": event_details,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    @app.get("/api/v1/logistics/track-vehicle/{vehicle_id}", tags=["Logistics Management"])
    async def track_vehicle(
        vehicle_id: str,
        current_user = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
        """Track vehicle location history"""
        if not quantum_system.auth_manager.check_permission(current_user.get("token", ""), "read"):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        try:
            gps_records = quantum_system.gps_manager.get_gps_data_for_device(vehicle_id, db)
            
            if not gps_records:
                # Generate demo tracking data if no real data
                demo_points = quantum_system.gps_simulator.generate_gps_data(vehicle_id, 10)
                processed_records = []
                for point in demo_points:
                    processed_records.append({
                        "timestamp": point["timestamp"].isoformat(),
                        "latitude": point["latitude"],
                        "longitude": point["longitude"],
                        "status": "DEMO",
                        "quantum_secured": True
                    })
            else:
                # Process records for decryption simulation
                processed_records = []
                for record in gps_records:
                    if record.get("encrypted_payload"):
                        decrypted_info = {
                            "device_id": record["device_id"],
                            "timestamp": record["timestamp"].isoformat() if hasattr(record["timestamp"], 'isoformat') else str(record["timestamp"]),
                            "latitude": record.get("latitude"),
                            "longitude": record.get("longitude"),
                            "status": "ENCRYPTED",
                            "quantum_token_id": record.get("quantum_token_id"),
                            "location_encrypted": True,
                            "quantum_secured": True
                        }
                    else:
                        decrypted_info = record
                    
                    processed_records.append(decrypted_info)
            
            return {
                "vehicle_id": vehicle_id,
                "gps_history": processed_records,
                "records_count": len(processed_records),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to track vehicle {vehicle_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve vehicle data")
    
    # System Integrity Endpoints
    @app.get("/api/v1/logs/verify-integrity", tags=["System Integrity"])
    async def verify_log_integrity(
        current_user = Depends(get_current_user), 
        db: Session = Depends(get_db)
    ):
        """Verify tamper-evident log chain integrity"""
        if not quantum_system.auth_manager.check_permission(current_user.get("token", ""), "audit"):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        integrity_status = quantum_system.tamper_logger.verify_chain_integrity(db)
        
        return {
            "integrity_verified": integrity_status,
            "verification_timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "SECURE" if integrity_status else "COMPROMISED"
        }
    
    @app.get("/api/v1/logs/recent", tags=["System Logs"])
    async def get_recent_logs(
        limit: int = 50,
        event_type: Optional[str] = None,
        current_user = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
        """Get recent tamper-evident log entries"""
        if not quantum_system.auth_manager.check_permission(current_user.get("token", ""), "read"):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        try:
            query = db.query(DBLogEntry).order_by(DBLogEntry.id.desc())
            
            if event_type:
                query = query.filter(DBLogEntry.event == event_type)
            
            recent_logs_db = query.limit(limit).all()
            
            # Convert to response format
            recent_logs = []
            for log_entry in recent_logs_db:
                log_dict = {
                    "id": log_entry.id,
                    "event": log_entry.event,
                    "timestamp": log_entry.timestamp,
                    "hash_prev": log_entry.hash_prev,
                    "hash_current": log_entry.hash_current
                }
                
                # Parse details JSON
                try:
                    log_dict["details"] = json.loads(log_entry.details)
                except (json.JSONDecodeError, TypeError):
                    log_dict["details"] = log_entry.details
                
                recent_logs.append(log_dict)
            
            return {
                "message": f"Retrieved {len(recent_logs)} recent log entries",
                "event_type_filter": event_type,
                "logs": recent_logs,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to retrieve logs: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve logs")
    
    # Security Management Endpoints
    @app.get("/api/v1/security/dashboard", tags=["Security Management"])
    async def security_dashboard(current_user = Depends(get_current_user)):
        """Get security monitoring dashboard"""
        if not quantum_system.auth_manager.check_permission(current_user.get("token", ""), "audit"):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        # Get recent security events
        recent_events = quantum_system.security_monitor.security_events[-10:]
        
        # Calculate statistics
        total_events = len(quantum_system.security_monitor.security_events)
        blocked_ips_count = len(quantum_system.security_monitor.blocked_ips)
        
        # Get event type breakdown
        event_types = {}
        for event in quantum_system.security_monitor.security_events:
            event_type = event.get("event_type", "UNKNOWN")
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        return {
            "dashboard": {
                "total_security_events": total_events,
                "blocked_ips_count": blocked_ips_count,
                "active_sessions": len(quantum_system.auth_manager.active_sessions),
                "recent_events": recent_events,
                "event_type_breakdown": event_types,
                "system_status": "OPERATIONAL",
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
        }
    
    @app.get("/api/v1/security/events", tags=["Security Management"])
    async def get_security_events(
        hours: int = 24,
        severity: Optional[str] = None,
        current_user = Depends(get_current_user)
    ):
        """Get security events"""
        if not quantum_system.auth_manager.check_permission(current_user.get("token", ""), "audit"):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        # Filter events by time window
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        filtered_events = [
            event for event in quantum_system.security_monitor.security_events
            if event["timestamp"] > cutoff
        ]
        
        # Filter by severity if specified
        if severity:
            filtered_events = [
                event for event in filtered_events
                if event.get("severity") == severity
            ]
        
        # Add demo events if none exist
        if not filtered_events:
            filtered_events = [{
                "event_id": f"demo_{int(datetime.now().timestamp())}",
                "timestamp": datetime.now(timezone.utc),
                "event_type": "SYSTEM_STATUS",
                "severity": "INFO",
                "details": {"reason": "System operational - no recent security events"},
                "action_taken": "LOGGED"
            }]
        
        return {
            "events": filtered_events,
            "count": len(filtered_events),
            "time_window_hours": hours,
            "severity_filter": severity,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    @app.post("/api/v1/security/block-ip", tags=["Security Management"])
    async def block_ip(
        ip_address: str,
        reason: str,
        current_user = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
        """Block IP address"""
        if not quantum_system.auth_manager.check_permission(current_user.get("token", ""), "admin"):
            raise HTTPException(status_code=403, detail="Admin permissions required")
        
        quantum_system.security_monitor.block_ip(ip_address, reason)
        
        quantum_system.tamper_logger.add_log_entry("IP_BLOCKED", {
            "blocked_ip": ip_address,
            "reason": reason,
            "admin_user": current_user.get("username"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }, db)
        
        return {
            "status": "success",
            "message": f"IP address {ip_address} has been blocked",
            "reason": reason,
            "blocked_by": current_user.get("username")
        }
    
    @app.get("/api/v1/system/health", tags=["System Management"])
    async def comprehensive_health_check(db: Session = Depends(get_db)):
        """Comprehensive system health check"""
        try:
            health_status = {
                "overall_status": "HEALTHY",
                "components": {},
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Test tamper logging integrity
            try:
                integrity_ok = quantum_system.tamper_logger.verify_chain_integrity(db)
                health_status["components"]["tamper_logging"] = "HEALTHY" if integrity_ok else "COMPROMISED"
            except Exception as e:
                health_status["components"]["tamper_logging"] = f"ERROR: {str(e)}"
            
            # Test quantum simulator
            try:
                test_token = quantum_system.quantum_simulator.generate_quantum_token("health_check")
                is_valid = quantum_system.quantum_simulator.verify_quantum_token(test_token)
                health_status["components"]["quantum_simulation"] = "HEALTHY" if is_valid else "ERROR"
            except Exception as e:
                health_status["components"]["quantum_simulation"] = f"ERROR: {str(e)}"
            
            # Test database connectivity
            try:
                db.execute("SELECT 1")
                health_status["components"]["database"] = "HEALTHY"
            except Exception as e:
                health_status["components"]["database"] = f"ERROR: {str(e)}"
            
            # Test authentication system
            try:
                active_sessions = len(quantum_system.auth_manager.active_sessions)
                health_status["components"]["authentication"] = f"HEALTHY ({active_sessions} active sessions)"
            except Exception as e:
                health_status["components"]["authentication"] = f"ERROR: {str(e)}"
            
            # Test security monitoring
            try:
                event_count = len(quantum_system.security_monitor.security_events)
                health_status["components"]["security_monitoring"] = f"HEALTHY ({event_count} events tracked)"
            except Exception as e:
                health_status["components"]["security_monitoring"] = f"ERROR: {str(e)}"
            
            # Determine overall status
            error_count = sum(1 for status in health_status["components"].values() if "ERROR" in str(status))
            compromised_count = sum(1 for status in health_status["components"].values() if "COMPROMISED" in str(status))
            
            if compromised_count > 0:
                health_status["overall_status"] = "CRITICAL"
            elif error_count > 0:
                health_status["overall_status"] = "DEGRADED"
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "overall_status": "CRITICAL",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    # Add UI serving endpoint
    @app.get("/ui")
    async def serve_ui():
        """Serve the frontend UI"""
        # This would serve your HTML file - adjust path as needed
        try:
            return FileResponse("quantum_frontend.html")
        except:
            return {"message": "UI file not found. Please place quantum_interface.html in the same directory as this script."}

    return app

# Production deployment configuration
def run_production_server():
    """Run the production server with optimal configuration"""
    
    app = create_production_app()
    
    logger.info("Starting Complete Quantum Cybersecurity Backend")
    logger.info("=" * 80)
    logger.info("Available at: http://localhost:8000")
    logger.info("API Documentation: http://localhost:8000/api/docs")
    logger.info("Frontend UI: http://localhost:8000/ui")
    logger.info("Test credentials: admin/admin123, operator/operator123")
    
    # Production server configuration
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker for SQLite compatibility
        log_level="info",
        access_log=True,
        use_colors=True,
        server_header=False,
        date_header=False
    )

# Development server
def run_development_server():
    """Run development server with hot reload"""
    app = create_production_app()
    
    logger.info("Starting Quantum Cybersecurity Backend - Development Mode")
    logger.info("Available at: http://localhost:8000")
    logger.info("API Documentation: http://localhost:8000/api/docs")
    logger.info("Frontend UI: http://localhost:8000/ui")
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,  # Or any other available port
        reload=True,
        log_level="debug"
    )

if __name__ == "__main__":
    run_production_server()  # Use production server instead
    
    if len(sys.argv) > 1 and sys.argv[1] == "production":
        run_production_server()
    else:
        run_development_server()
    logger.info("Features Integrated:")
    logger.info("  ✓ Core Infrastructure & Setup")
    logger.info("  ✓ GPS Data Encryption & Storage")
    logger.info("  ✓ Quantum Token Simulation")
    logger.info("  ✓ Tamper-Evident Logging")
    logger.info("  ✓ Authentication & Authorization")
    logger.info("  ✓ Security Monitoring & Rate Limiting")
    logger.info("  ✓ Logistics Management")
    logger.info("  ✓ Frontend Compatibility Endpoints")
    logger.info("  ✓ GPS Simulation System")
    logger.info("  ✓ Demo Data Generation")
    logger.info("=" * 80)