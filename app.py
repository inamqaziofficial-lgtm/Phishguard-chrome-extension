from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import whois
import dns.resolver
import ssl
import socket
import math
import tldextract
from datetime import datetime
from urllib.parse import urlparse

app = FastAPI()

# ---------------------------
# CORS
# ---------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

URL_BLEND = 0.3

# ---------------------------
# Global Model Variables
# ---------------------------

url_model = None
email_model = None
coord_model = None
url_vec = None
email_vec = None

# ---------------------------
# Load Models on Startup
# ---------------------------

@app.on_event("startup")
def load_models():
    global url_model, email_model, coord_model, url_vec, email_vec
    url_model = joblib.load("url_agent.pkl")
    email_model = joblib.load("email_agent.pkl")
    coord_model = joblib.load("coordinator_agent.pkl")
    url_vec = joblib.load("url_vectorizer.pkl")
    email_vec = joblib.load("email_vectorizer.pkl")


# ---------------------------
# Request Schemas
# ---------------------------

class URLRequest(BaseModel):
    url: str

class EmailRequest(BaseModel):
    content: str

class CombinedRequest(BaseModel):
    url: str
    content: str


# ---------------------------
# DOMAIN RULE AGENT
# ---------------------------

def shannon_entropy(s):
    if not s:
        return 0.0
    prob = [float(s.count(c)) / len(s) for c in set(s)]
    return -sum([p * math.log2(p) for p in prob])

def extract_domain_info(url):
    parsed = urlparse(url)
    domain = parsed.netloc or url
    ext = tldextract.extract(domain)
    registered = ext.registered_domain

    info = {}
    now = datetime.utcnow()

    try:
        w = whois.whois(registered)
        created = w.creation_date
        if isinstance(created, list):
            created = created[0]
        if isinstance(created, datetime):
            info["age"] = (now - created).days
        else:
            info["age"] = None
    except:
        info["age"] = None

    try:
        dns.resolver.resolve(registered, 'A')
        info["dns"] = True
    except:
        info["dns"] = False

    try:
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(socket.socket(), server_hostname=registered) as s:
            s.settimeout(2)
            s.connect((registered, 443))
        info["ssl"] = True
    except:
        info["ssl"] = False

    label = registered.split(".")[0]
    info["entropy"] = shannon_entropy(label)

    return info

def rule_score(info):
    score = 0
    total = 4

    if info["age"] is not None and info["age"] < 30:
        score += 1
    if not info["dns"]:
        score += 1
    if not info["ssl"]:
        score += 1
    if info["entropy"] > 3.5:
        score += 1

    return score / total

def risk_level(prob):
    if prob < 0.3:
        return "LOW"
    elif prob < 0.6:
        return "MEDIUM"
    return "HIGH"


# ---------------------------
# URL ONLY (WITH BLENDING)
# ---------------------------

@app.post("/scan_url")
def scan_url(data: URLRequest):

    url_ml = float(
        url_model.predict_proba(
            url_vec.transform([data.url])
        )[0][1]
    )

    info = extract_domain_info(data.url)
    domain_rule = rule_score(info)

    blended = (URL_BLEND * url_ml) + ((1 - URL_BLEND) * domain_rule)

    return {
        "ml_score": url_ml,
        "domain_score": domain_rule,
        "blended_score": blended,
        "risk": risk_level(blended),
        "phishing": bool(blended >= 0.5)
    }


# ---------------------------
# EMAIL ONLY
# ---------------------------

@app.post("/scan_email")
def scan_email(data: EmailRequest):

    email_prob = float(
        email_model.predict_proba(
            email_vec.transform([data.content])
        )[0][1]
    )

    return {
        "probability": email_prob,
        "risk": risk_level(email_prob),
        "phishing": bool(email_prob >= 0.5)
    }


# ---------------------------
# COORDINATOR (NO BLENDING HERE)
# ---------------------------

@app.post("/scan_combined")
def scan_combined(data: CombinedRequest):

    # URL ML ONLY (consistent with training)
    url_ml = float(
        url_model.predict_proba(
            url_vec.transform([data.url])
        )[0][1]
    )

    # Email ML
    email_prob = float(
        email_model.predict_proba(
            email_vec.transform([data.content])
        )[0][1]
    )

    # Coordinator
    X_meta = np.array([[url_ml, email_prob, 1, 1]])
    final_prob = float(coord_model.predict_proba(X_meta)[0][1])

    return {
        "final_probability": final_prob,
        "risk": risk_level(final_prob),
        "phishing": bool(final_prob >= 0.5)
    }