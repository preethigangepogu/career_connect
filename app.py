# CareerConnect - Backend Server (Fixed & Enhanced)

from flask import Flask, request, jsonify, send_from_directory
import sqlite3, hashlib, os, joblib, numpy as np, base64

app = Flask(__name__, static_folder="../Frontend")

@app.after_request
def cors(res):
    res.headers["Access-Control-Allow-Origin"]  = "*"
    res.headers["Access-Control-Allow-Headers"] = "Content-Type"
    res.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
    return res

# ── Load Scoring Model ─────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.join(BASE, "../ML")

scoring_model  = joblib.load(os.path.join(ML_DIR, "rf_model.pkl"))
edu_enc        = joblib.load(os.path.join(ML_DIR, "edu_encoder.pkl"))
result_enc     = joblib.load(os.path.join(ML_DIR, "label_encoder.pkl"))

# ── Database ───────────────────────────────────────────────────────
DB = os.path.join(BASE, "careerconnect.db")

def db():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c

def setup():
    conn = db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT NOT NULL,
            email      TEXT UNIQUE NOT NULL,
            password   TEXT NOT NULL,
            role       TEXT NOT NULL,
            company    TEXT DEFAULT '',
            skills     TEXT DEFAULT '',
            experience INTEGER DEFAULT 0,
            education  TEXT DEFAULT 'Bachelor''s',
            phone      TEXT DEFAULT '',
            bio        TEXT DEFAULT '',
            linkedin   TEXT DEFAULT '',
            resume_name TEXT DEFAULT '',
            resume_data TEXT DEFAULT '',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS jobs (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            recruiter_id        INTEGER,
            title               TEXT NOT NULL,
            company             TEXT NOT NULL,
            description         TEXT DEFAULT '',
            skills              TEXT DEFAULT '',
            experience_required INTEGER DEFAULT 0,
            salary              TEXT DEFAULT '',
            location            TEXT DEFAULT '',
            job_type            TEXT DEFAULT 'Full-time',
            status              TEXT DEFAULT 'Open',
            created_at          TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS applications (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id       INTEGER,
            candidate_id INTEGER,
            cover_letter TEXT DEFAULT '',
            status       TEXT DEFAULT 'Applied',
            match_label  TEXT DEFAULT '',
            match_score  REAL DEFAULT 0,
            applied_at   TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS notifications (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER,
            message      TEXT,
            is_read      INTEGER DEFAULT 0,
            created_at   TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)
    # Add new columns to existing users table if upgrading
    try:
        conn.execute("ALTER TABLE users ADD COLUMN phone TEXT DEFAULT ''")
    except: pass
    try:
        conn.execute("ALTER TABLE users ADD COLUMN bio TEXT DEFAULT ''")
    except: pass
    try:
        conn.execute("ALTER TABLE users ADD COLUMN linkedin TEXT DEFAULT ''")
    except: pass
    try:
        conn.execute("ALTER TABLE users ADD COLUMN resume_name TEXT DEFAULT ''")
    except: pass
    try:
        conn.execute("ALTER TABLE users ADD COLUMN resume_data TEXT DEFAULT ''")
    except: pass
    conn.commit(); conn.close()

setup()

def hpw(p): return hashlib.sha256(p.encode()).hexdigest()

def add_notification(conn, user_id, message):
    conn.execute("INSERT INTO notifications (user_id, message) VALUES (?,?)", (user_id, message))

# ── Compatibility Scoring ─────────────────────────────────────────
def score_candidate(candidate, job):
    try:
        edu = candidate.get("education","Bachelor's")
        if edu not in list(edu_enc.classes_): edu = "Bachelor's"
        edu_encoded = edu_enc.transform([edu])[0]

        c_skills = set(s.strip().lower() for s in str(candidate.get("skills","")).split(",") if s.strip())
        j_skills = set(s.strip().lower() for s in str(job.get("skills","")).split(",") if s.strip())
        skills_score = int(len(c_skills & j_skills) / len(j_skills) * 100) if j_skills else 50

        exp     = int(candidate.get("experience") or 0)
        exp_req = int(job.get("experience_required") or 0)
        relevance  = min(100, int((exp / max(exp_req,1)) * 70) + skills_score // 3)
        history    = min(100, exp * 8 + 20)

        X   = np.array([[skills_score, exp, edu_encoded, relevance, history]])
        pred = scoring_model.predict(X)[0]
        prob = scoring_model.predict_proba(X)[0]
        conf  = round(float(max(prob)) * 100, 1)

        # FIX: Derive label from skills_score directly so label matches the score
        # The ML model gives a label, but we override with a human-readable one
        # that actually aligns with the compatibility percentage shown to users.
        # skills_score is the primary driver of "fit" for a role.
        if skills_score >= 70:
            label = "Strong Match"
        elif skills_score >= 40:
            label = "Moderate Match"
        else:
            label = "Weak Match"

        # The "score" shown to users is skills_score (actual skill overlap %)
        # conf is the ML model confidence — we use skills_score as the user-facing score
        return label, skills_score, skills_score
    except Exception as e:
        print("Scoring error:", e)
        return "Moderate Match", 60, 50

# ── Description / Summary Generators ──────────────────────────────
def build_job_description(title, skills, experience, salary, location, job_type):
    slist = [s.strip() for s in str(skills).split(",") if s.strip()]
    bullets = "\n".join(["  - " + s for s in slist[:5]])
    return (
        "Job Title: {t}\n"
        "Location: {loc}  |  Type: {jt}  |  Salary: {sal}\n\n"
        "About the Role:\n"
        "We are looking for a skilled {t} to join our growing team.\n\n"
        "Key Responsibilities:\n"
        "  - Design, develop, and maintain {t} solutions\n"
        "  - Collaborate with team members in planning and code reviews\n"
        "  - Write clean, well-structured, and documented code\n"
        "  - Troubleshoot and resolve issues in existing systems\n\n"
        "Requirements:\n"
        "  - {exp}+ years of relevant experience\n"
        "  - Proficiency in {skills_inline}\n"
        "  - Strong problem-solving and communication skills\n"
        "  - Bachelor's degree in Computer Science or related field (preferred)\n\n"
        "Technical Skills:\n{bullets}\n\n"
        "What We Offer:\n"
        "  - Competitive salary: {sal}\n"
        "  - Flexible working hours\n"
        "  - Health and wellness benefits\n"
        "  - Learning and growth opportunities"
    ).format(
        t=title, loc=location or "Hyderabad / Remote",
        jt=job_type, sal=salary or "Competitive",
        exp=experience, skills_inline=", ".join(slist[:3]) or "relevant tools",
        bullets=bullets or "  - Relevant technical skills"
    )

def build_candidate_summary(name, skills, experience, education, job_title, match_score, match_label):
    slist = [s.strip() for s in str(skills or "").split(",") if s.strip()]
    top   = ", ".join(slist[:3]) if slist else "various technical areas"
    exp   = int(experience or 0)
    level = "entry-level" if exp < 2 else ("mid-level" if exp < 5 else "senior-level")
    if match_label == "Strong Match":
        note   = "meets or exceeds most requirements for this position"
        action = "Recommended for the next interview round."
    elif match_label == "Moderate Match":
        note   = "shows reasonable potential but may need further evaluation"
        action = "Consider a brief screening call before proceeding."
    else:
        note   = "does not fully meet the core requirements at this time"
        action = "May be considered for a different or junior role."
    skills_text = "\n".join(["  - " + s for s in slist]) if slist else "  - Not specified"
    return (
        "Candidate Summary Report\n"
        "==========================================\n"
        "Name        : {name}\n"
        "Education   : {edu}\n"
        "Experience  : {exp} year(s)\n"
        "Role Applied: {role}\n"
        "Result      : {label} ({score}%)\n\n"
        "Summary:\n"
        "{name} is a {level} professional with {exp} year(s) of experience. "
        "Their key strengths include {top}. Based on the compatibility evaluation, "
        "this candidate {note}.\n\n"
        "Skills Listed:\n{skills_text}\n\n"
        "Recommendation:\n{action}\n"
        "==========================================\n"
        "Generated by CareerConnect"
    ).format(
        name=name, edu=education, exp=exp,
        role=job_title, label=match_label, score=match_score,
        level=level, top=top, note=note,
        skills_text=skills_text, action=action
    )

# ── Routes: Auth ──────────────────────────────────────────────────
@app.route("/api/register", methods=["POST"])
def register():
    d = request.json
    if not all([d.get("name"), d.get("email"), d.get("password"), d.get("role")]):
        return jsonify({"error": "All fields are required"}), 400
    try:
        conn = db()
        conn.execute(
            "INSERT INTO users (name,email,password,role,company,skills,experience,education) VALUES(?,?,?,?,?,?,?,?)",
            (d["name"], d["email"], hpw(d["password"]), d["role"],
             d.get("company",""), d.get("skills",""), d.get("experience",0), d.get("education","Bachelor's"))
        )
        conn.commit(); conn.close()
        return jsonify({"message": "Account created"}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "This email is already registered"}), 409

@app.route("/api/login", methods=["POST"])
def login():
    d    = request.json
    conn = db()
    u    = conn.execute("SELECT * FROM users WHERE email=? AND password=?", (d.get("email"), hpw(d.get("password","")))).fetchone()
    conn.close()
    if not u: return jsonify({"error": "Incorrect email or password"}), 401
    return jsonify({"id":u["id"],"name":u["name"],"email":u["email"],"role":u["role"],
                    "company":u["company"],"skills":u["skills"],"experience":u["experience"],
                    "education":u["education"],"phone":u["phone"],"bio":u["bio"],"linkedin":u["linkedin"],
                    "resume_name":u["resume_name"]})

@app.route("/api/profile/<int:uid>", methods=["GET"])
def get_profile(uid):
    conn = db()
    u = conn.execute("SELECT id,name,email,role,company,skills,experience,education,phone,bio,linkedin,resume_name FROM users WHERE id=?", (uid,)).fetchone()
    conn.close()
    return (jsonify(dict(u)) if u else (jsonify({"error":"Not found"}), 404))

@app.route("/api/profile/<int:uid>", methods=["PUT"])
def update_profile(uid):
    d = request.json; conn = db()
    conn.execute("UPDATE users SET name=?,skills=?,experience=?,education=?,company=?,phone=?,bio=?,linkedin=? WHERE id=?",
        (d.get("name"),d.get("skills"),d.get("experience",0),d.get("education"),
         d.get("company"),d.get("phone",""),d.get("bio",""),d.get("linkedin",""),uid))
    conn.commit(); conn.close()
    return jsonify({"message":"Profile updated"})

@app.route("/api/profile/<int:uid>/resume", methods=["POST"])
def upload_resume(uid):
    d = request.json
    resume_name = d.get("resume_name","")
    resume_data = d.get("resume_data","")  # base64
    conn = db()
    conn.execute("UPDATE users SET resume_name=?,resume_data=? WHERE id=?", (resume_name, resume_data, uid))
    conn.commit(); conn.close()
    return jsonify({"message":"Resume uploaded"})

@app.route("/api/profile/<int:uid>/resume", methods=["GET"])
def get_resume(uid):
    conn = db()
    u = conn.execute("SELECT resume_name,resume_data FROM users WHERE id=?", (uid,)).fetchone()
    conn.close()
    if not u or not u["resume_data"]:
        return jsonify({"error":"No resume"}), 404
    return jsonify({"resume_name":u["resume_name"],"resume_data":u["resume_data"]})

# ── Routes: Jobs ──────────────────────────────────────────────────
@app.route("/api/jobs", methods=["GET"])
def get_jobs():
    conn = db()
    jobs = conn.execute("SELECT j.*,u.name recruiter_name FROM jobs j JOIN users u ON j.recruiter_id=u.id WHERE j.status='Open' ORDER BY j.created_at DESC").fetchall()
    conn.close(); return jsonify([dict(j) for j in jobs])

@app.route("/api/jobs/<int:jid>", methods=["GET"])
def get_job(jid):
    conn = db()
    j = conn.execute("SELECT * FROM jobs WHERE id=?",(jid,)).fetchone()
    conn.close()
    return jsonify(dict(j)) if j else (jsonify({"error":"Not found"}), 404)

@app.route("/api/jobs", methods=["POST"])
def create_job():
    d = request.json; conn = db()
    cur = conn.execute(
        "INSERT INTO jobs (recruiter_id,title,company,description,skills,experience_required,salary,location,job_type) VALUES(?,?,?,?,?,?,?,?,?)",
        (d["recruiter_id"],d["title"],d.get("company",""),d.get("description",""),
         d.get("skills",""),d.get("experience_required",0),d.get("salary",""),d.get("location",""),d.get("job_type","Full-time"))
    )
    conn.commit(); jid = cur.lastrowid; conn.close()
    return jsonify({"message":"Job posted","job_id":jid}), 201

@app.route("/api/jobs/recruiter/<int:rid>", methods=["GET"])
def recruiter_jobs(rid):
    conn = db()
    jobs = conn.execute("SELECT * FROM jobs WHERE recruiter_id=? ORDER BY created_at DESC",(rid,)).fetchall()
    conn.close(); return jsonify([dict(j) for j in jobs])

# ── Routes: Applications ──────────────────────────────────────────
@app.route("/api/apply", methods=["POST"])
def apply():
    d    = request.json
    jid  = d["job_id"]
    cid  = d["candidate_id"]
    conn = db()

    if conn.execute("SELECT id FROM applications WHERE job_id=? AND candidate_id=?",(jid,cid)).fetchone():
        conn.close(); return jsonify({"error":"You already applied to this job"}), 409

    candidate = conn.execute("SELECT * FROM users WHERE id=?",(cid,)).fetchone()
    job       = conn.execute("SELECT * FROM jobs  WHERE id=?",(jid,)).fetchone()
    label, score, _ = score_candidate(dict(candidate), dict(job))

    conn.execute("INSERT INTO applications (job_id,candidate_id,cover_letter,match_label,match_score) VALUES(?,?,?,?,?)",
        (jid, cid, d.get("cover_letter",""), label, score))

    # Notify candidate
    add_notification(conn, cid, f"Your application for '{job['title']}' was submitted successfully! Your match: {label} ({score}%)")
    conn.commit(); conn.close()
    return jsonify({"message":"Application submitted","match_label":label,"match_score":score}), 201

@app.route("/api/applications/candidate/<int:cid>", methods=["GET"])
def cand_apps(cid):
    conn = db()
    apps = conn.execute("""SELECT a.*,j.title,j.company,j.location,j.salary,j.job_type
        FROM applications a JOIN jobs j ON a.job_id=j.id
        WHERE a.candidate_id=? ORDER BY a.applied_at DESC""",(cid,)).fetchall()
    conn.close(); return jsonify([dict(a) for a in apps])

@app.route("/api/applications/job/<int:jid>", methods=["GET"])
def job_apps(jid):
    conn = db()
    apps = conn.execute("""SELECT a.*,u.name,u.email,u.skills,u.experience,u.education,u.resume_name
        FROM applications a JOIN users u ON a.candidate_id=u.id
        WHERE a.job_id=? ORDER BY a.match_score DESC""",(jid,)).fetchall()
    conn.close(); return jsonify([dict(a) for a in apps])

@app.route("/api/applications/<int:aid>/status", methods=["PUT"])
def update_status(aid):
    d = request.json; conn = db()
    conn.execute("UPDATE applications SET status=? WHERE id=?",(d["status"],aid))
    # Notify the candidate
    app_row = conn.execute("""SELECT a.candidate_id, j.title FROM applications a 
        JOIN jobs j ON a.job_id=j.id WHERE a.id=?""",(aid,)).fetchone()
    if app_row:
        status = d["status"]
        if status == "Hired":
            msg = f"🎉 Congratulations! You have been HIRED for '{app_row['title']}'! Check your email for next steps."
        elif status == "Rejected":
            msg = f"Update on your application for '{app_row['title']}': Unfortunately, you were not selected. Keep applying!"
        elif status == "Shortlisted":
            msg = f"Great news! You have been SHORTLISTED for '{app_row['title']}'. Expect to hear more soon!"
        else:
            msg = f"Your application status for '{app_row['title']}' was updated to: {status}"
        add_notification(conn, app_row["candidate_id"], msg)
    conn.commit(); conn.close()
    return jsonify({"message":"Updated"})

@app.route("/api/recruiter/dashboard/<int:rid>", methods=["GET"])
def dashboard(rid):
    conn = db()
    tj = conn.execute("SELECT COUNT(*) FROM jobs WHERE recruiter_id=?",(rid,)).fetchone()[0]
    ta = conn.execute("SELECT COUNT(*) FROM applications a JOIN jobs j ON a.job_id=j.id WHERE j.recruiter_id=?",(rid,)).fetchone()[0]
    sm = conn.execute("SELECT COUNT(*) FROM applications a JOIN jobs j ON a.job_id=j.id WHERE j.recruiter_id=? AND a.match_label='Strong Match'",(rid,)).fetchone()[0]
    conn.close()
    return jsonify({"total_jobs":tj,"total_applications":ta,"strong_matches":sm})

# NEW: Get ALL recent applications across all recruiter jobs
@app.route("/api/recruiter/recent-applications/<int:rid>", methods=["GET"])
def recent_apps(rid):
    conn = db()
    apps = conn.execute("""
        SELECT a.*, u.name, u.email, j.title as job_title
        FROM applications a
        JOIN jobs j ON a.job_id=j.id
        JOIN users u ON a.candidate_id=u.id
        WHERE j.recruiter_id=?
        ORDER BY a.applied_at DESC
        LIMIT 50
    """, (rid,)).fetchall()
    conn.close()
    return jsonify([dict(a) for a in apps])

# ── Routes: Notifications ─────────────────────────────────────────
@app.route("/api/notifications/<int:uid>", methods=["GET"])
def get_notifications(uid):
    conn = db()
    notifs = conn.execute("SELECT * FROM notifications WHERE user_id=? ORDER BY created_at DESC LIMIT 20",(uid,)).fetchall()
    conn.close()
    return jsonify([dict(n) for n in notifs])

@app.route("/api/notifications/<int:uid>/read", methods=["PUT"])
def mark_read(uid):
    conn = db()
    conn.execute("UPDATE notifications SET is_read=1 WHERE user_id=?",(uid,))
    conn.commit(); conn.close()
    return jsonify({"message":"Marked as read"})

@app.route("/api/notifications/<int:nid>/delete", methods=["DELETE"])
def delete_notification(nid):
    conn = db()
    conn.execute("DELETE FROM notifications WHERE id=?",(nid,))
    conn.commit(); conn.close()
    return jsonify({"message":"Deleted"})

# ── Routes: Tools ─────────────────────────────────────────────────
@app.route("/api/genai/job-description", methods=["POST"])
def gen_desc():
    d = request.json
    try:
        result = build_job_description(
            d.get("title","Software Developer"), d.get("skills",""),
            d.get("experience","1"), d.get("salary",""),
            d.get("location",""), d.get("job_type","Full-time")
        )
        return jsonify({"description": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/genai/candidate-summary", methods=["POST"])
def gen_summary():
    d = request.json
    try:
        result = build_candidate_summary(
            d.get("name","Candidate"), d.get("skills",""),
            d.get("experience",0), d.get("education","Bachelor's"),
            d.get("job_title","this role"), d.get("match_score",60),
            d.get("match_label","Moderate Match")
        )
        return jsonify({"summary": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Serve Frontend ────────────────────────────────────────────────
@app.route("/")
def home(): return send_from_directory("../Frontend","index.html")

@app.route("/css/<path:f>")
def css(f): return send_from_directory("../Frontend/css", f)

@app.route("/js/<path:f>")
def js(f):  return send_from_directory("../Frontend/js", f)

@app.route("/pages/<path:f>")
def pages(f): return send_from_directory("../Frontend/pages", f)

import os
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
