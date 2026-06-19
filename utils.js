// CareerConnect - Shared Utilities
var API = "/api";

function getUser() {
    var u = localStorage.getItem("cc_user");
    return u ? JSON.parse(u) : null;
}

function saveUser(user) {
    localStorage.setItem("cc_user", JSON.stringify(user));
}

function logout() {
    localStorage.removeItem("cc_user");
    window.location.href = "/";
}

function requireLogin(role) {
    var user = getUser();
    if (!user) { window.location.href = "/pages/login.html"; return null; }
    if (role && user.role !== role) { window.location.href = "/"; return null; }
    return user;
}

async function api(path, method, body) {
    var opts = { method: method || "GET", headers: { "Content-Type": "application/json" } };
    if (body) opts.body = JSON.stringify(body);
    var res = await fetch(API + path, opts);
    var data = await res.json();
    if (!res.ok) throw new Error(data.error || "Request failed");
    return data;
}

function showAlert(id, msg, type) {
    var el = document.getElementById(id);
    if (!el) return;
    el.innerHTML = '<div class="alert alert-' + (type||"info") + '">' + msg + '</div>';
    setTimeout(function() { if(el) el.innerHTML = ""; }, 4000);
}

function fmtDate(d) {
    if (!d) return "-";
    return new Date(d).toLocaleDateString("en-IN", { day:"2-digit", month:"short", year:"numeric" });
}

function skillTags(s) {
    if (!s) return "";
    return s.split(",").filter(function(x){ return x.trim(); }).map(function(x) {
        return '<span class="tag">' + x.trim() + '</span>';
    }).join("");
}

function matchBadge(label) {
    var map = { "Strong Match": "badge-strong", "Moderate Match": "badge-moderate", "Weak Match": "badge-weak" };
    return '<span class="badge ' + (map[label]||"badge-moderate") + '">' + (label||"-") + '</span>';
}

function statusBadge(s) {
    var map = { "Applied":"badge-applied","Shortlisted":"badge-shortlisted","Rejected":"badge-rejected","Hired":"badge-hired","Open":"badge-open" };
    return '<span class="badge ' + (map[s]||"badge-applied") + '">' + s + '</span>';
}

function scoreBar(n) {
    return '<div class="score-bar"><div class="score-fill" style="width:' + (n||0) + '%"></div></div>';
}

function switchTab(pages, active) {
    pages.forEach(function(p) {
        var pg = document.getElementById("page-" + p);
        var lk = document.getElementById("nav-" + p);
        if (pg) pg.classList.toggle("active", p === active);
        if (lk) lk.classList.toggle("active", p === active);
    });
}
