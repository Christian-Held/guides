## ✅ Projekt-Ziel  
Ein vollständiger **Fullstack-Deployment** auf einem **Raspberry Pi 5**, bestehend aus:
- Spring Boot (Backend mit JWT & HTTP-only Cookies)
- React (Frontend mit HTTPS)
- PostgreSQL
- NGINX mit SSL
- GitHub Packages (Docker Images, Config Repo)

---

## 🔧 Hardware & System

| Komponente           | Details                             |
|----------------------|-------------------------------------|
| Gerät                | Raspberry Pi 5                      |
| OS                   | Ubuntu 22.04                        |
| Benutzer             | `admin`                             |
| IP-Adresse (LAN)     | `192.168.1.185`                     |
| Öffentliche IP       | `89.217.42.224` (Sunrise)           |
| Router               | Sunrise, Portweiterleitungen aktiv  |

---

## 📦 Software & Dienste

| Tool / Dienst           | Version / Status                            |
|--------------------------|---------------------------------------------|
| Docker                   | `25.x` (funktioniert einwandfrei)           |
| Docker Compose           | `v2.24.0` (manuell installiert)             |
| Java                     | OpenJDK 21 (für Spring Boot)                |
| Node.js / NPM            | Node `20.x`, NPM `10.x`                     |
| React                    | CRA `5.0.1` (Build-Ordner wird deployed)    |
| Spring Boot              | 3.x mit JWT-Security & Cookie-Auth         |
| PostgreSQL               | Lokal installiert, Port `5432`              |
| Anaconda                 | (Optional) für spätere Python-Projekte      |
| NGINX                    | 1.24.0 (mit Reverse Proxy & SSL)            |
| GitHub Container Registry| genutzt für alle Images (`ghcr.io`)         |

---

## 📁 Verzeichnisstruktur auf dem Pi

```bash
/home/admin/helddigital
  ├── backend
  │   ├── config-server
  │   ├── chatbot-backend
  ├── config-repo
/var/www/html/
  ├── helddigital          # React-Frontend
  ├── physioheld           # Statische HTML-Seite
```

---

## 🌐 Domains & DNS (via IONOS)

| Domain              | Weiterleitung                          | SSL             |
|---------------------|-----------------------------------------|-----------------|
| `helddigital.com`   | A-Record auf `89.217.42.224`            | ✅ IONOS-Zertifikat (manuell eingebunden) |
| `physiohinwil.ch`   | A-Record auf `89.217.42.224`            | ❌ noch kein SSL (geplant mit Let's Encrypt) |

---

## 🔐 Auth & Sicherheit

- Login/Signup über `/api/auth/**` erlaubt
- JWT wird als **HTTP-only Cookie** gesetzt
- Keine Session, vollständig **stateless**
- CORS auf Frontend-Domains begrenzt (`helddigital.com`, `localhost:3443`, etc.)
- NGINX leitet `/api/` sauber ans Backend weiter

---

## 🔁 Deployment-Workflow

### Docker:
- Images werden auf dem **Dev-PC gebaut**
- Dann mit `docker push` nach `ghcr.io/helddigital/...`
- Auf dem Pi mit `docker pull` + `docker-compose up -d`

### Frontend:
- `npm run build` auf dem Dev-PC
- Deployment via:
  - `scp -r build/* admin@192.168.1.185:/var/www/html/helddigital/`
  - alternativ direkt mit VS Code SSH Extension

---

## 🔍 Nützliche Befehle

```bash
# SSH auf den Pi
ssh admin@192.168.1.185

# Docker Login (GHCR)
echo "<TOKEN>" | docker login ghcr.io -u HeldDigital --password-stdin

# Start Services
cd ~/helddigital/backend/config-server
docker-compose up -d

# Logs vom Backend anzeigen
docker logs -f <container-name>

# Zertifikat testen
curl -vk https://helddigital.com

# NGINX neu laden
sudo nginx -t && sudo systemctl reload nginx
```

---

## 🚀 Ergebnis

- Die Seite **https://helddigital.com** ist öffentlich mit SSL & Login erreichbar.
- Die statische Seite **http://physiohinwil.ch** zeigt korrekt Inhalte.
- Die komplette Plattform läuft auf einem **self-hosted Pi 5**, performant & sicher.
