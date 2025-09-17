Self‑Hosted Deployment (Nginx + Uvicorn + TLS)

Assumptions
- Project dir: `/home/ansible/git_repos/emBed`
- Venv: `/home/ansible/git_repos/emBed/env`
- App: `src.webapp:app`
- HTTPS on this server’s public IP/DNS

1) Install prerequisites
    sudo bash deploy/scripts/install_prereqs.sh

2) Create a self-signed TLS certificate
- Quick self-signed (browser warning):
    bash deploy/scripts/generate_cert.sh quick --ip YOUR_SERVER_IP
- Local CA (nicer; import `ca.crt` on clients to avoid warnings):
    bash deploy/scripts/generate_cert.sh local-ca --ip YOUR_SERVER_IP

Files: `/etc/ssl/self/lancedb.key` and `/etc/ssl/self/lancedb.crt`.

3) (Recommended) Protect with Basic Auth
    bash deploy/scripts/setup_basic_auth.sh demo

Adds/updates `/etc/nginx/.htpasswd`. You’ll be prompted for a password.

4) Systemd unit for Uvicorn
    sudo install -m 0644 deploy/systemd/lancedb-webapp.service /etc/systemd/system/lancedb-webapp.service
    sudo systemctl daemon-reload
    sudo systemctl enable --now lancedb-webapp
    sudo systemctl status lancedb-webapp

The unit binds to `127.0.0.1:8000`; Nginx is the public entrypoint. The process loads env from `.env` at the repo root.

5) Nginx reverse proxy with HTTPS and headers
    sudo install -m 0644 deploy/nginx/lancedb /etc/nginx/sites-available/lancedb
    sudo ln -sf /etc/nginx/sites-available/lancedb /etc/nginx/sites-enabled/lancedb
    sudo nginx -t && sudo systemctl reload nginx

6) Firewall (optional)
If using UFW:
    sudo ufw allow OpenSSH
    sudo ufw allow 80/tcp
    sudo ufw allow 443/tcp
    sudo ufw enable
    sudo ufw status

7) Test end-to-end
Local health:
    curl -fsS http://127.0.0.1:8000/health && echo OK

Remote test (ignore warning for self-signed):
    curl -k https://YOUR_SERVER/health

Open `https://YOUR_SERVER/` in a browser and log in with your Basic Auth user.

8) Updates and ingestion
- New data ingested into the same `LANCEDB_URI`/table is immediately visible.
- To change environment (e.g., different `LANCEDB_URI`), edit `.env` then:
    sudo systemctl restart lancedb-webapp

Logs:
    journalctl -u lancedb-webapp -f

9) Optional hardening knobs
- Set `WEBAPP_CORS_ORIGINS=https://YOUR_SERVER` in `.env`.
- Configure `WEBAPP_PRESIGN_TTL` (seconds) in `.env` to shorten preview URL lifetime.
- Toggle features via env: `WEBAPP_SHOW_IMAGES=0`, reduce `WEBAPP_MAX_UPLOAD_BYTES`, etc.
- Restrict by IP or place behind a VPN using Nginx or firewall rules.

One‑shot convenience deploy
After generating TLS and Basic Auth, you can run:
    bash deploy/scripts/deploy.sh

This wires up the systemd unit, Nginx site, reloads services, and runs a quick health check.
