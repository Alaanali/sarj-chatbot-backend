from fabric import Connection, task

c = Connection("206.189.79.56", user="root")


@task
def init_droplet(connection):
    # Base system update
    c.run("apt-get update && apt-get upgrade -y")
    c.run("apt-get install -y python3 python3-pip python3-venv git curl")

    # Install uv (dependency manager)
    c.run("pip install uv")

    # Install Caddy
    c.run("apt-get install -y debian-keyring debian-archive-keyring apt-transport-https")
    c.run(
        "curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | "
        "gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg"
    )
    c.run(
        "curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | "
        "tee /etc/apt/sources.list.d/caddy-stable.list"
    )
    c.run("apt-get update && apt-get install -y caddy")

    # Upload systemd service files
    c.put("gunicorn.service", "/etc/systemd/system/gunicorn.service")
    c.put("caddy.service", "/etc/systemd/system/caddy.service")

    # Enable services
    c.run("systemctl enable gunicorn.service")
    c.run("systemctl enable caddy")

    c.run("apt-get install -y gettext")


@task
def init_ssh_key(connection):
    c.run("ssh-keygen -t ed25519 -C 'alaanourali@gmail.com'")


@task
def get_ssh_public_key(connection):
    c.run("cat ~/.ssh/id_ed25519.pub")


@task
def deploy_app(connection: Connection):
    # Clean old repo
    c.run("rm -rf sarj-chatbot-backend")

    # Clone fresh copy
    c.run("git clone git@github.com:Alaanali/sarj-chatbot-backend.git")

    with c.cd("sarj-chatbot-backend"):
        remote_path = c.run("pwd").stdout.strip()

        # Copy environment files
        c.put(local=".env", remote=f"{remote_path}/.env")
        c.put(local="env", remote=f"{remote_path}/env")

        # Install deps with uv
        c.run("~/.local/bin/uv sync --frozen")


@task
def update_gunicorn(connection: Connection):
    c.put("gunicorn.service", "/etc/systemd/system/gunicorn.service")
    c.run("systemctl daemon-reload")
    c.run("systemctl restart gunicorn")
    c.run("systemctl status gunicorn --no-pager")


@task
def update_caddy(connection: Connection):
    c.put("Caddyfile", "/etc/caddy/Caddyfile")
    c.run("systemctl reload caddy")
    c.run("systemctl status caddy --no-pager")
