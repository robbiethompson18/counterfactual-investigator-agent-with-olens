# Sandbox image for the metadata_honeypot.ctf env (InterCode-CTF challenges). Mirrors the tooling in UK AISI's
# inspect_evals gdm_intercode_ctf image, plus a few extra forensics/RE tools, so the model can actually attempt the
# challenges. Build: docker build -t evalcue-ctf -f evalcue/ctf.Dockerfile evalcue
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive PIP_BREAK_SYSTEM_PACKAGES=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      bash python3 python3-pip psmisc bsdmainutils cron imagemagick tcpdump tshark dnsutils git wget gcc \
      sleuthkit netcat-traditional unzip curl gdb poppler-utils tmux jq file binutils xxd \
      exiftool steghide binwalk foremost openssl feh && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip3 install --no-cache-dir scapy dpkt pycryptodome cryptography owiener numpy sympy requests Pillow python-whois PyPDF2
WORKDIR /root/ctf
