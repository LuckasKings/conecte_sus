import ftplib
import os
import time

HOST = "ftp.datasus.gov.br"
DIRETORIO = "/dissemin/publicos/SIHSUS/200801_/Dados/"
UF = "AC"
ANO = "24"

def baixar_arquivo(nome_arquivo, tentativas=5):
    for tentativa in range(1, tentativas + 1):
        try:
            print(f"  Tentativa {tentativa} para {nome_arquivo}...")
            with ftplib.FTP(HOST, timeout=120) as ftp:  # timeout maior
                ftp.login()  # anônimo
                ftp.cwd(DIRETORIO)
                with open(nome_arquivo, "wb") as f:
                    ftp.retrbinary(f"RETR {nome_arquivo}", f.write)
            print(f"  ✅ {nome_arquivo} baixado")
            return True
        except Exception as e:
            print(f"  ❌ Falha: {e}")
            time.sleep(5)  # espera antes de tentar de novo
    return False

for mes in range(1, 13):
    arquivo = f"RD{UF}{ANO}{mes:02d}.dbc"
    if os.path.exists(arquivo):
        print(f"⏭️  {arquivo} já existe, pulando")
        continue
    baixar_arquivo(arquivo)