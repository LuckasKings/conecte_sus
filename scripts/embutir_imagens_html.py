"""
Script para embutir imagens em Base64 nos relatórios HTML do SUS Acre.
Isso torna os arquivos HTML 100% autônomos (standalone), permitindo que sejam compartilhados
por e-mail, Teams, WhatsApp, etc. sem depender de arquivos externos ou quebra de caminhos.
"""

import os
import re
import base64
from pathlib import Path

def embutir_imagens_em_html(html_path: str, output_path: str = None):
    html_file = Path(html_path)
    if not html_file.exists():
        print(f"[-] Arquivo não encontrado: {html_path}")
        return
    
    if output_path is None:
        output_path = html_path  # Sobrescreve para atualizar direto o relatório

    with open(html_file, 'r', encoding='utf-8') as f:
        conteudo = f.read()

    # Expressão regular para capturar tags <img> com src
    padrao_img = re.compile(r'(<img\s+[^>]*?src=["\'])([^"\']+)(["\'][^>]*?>)', re.IGNORECASE)

    substituicoes = 0

    def substituir_src(match):
        nonlocal substituicoes
        prefixo = match.group(1)
        src = match.group(2)
        sufixo = match.group(3)

        # Se já for Base64 ou URL http, ignora
        if src.startswith('data:') or src.startswith('http://') or src.startswith('https://'):
            return match.group(0)

        # Resolve o caminho do arquivo relativo ao diretório do HTML
        caminho_imagem = (html_file.parent / src).resolve()

        # Caso não encontre diretamente, tenta procurar no diretório de figuras do projeto
        if not caminho_imagem.exists():
            caminho_alt = Path("figuras") / Path(src).name
            if caminho_alt.exists():
                caminho_imagem = caminho_alt.resolve()

        if caminho_imagem.exists():
            extensao = caminho_imagem.suffix.lower().replace('.', '')
            mime_type = "image/png" if extensao == "png" else f"image/{extensao}"
            with open(caminho_imagem, 'rb') as img_f:
                b64_dados = base64.b64encode(img_f.read()).decode('utf-8')
            substituicoes += 1
            print(f"  [+] Embutida imagem: {caminho_imagem.name} ({len(b64_dados) // 1024} KB)")
            return f'{prefixo}data:{mime_type};base64,{b64_dados}{sufixo}'
        else:
            print(f"  [!] Imagem não encontrada: {src} (resolvido como: {caminho_imagem})")
            return match.group(0)

    novo_conteudo = padrao_img.sub(substituir_src, conteudo)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(novo_conteudo)

    tamanho_final_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[✓] Concluído: {output_path} | Imagens embutidas: {substituicoes} | Tamanho final: {tamanho_final_mb:.2f} MB\n")

if __name__ == '__main__':
    print("Processando relatorios HTML para embutir imagens em Base64...\n")
    
    # 1. Relatório Principal de Análise
    print("1. Processando relatorio_analise_sus.html:")
    embutir_imagens_em_html('relatorios/relatorio_analise_sus.html')
    
    # 2. Resumo de Mudanças de Variáveis V3
    print("2. Processando resumo_mudanca_variaveis_v3.html:")
    embutir_imagens_em_html('relatorios/resumo_mudanca_variaveis_v3.html')
