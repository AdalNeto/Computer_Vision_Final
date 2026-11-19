import os
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =========================================================================
# 0. CONFIGURAÇÃO DE DIRETÓRIOS, HOMOGRAFIA E PARÂMETROS
# =========================================================================
# Caminho do vídeo de entrada e criação automática da pasta de resultados
CAMINHO_VIDEO = "video2.mp4"
nome_video = os.path.splitext(os.path.basename(CAMINHO_VIDEO))[0]
PASTA_SAIDA = f"Resultados_Projeto_{nome_video}/"
os.makedirs(PASTA_SAIDA, exist_ok=True)

# Fator de escala espacial aplicado aos quadros:
# 1.0 -> resolução nativa 4K (3840x2160)
# 0.5 -> Full HD (1920x1080) | 0.25 -> qHD (960x540)
ESCALA = 1

# Matriz de homografia HL (3x3) calibrada na resolução 3840 x 2160.
# Mapeia coordenadas homogêneas do plano do piso (Xw, Yw, 1)^T para pixels (u, v, 1)^T.
HL = np.array(
    [
        [0.0, -0.9102, 1952.5904],
        [-0.9236, 0.0, 1081.7977],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# Matriz inversa HL^-1: realiza o mapeamento reverso (plano da imagem -> plano do mundo Zw = 0)
HL_INV = np.linalg.inv(HL)

# Parâmetros de segmentação e rastreamento:
# - LIMIAR_CROMATICIDADE = 0.09 exige predominância mínima de 9% do canal vermelho normalizado
LIMIAR_CROMATICIDADE = 0.09
# - CIRCULARIDADE_MIN = 0.40 separa formas compactas como o retângulo (C ~ 0.65)
#   de interferentes alongados/côncavos como a cruz vermelha no piso (C ~ 0.31)
CIRCULARIDADE_MIN = 0.40
# - AREA_MIN_4K descarta ruídos pontuais de poucos pixels na resolução 4K
AREA_MIN_4K = 600.0
# - DIST_MAX_GATE restringe o deslocamento máximo aceito entre quadros consecutivos (evita saltos)
DIST_MAX_GATE = 800.0 * ESCALA
# - RAIO_ROI define a meia-largura da janela de busca local (Region of Interest) ao redor do robô
RAIO_ROI = int(round(900.0 * ESCALA))
# - THETA_INICIAL_REF orienta o desempate de 180° exclusivamente no primeiro quadro (k = 0)
THETA_INICIAL_REF = np.deg2rad(0.0)


def desenhar_quadro_info(img, linhas, cores):
    """
    Desenha uma caixa preta ajustada ao texto no canto superior esquerdo da imagem,
    escalando fonte, margens e espessura proporcionalmente à largura do vídeo.
    """
    largura_video = img.shape[1]
    fator = largura_video / 960.0  # Fator de proporção (1.0 em 960px, 4.0 em 3840px)

    fonte = cv2.FONT_HERSHEY_SIMPLEX
    escala_fonte = max(0.35, 0.55 * fator)
    espessura = max(1, int(round(1.6 * fator)))
    margem = max(6, int(round(10 * fator)))
    pad = max(4, int(round(8 * fator)))
    espaco_linhas = max(3, int(round(6 * fator)))

    # Calcula previamente a largura e altura exatas ocupadas por cada linha de texto
    medidas = [
        cv2.getTextSize(txt, fonte, escala_fonte, espessura) for txt in linhas
    ]
    largura_max = max(m[0][0] for m in medidas)
    altura_total = sum(m[0][1] + m[1] for m in medidas) + espaco_linhas * (
        len(linhas) - 1
    )

    # Coordenadas do retângulo de fundo que envolve todo o bloco de texto
    x0, y0 = margem, margem
    x1 = x0 + largura_max + 2 * pad
    y1 = y0 + altura_total + 2 * pad

    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), -1)

    # Escreve cada linha sequencialmente respeitando a linha de base (baseline) da fonte
    y_cursor = y0 + pad
    for txt, cor, ((_, alt_txt), baseline) in zip(linhas, cores, medidas):
        y_cursor += alt_txt
        cv2.putText(
            img, txt, (x0 + pad, y_cursor), fonte, escala_fonte, cor, espessura
        )
        y_cursor += baseline + espaco_linhas


# =========================================================================
# ETAPA 1: LEITURA E PADRONIZAÇÃO DO VÍDEO
# =========================================================================
cap = cv2.VideoCapture(CAMINHO_VIDEO)
if not cap.isOpened():
    raise FileNotFoundError(f"Não foi possível abrir o vídeo: {CAMINHO_VIDEO}")

# Extração das propriedades temporais e espaciais do contêiner de vídeo
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0
W_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Dimensões de processamento após aplicar ESCALA e ajuste quadrático da área mínima
W = int(round(W_orig * ESCALA))
H = int(round(H_orig * ESCALA))
area_min_proc = max(15.0, AREA_MIN_4K * (ESCALA**2))

# Parâmetros gráficos proporcionais à resolução de saída
fator_vis = max(1.0, W / 960.0)
esp_desenho = max(2, int(round(2.0 * fator_vis)))
raio_centro = max(4, int(round(4.0 * fator_vis)))
comp_seta = 140.0 * ESCALA

# Elemento estruturante elíptico ímpar escalado para as operações morfológicas
k_size = max(3, int(round(18 * ESCALA)) | 1)
kernel_morf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

# Inicialização do gravador do vídeo anotado (.mp4)
caminho_video_saida = os.path.join(PASTA_SAIDA, f"{nome_video}_anotado.mp4")
writer = cv2.VideoWriter(
    caminho_video_saida, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)
)

# Seleciona três quadros distribuídos (15%, 50% e 85%) para ilustrar o pipeline no relatório
indices_exemplo = {
    max(1, int(total_frames_meta * 0.15)),
    max(2, int(total_frames_meta * 0.50)),
    max(3, int(total_frames_meta * 0.85)),
}
exemplos_salvos = []

print("=" * 74)
print("ETAPA 1 - PROPRIEDADES DO VÍDEO CARREGADO")
print("=" * 74)
print(f"Arquivo:             {CAMINHO_VIDEO}")
print(f"Resolução Original:  {W_orig} x {H_orig} pixels")
print(f"Resolução Process.:  {W} x {H} pixels (escala {ESCALA}x)")
print(f"Taxa de quadros:     {fps:.2f} FPS")
print(f"Quadros (metadado):  {total_frames_meta} frames ({total_frames_meta / fps:.2f} s)")
print("=" * 74)
print("Processando vídeo...")


# =========================================================================
# LOOP QUADRO A QUADRO: ETAPAS 2, 3, 4 E 5
# =========================================================================
registros = []
tempos_proc_ms = []
uc_anterior = None       # Posição horizontal (u) do marcador no quadro anterior
vc_anterior = None       # Posição vertical (v) do marcador no quadro anterior
theta_I_anterior = None  # Orientação na imagem no quadro anterior (para continuidade)
k = 0                    # Contador sequencial de quadros decodificados

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break

    t_inicio = time.perf_counter()
    tempo_s = k / fps

    # Redimensiona o quadro para a resolução de processamento (INTER_AREA evita aliasing)
    frame_proc = cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_AREA)
    frame_desenhado = frame_proc.copy()

    # Otimização por ROI (Region of Interest):
    # Se o robô já foi localizado no quadro anterior (e não é um quadro de exemplo completo),
    # recorta apenas a vizinhança local, reduzindo o número de pixels processados.
    usar_roi = (uc_anterior is not None) and (k not in indices_exemplo)
    if usar_roi:
        x0 = max(0, int(round(uc_anterior)) - RAIO_ROI)
        y0 = max(0, int(round(vc_anterior)) - RAIO_ROI)
        x1 = min(W, int(round(uc_anterior)) + RAIO_ROI)
        y1 = min(H, int(round(vc_anterior)) + RAIO_ROI)
    else:
        x0, y0, x1, y1 = 0, 0, W, H

    roi_bgr = frame_proc[y0:y1, x0:x1]

    # ---------------------------------------------------------------------
    # ETAPA 2: SEGMENTAÇÃO POR CROMATICIDADE NORMALIZADA
    # ---------------------------------------------------------------------
    # Converte BGR -> RGB em ponto flutuante e normaliza cada canal pela soma RGB
    # para obter invariância a variações de iluminação e sombras
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    soma_rgb = np.sum(roi_rgb, axis=2) + 1e-6
    r_norm = roi_rgb[:, :, 0] / soma_rgb
    g_norm = roi_rgb[:, :, 1] / soma_rgb
    b_norm = roi_rgb[:, :, 2] / soma_rgb

    # Evidência de vermelho: r - max(g, b) > 0.09
    evidencia_vermelho = r_norm - np.maximum(g_norm, b_norm)
    mascara_binaria = (evidencia_vermelho > LIMIAR_CROMATICIDADE).astype(np.uint8)

    # ---------------------------------------------------------------------
    # ETAPA 3: MORFOLOGIA E SELEÇÃO POR CIRCULARIDADE + MAIOR ÁREA
    # ---------------------------------------------------------------------
    # Abertura (erosão -> dilatação): elimina ruídos isolados e reflexos finos
    mascara_morf = cv2.morphologyEx(mascara_binaria, cv2.MORPH_OPEN, kernel_morf)
    # Fechamento (dilatação -> erosão): preenche pequenas falhas internas no marcador
    mascara_morf = cv2.morphologyEx(mascara_morf, cv2.MORPH_CLOSE, kernel_morf)

    # Rotula cada região conectada (8-conectividade) e extrai suas estatísticas geométricas
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mascara_morf, connectivity=8
    )

    melhor_candidato = None
    melhor_score = -1e9

    # Itera sobre os componentes detectados (índice 0 é o fundo)
    for lbl in range(1, num_labels):
        area_cc = float(stats[lbl, cv2.CC_STAT_AREA])
        if area_cc < area_min_proc:
            continue

        # Recorta apenas a caixa delimitadora (bounding box) do componente.
        # Como os momentos centrais são invariantes à translação, o cálculo de ângulo
        # é idêntico ao da imagem inteira, porém muito mais rápido.
        bx = int(stats[lbl, cv2.CC_STAT_LEFT])
        by = int(stats[lbl, cv2.CC_STAT_TOP])
        bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])

        sub_mask = (labels[by : by + bh, bx : bx + bw] == lbl).astype(np.uint8)
        contornos, _ = cv2.findContours(
            sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contornos:
            continue

        cnt = max(contornos, key=cv2.contourArea)
        area_cnt = cv2.contourArea(cnt)
        perimetro = cv2.arcLength(cnt, True)
        if perimetro <= 0:
            continue

        # Índice de compacidade isoperimétrica (Circularidade): C = 4*pi*A / P^2
        # Círculo perfeito = 1.0 | Retângulo 2:1 ~ 0.65 | Cruz do piso ~ 0.31
        circularidade = (4.0 * np.pi * area_cnt) / (perimetro**2)
        if circularidade < CIRCULARIDADE_MIN:
            continue

        # Momentos espaciais e centrais do componente candidato
        M_cand = cv2.moments(sub_mask)
        if M_cand["m00"] <= 0:
            continue

        # Converte o centroide local da sub-máscara para coordenadas globais da imagem
        offset_x = x0 + bx
        offset_y = y0 + by
        uc_cand = offset_x + (M_cand["m10"] / M_cand["m00"])
        vc_cand = offset_y + (M_cand["m01"] / M_cand["m00"])

        # Gating espacial: rejeita candidatos muito distantes da última posição conhecida
        dist_prev = 0.0
        if uc_anterior is not None:
            dist_prev = float(np.hypot(uc_cand - uc_anterior, vc_cand - vc_anterior))
            if dist_prev > DIST_MAX_GATE:
                continue

        # Função de pontuação: prioriza o maior objeto compacto (marcador retangular)
        # sobre pequenos LEDs/adesivos do robô, penalizando saltos bruscos
        score = area_cnt - (1.5 * dist_prev)
        if score > melhor_score:
            melhor_score = score
            cnt_global = cnt + np.array([[[offset_x, offset_y]]], dtype=np.int32)
            melhor_candidato = (
                lbl,
                cnt_global,
                M_cand,
                uc_cand,
                vc_cand,
                circularidade,
            )

    # ---------------------------------------------------------------------
    # ETAPAS 4 E 5: POSE NA IMAGEM (q_I) E POSE NO MUNDO (q_W)
    # ---------------------------------------------------------------------
    if melhor_candidato is not None:
        valido = 1
        lbl_sel, contorno_sel, M, uc_red, vc_red, circ_sel = melhor_candidato

        # Etapa 4: Matriz de inércia / covariância dos momentos centrais de 2ª ordem
        J = np.array(
            [[M["mu20"], M["mu11"]], [M["mu11"], M["mu02"]]], dtype=np.float64
        )
        # O autovetor associado ao maior autovalor define o eixo longitudinal do marcador
        autovalores, autovetores = np.linalg.eigh(J)
        q1 = autovetores[:, np.argsort(autovalores)[::-1][0]]
        theta_bruto = float(np.arctan2(q1[1], q1[0]))

        # Resolução da ambiguidade de 180° (eixo principal não possui sentido intrínseco):
        # Compara os dois sentidos opostos (theta e theta + pi) com a orientação anterior
        # (ou com THETA_INICIAL_REF no quadro 0) e escolhe o de menor diferença angular.
        t_cand1 = float(np.arctan2(np.sin(theta_bruto), np.cos(theta_bruto)))
        t_cand2 = float(
            np.arctan2(np.sin(theta_bruto + np.pi), np.cos(theta_bruto + np.pi))
        )
        ref_theta = (
            theta_I_anterior if theta_I_anterior is not None else THETA_INICIAL_REF
        )

        diff_t1 = abs(
            np.arctan2(np.sin(t_cand1 - ref_theta), np.cos(t_cand1 - ref_theta))
        )
        diff_t2 = abs(
            np.arctan2(np.sin(t_cand2 - ref_theta), np.cos(t_cand2 - ref_theta))
        )
        theta_I = t_cand1 if diff_t1 <= diff_t2 else t_cand2

        # Etapa 5: Transformação para o referencial do mundo q_W = [Xw, Yw, psi]^T
        # Primeiro reescala o centroide para a resolução 4K original em que HL foi calibrada
        uc = uc_red / ESCALA
        vc = vc_red / ESCALA

        # Projeta o centroide (uc, vc, 1)^T para o plano do piso via homografia inversa
        Pw1 = HL_INV @ np.array([uc, vc, 1.0], dtype=np.float64)
        Xw = float(Pw1[0] / Pw1[2])  # Deshomogeneização (divisão pela 3ª coordenada)
        Yw = float(Pw1[1] / Pw1[2])

        # Projeta um segundo ponto deslocado l_px = 100 px na direção da frente do robô (theta_I)
        # para obter o vetor direção já transformado no espaço métrico do mundo
        l_px = 100.0
        Pw2 = HL_INV @ np.array(
            [uc + l_px * np.cos(theta_I), vc + l_px * np.sin(theta_I), 1.0],
            dtype=np.float64,
        )
        X2 = float(Pw2[0] / Pw2[2])
        Y2 = float(Pw2[1] / Pw2[2])

        # Ângulo de guinada (yaw) psi no referencial do mundo
        psi = float(np.arctan2(Y2 - Yw, X2 - Xw))

        # Atualiza o estado anterior para rastreamento no próximo quadro
        uc_anterior = uc_red
        vc_anterior = vc_red
        theta_I_anterior = theta_I

        # Desenho do retângulo envolvente mínimo (verde), centroide (ciano) e seta frontal (amarela)
        caixa_pts = np.int32(np.round(cv2.boxPoints(cv2.minAreaRect(contorno_sel))))
        cv2.drawContours(frame_desenhado, [caixa_pts], 0, (0, 255, 0), esp_desenho)

        centro_pt = (int(round(uc_red)), int(round(vc_red)))
        cv2.circle(frame_desenhado, centro_pt, raio_centro, (255, 255, 0), -1)

        ponta_pt = (
            int(round(uc_red + comp_seta * np.cos(theta_I))),
            int(round(vc_red + comp_seta * np.sin(theta_I))),
        )
        cv2.arrowedLine(
            frame_desenhado,
            centro_pt,
            ponta_pt,
            (0, 255, 255),
            esp_desenho,
            tipLength=0.25,
        )

        # Renderiza o quadro de telemetria proporcional com as poses na imagem e no mundo
        linha1 = (
            f"Frame {k:04d} | q_I=({uc:.0f}, {vc:.0f} px,"
            f" {np.degrees(theta_I):.1f} deg)"
        )
        linha2 = f"q_W=({Xw:.0f} mm, {Yw:.0f} mm, psi={np.degrees(psi):.1f} deg)"
        desenhar_quadro_info(
            frame_desenhado, [linha1, linha2], [(0, 255, 0), (0, 255, 255)]
        )

    else:
        # Caso nenhum candidato válido seja encontrado no quadro atual
        valido = 0
        lbl_sel = -1
        uc = vc = theta_I = Xw = Yw = psi = circ_sel = np.nan
        # Reseta a posição anterior para que o próximo quadro busque na imagem inteira
        uc_anterior = None
        vc_anterior = None

        desenhar_quadro_info(
            frame_desenhado, [f"Frame {k:04d} | FALHA NA DETECCAO"], [(0, 0, 255)]
        )

    # Mede o tempo total de processamento do quadro atual em milissegundos
    dt_ms = (time.perf_counter() - t_inicio) * 1000.0
    tempos_proc_ms.append(dt_ms)

    # Armazena os dados visuais dos 3 quadros intermediários para a Figura 1
    if k in indices_exemplo:
        mask_vis = (
            (labels == lbl_sel).astype(np.uint8)
            if lbl_sel > 0
            else np.zeros_like(mascara_morf)
        )
        exemplos_salvos.append((
            k,
            cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB),
            evidencia_vermelho.copy(),
            mask_vis,
            cv2.cvtColor(frame_desenhado, cv2.COLOR_BGR2RGB),
        ))

    # Registra todas as variáveis do quadro para exportação tabular
    registros.append({
        "frame": k,
        "tempo_s": tempo_s,
        "valido": valido,
        "circularidade": circ_sel,
        "uc_px": uc,
        "vc_px": vc,
        "theta_I_rad": theta_I,
        "Xw_mm": Xw,
        "Yw_mm": Yw,
        "psi_rad": psi,
        "tempo_ms": dt_ms,
    })

    writer.write(frame_desenhado)
    k += 1

# Libera os recursos de leitura e escrita de vídeo
cap.release()
writer.release()


# =========================================================================
# ETAPA 6: EXPORTAÇÃO TABULAR, RESUMO QUANTITATIVO E GRÁFICOS FINAIS
# =========================================================================
df = pd.DataFrame(registros)

# Exporta os dados em CSV compatível com Excel em português (separador ';' e decimal ',')
# Inclui tratamento caso o arquivo original já esteja aberto no Excel
caminho_csv = os.path.join(PASTA_SAIDA, f"{nome_video}_poses.csv")
try:
    df.to_csv(
        caminho_csv,
        sep=";",
        decimal=",",
        float_format="%.4f",
        encoding="utf-8-sig",
        index=False,
    )
except PermissionError:
    caminho_csv = os.path.join(PASTA_SAIDA, f"{nome_video}_poses_atualizado.csv")
    df.to_csv(
        caminho_csv,
        sep=";",
        decimal=",",
        float_format="%.4f",
        encoding="utf-8-sig",
        index=False,
    )
    print(f"[AVISO] Arquivo original aberto no Excel. Salvo em: {caminho_csv}")

# Cálculo das métricas quantitativas de desempenho e robustez
taxa_validos = 100.0 * df["valido"].mean()
total_invalidos = int((df["valido"] == 0).sum())
tempo_medio_ms = float(np.mean(tempos_proc_ms))
duracao_real_s = (len(df) - 1) / fps if len(df) > 0 else 0.0

print("\n" + "=" * 74)
print(f"ETAPA 6 - RESUMO QUANTITATIVO ({nome_video.upper()})")
print("=" * 74)
print(f"Quadros reais decodificados: {len(df)} (metadado: {total_frames_meta})")
print(f"Último quadro / duração:     frame {len(df) - 1} ({duracao_real_s:.2f} s)")
print(f"Detecções válidas:           {int(df['valido'].sum())} ({taxa_validos:.2f}%)")
print(f"Falhas de detecção:          {total_invalidos}")
print(f"Tempo médio por quadro:      {tempo_medio_ms:.2f} ms/frame")
print(f"Vídeo anotado salvo em:      {caminho_video_saida}")
print(f"Dados CSV salvos em:         {caminho_csv}")
print("=" * 74)

# --- GRÁFICO 1: Três Exemplos Intermediários da Segmentação ---
# Exibe lado a lado: RGB original, mapa de cromaticidade, máscara filtrada e pose final
if exemplos_salvos:
    fig, axes = plt.subplots(
        len(exemplos_salvos), 4, figsize=(16, 3.6 * len(exemplos_salvos))
    )
    if len(exemplos_salvos) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, (fk, rgb_ex, ev_ex, mask_ex, vis_ex) in enumerate(exemplos_salvos):
        axes[row, 0].imshow(rgb_ex)
        axes[row, 0].set_title(f"Quadro {fk} - RGB")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(ev_ex, cmap="inferno", vmin=-0.05, vmax=0.35)
        axes[row, 1].set_title(r"Cromaticidade $r - \max(g, b)$")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(mask_ex, cmap="gray")
        axes[row, 2].set_title("Máscara (Circularidade)")
        axes[row, 2].axis("off")

        axes[row, 3].imshow(vis_ex)
        axes[row, 3].set_title("Retângulo, Centroide e Seta")
        axes[row, 3].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_SAIDA, "1_exemplos_intermediarios.png"), dpi=150)
    plt.show()

# Desenrola a fase (np.unwrap) dos ângulos válidos:
# Evita descontinuidades verticais de 360° quando o robô ultrapassa +-180°
mask_val = df["valido"] == 1
theta_unwrapped = np.full(len(df), np.nan)
psi_unwrapped = np.full(len(df), np.nan)
if mask_val.any():
    theta_unwrapped[mask_val] = np.degrees(
        np.unwrap(df.loc[mask_val, "theta_I_rad"])
    )
    psi_unwrapped[mask_val] = np.degrees(np.unwrap(df.loc[mask_val, "psi_rad"]))

# --- GRÁFICO 2: Posição e Orientação no Plano da Imagem (uc, vc, theta_I) ---
fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

axes[0].plot(df["tempo_s"], df["uc_px"], "b-", linewidth=1.6)
axes[0].set_ylabel(r"$u_c$ (pixels)")
axes[0].set_title("Pose no Plano da Imagem ao Longo do Tempo")
axes[0].grid(True, linestyle="--", alpha=0.5)

axes[1].plot(df["tempo_s"], df["vc_px"], "g-", linewidth=1.6)
axes[1].set_ylabel(r"$v_c$ (pixels)")
axes[1].grid(True, linestyle="--", alpha=0.5)

axes[2].plot(df["tempo_s"], theta_unwrapped, "r-", linewidth=1.6)
axes[2].set_ylabel(r"$\theta_I$ (graus)")
axes[2].set_xlabel("Tempo (s)")
axes[2].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(PASTA_SAIDA, "2_pose_plano_imagem.png"), dpi=150)
plt.show()

# --- GRÁFICO 3: Posição e Orientação no Plano do Mundo (Xw, Yw, psi) ---
fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

axes[0].plot(df["tempo_s"], df["Xw_mm"], "b-", linewidth=1.6)
axes[0].set_ylabel(r"$X_W$ (mm)")
axes[0].set_title("Pose no Plano do Mundo ao Longo do Tempo")
axes[0].grid(True, linestyle="--", alpha=0.5)

axes[1].plot(df["tempo_s"], df["Yw_mm"], "g-", linewidth=1.6)
axes[1].set_ylabel(r"$Y_W$ (mm)")
axes[1].grid(True, linestyle="--", alpha=0.5)

axes[2].plot(df["tempo_s"], psi_unwrapped, "m-", linewidth=1.6)
axes[2].set_ylabel(r"$\psi$ (graus)")
axes[2].set_xlabel("Tempo (s)")
axes[2].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(PASTA_SAIDA, "3_pose_plano_mundo.png"), dpi=150)
plt.show()

# --- GRÁFICO 4: Trajetória Planar Xw x Yw com Setas de Orientação ---
df_val = df[mask_val].reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8.5, 7))
ax.plot(
    df_val["Xw_mm"],
    df_val["Yw_mm"],
    "b-",
    linewidth=2.0,
    label="Trajetória do Robô",
)

if len(df_val) > 0:
    # Marca o ponto inicial (círculo verde) e o ponto final (quadrado vermelho)
    ax.plot(
        df_val["Xw_mm"].iloc[0],
        df_val["Yw_mm"].iloc[0],
        "go",
        markersize=10,
        label="Início",
    )
    ax.plot(
        df_val["Xw_mm"].iloc[-1],
        df_val["Yw_mm"].iloc[-1],
        "rs",
        markersize=10,
        label="Fim",
    )

    # Plota vetores unitários (cos(psi), sin(psi)) subamostrados ao longo do caminho
    passo_setas = max(1, len(df_val) // 25)
    sub = df_val.iloc[::passo_setas]
    ax.quiver(
        sub["Xw_mm"],
        sub["Yw_mm"],
        np.cos(sub["psi_rad"]),
        np.sin(sub["psi_rad"]),
        color="darkorange",
        scale=22,
        width=0.004,
        label=r"Orientação $\psi$",
    )

ax.set_xlabel(r"$X_W$ (mm)")
ax.set_ylabel(r"$Y_W$ (mm)")
ax.set_title(r"Trajetória Planar no Mundo ($X_W \times Y_W$)")
ax.axis("equal")  # Garante escala 1:1 entre os eixos Xw e Yw sem distorção geométrica
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="best")

plt.tight_layout()
plt.savefig(os.path.join(PASTA_SAIDA, "4_trajetoria_mundo.png"), dpi=150)
plt.show()

# --- GRÁFICO 5: Indicador Temporal de Validade das Detecções ---
fig, ax = plt.subplots(figsize=(11, 2.8))
ax.step(df["tempo_s"], df["valido"], where="post", color="teal", linewidth=1.8)
ax.set_ylim(-0.15, 1.25)
ax.set_yticks([0, 1])
ax.set_yticklabels(["Inválido (0)", "Válido (1)"])
ax.set_xlabel("Tempo (s)")
ax.set_title(
    f"Validade das Detecções ao Longo do Vídeo (Taxa Válida: {taxa_validos:.1f}%)"
)
ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(PASTA_SAIDA, "5_validade_deteccoes.png"), dpi=150)
plt.show()