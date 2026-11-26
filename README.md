# Estimativa de Pose Planar de Robô Móvel (ENGG64 — Visão Computacional)

Pipeline em Python e OpenCV para detecção, rastreamento e estimativa da pose planar $q_W = [X_W, Y_W, \psi]^T$ de um robô móvel equipado com marcador retangular vermelho, observado por uma câmera fixa superior com calibração por homografia ($H_L$).

## Autores

* **Adalberto Manoel do Bomfim Neto**
* **Maria Eduarda Souza Andrade**
* **Natália de Souza Haine Bastos**

---

## Visão Geral do Pipeline

1. **Leitura e Padronização Espacial:** Extração dos metadados do vídeo, reamostragem dos quadros pelo fator `ESCALA` e dimensionamento proporcional dos elementos morfológicos e gráficos (retângulo envolvente, seta de orientação e quadro de telemetria).
2. **Segmentação por Cromaticidade Normalizada:** Isolamento do marcador vermelho por meio da evidência cromática $r - \max(g, b) > 0{,}09$, garantindo invariância a variações de iluminação e sombras, aliado à busca local por Região de Interesse (ROI).
3. **Filtragem Morfológica e Seleção Geométrica:** Operações de abertura e fechamento com elemento estruturante elíptico, rotulagem de componentes conexos, filtragem pela compacidade isoperimétrica ($C = \frac{4\pi A}{P^2} \ge 0{,}40$) para rejeitar a cruz vermelha do piso e seleção pela maior área com *gating* espacial entre quadros consecutivos.
4. **Estimativa de Pose no Plano da Imagem ($q_I$):** Cálculo do centroide $(u_c, v_c)$ via momentos espaciais de 1ª ordem e extração do eixo principal $\theta_I$ pelos autovetores da matriz de inércia de 2ª ordem, resolvendo a ambiguidade de $180^\circ$ por continuidade temporal.
5. **Projeção para o Plano do Mundo ($q_W$):** Mapeamento projetivo inverso via $H_L^{-1}$ para o plano do piso ($Z_W = 0$), obtendo a posição métrica $(X_W, Y_W)$ em milímetros e a orientação $\psi$ em radianos.
6. **Análise Quantitativa e Visualização:** Exportação dos registros em `.csv`, geração do vídeo anotado (`.mp4`) e plotagem dos gráficos de evolução temporal (com desenrolamento de fase via `np.unwrap`), trajetória planar e taxa de detecções válidas.

---

## Estrutura do Repositório

```text
├── projeto_final.py              # Script principal do pipeline de visão computacional
├── requirements.txt                  # Dependências do projeto
├── video1.mp4                        # Vídeo de entrada 1
├── video2.mp4                        # Vídeo de entrada 2
├── Resultados_Projeto_video1/        # Vídeo anotado, CSV de poses e gráficos do Vídeo 1
└── Resultados_Projeto_video2/        # Vídeo anotado, CSV de poses e gráficos do Vídeo 2
```

---

## Como Executar

1. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure o vídeo de entrada:**
   No início do arquivo `projeto_final.py`, defina `CAMINHO_VIDEO` (`"video1.mp4"` ou `"video2.mp4"`) e o fator de resolução `ESCALA` (`1` para 4K nativo, `0.5` para Full HD ou `0.25` para qHD).

3. **Execute o script:**
   ```bash
   python projeto_final.py
   ```


Cada pasta `Resultados_Projeto_<nome_do_video>/` contém:
 
| Arquivo | Descrição |
|---------|-----------|
| `<video>_anotado.mp4` | Vídeo com contorno, centroide, seta de orientação e telemetria sobrepostos |
| `<video>_poses.csv` | Poses e indicadores de validade por quadro |
| `1_exemplos_intermediarios.png` | Três quadros ilustrando RGB → cromaticidade → máscara → pose |
| `2_pose_plano_imagem.png` | Gráficos temporais de `u_c`, `v_c` e `θ_I` |
| `3_pose_plano_mundo.png` | Gráficos temporais de `X_W`, `Y_W` e `ψ` |
| `4_trajetoria_mundo.png` | Trajetória `X_W × Y_W` com início, fim e setas de orientação |
| `5_validade_deteccoes.png` | Indicador temporal de validade das detecções |
 
## Requisitos
 
- Python 3.9+
- Dependências (em `requirements.txt`):
```
numpy>=1.24.0
opencv-python>=4.8.0
pandas>=2.0.0
matplotlib>=3.7.0
```
 
## Como executar
 
1. Crie e ative um ambiente virtual (opcional, recomendado) e instale as dependências:
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
 
2. Coloque o vídeo na mesma pasta do script e ajuste a variável `CAMINHO_VIDEO` no topo de `projeto_final.py`:
```python
CAMINHO_VIDEO = "video1.mp4"   # ou "video2.mp4"
```
 
   > **Atenção:** os arquivos originais têm extensão `.MP4` (maiúscula). Em Linux/macOS o nome é sensível a maiúsculas, então use exatamente o nome do arquivo (por exemplo `"video1.MP4"`) ou renomeie o arquivo.
 
3. Execute:
```bash
python projeto_final.py
```
 
4. Repita o passo 2 e 3 para o outro vídeo. A pasta de saída é criada automaticamente com base no nome do vídeo.
O script imprime no terminal as propriedades do vídeo (etapa 1) e o resumo quantitativo (etapa 6), além de exibir os gráficos com `plt.show()`. Em ambientes sem interface gráfica, use `MPLBACKEND=Agg python projeto_final.py`; as figuras continuam sendo salvas em disco.
 
### Parâmetros principais
 
Todos ficam no início de `projeto_final.py`:
 
| Parâmetro | Valor | Função |
|-----------|-------|--------|
| `ESCALA` | `1` | Fator de escala espacial (1.0 = 4K nativo, 0.5 = Full HD, 0.25 = qHD). Reduzir acelera o processamento |
| `HL` | matriz 3×3 | Homografia piso → imagem, calibrada em 3840×2160 |
| `LIMIAR_CROMATICIDADE` | `0.09` | Limiar sobre `r − max(g, b)` na segmentação |
| `CIRCULARIDADE_MIN` | `0.40` | Rejeita formas alongadas/côncavas (ex.: cruz vermelha no piso, C ≈ 0.31) |
| `AREA_MIN_4K` | `600` px | Descarta ruídos pequenos (escalada com `ESCALA²`) |
| `DIST_MAX_GATE` | `800 · ESCALA` px | Deslocamento máximo aceito entre quadros consecutivos |
| `RAIO_ROI` | `900 · ESCALA` px | Meia-largura da janela de busca ao redor da última posição |
| `THETA_INICIAL_REF` | `0 rad` | Referência para resolver a ambiguidade de 180° no primeiro quadro |
 


 
