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
├── rastreamento_pose.py              # Script principal do pipeline de visão computacional
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
   No início do arquivo `rastreamento_pose.py`, defina `CAMINHO_VIDEO` (`"video1.mp4"` ou `"video2.mp4"`) e o fator de resolução `ESCALA` (`1` para 4K nativo, `0.5` para Full HD ou `0.25` para qHD).

3. **Execute o script:**
   ```bash
   python rastreamento_pose.py
   ```
