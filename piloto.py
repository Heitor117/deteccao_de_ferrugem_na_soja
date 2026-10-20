"""
PILOTO v2 - Deteccao de ferrugem na soja (clorose / necrose)

Pipeline:
  imagem -> mascara da folha (sem caule e sem borda) -> superpixels SLIC
  -> features em CIELab por superpixel (matiz, luminosidade, croma)
  -> classificacao: saudavel / clorose / necrose -> severidade (%)

Por que MATIZ (hue) e nao a distancia a cor mediana da folha (versao 1):
  * sombra escurece a folha mas quase nao muda o matiz -> deixa de virar "necrose";
  * em folhas muito doentes (maioria amarela ou escura) a mediana da folha ja e doenca,
    entao compara-la com "o normal da folha" falha. O matiz tem referencia absoluta:
        verde ~105-120 graus | amarelo (clorose) ~85-100 | marrom (necrose) < ~80
  * manchas pequenas diluem na media do superpixel; por isso ha tambem um teste de
    CONTRASTE LOCAL: matiz bem abaixo do matiz dos vizinhos = lesao.

Uso:
    python piloto.py                                   # 6 folhas de images_base/detached_leaves
    python piloto.py --n 12
    python piloto.py --imagem images_base/detached_leaves/001_1.jpg
    python piloto.py --metodo kmeans                   # agrupa com k-means e nomeia os grupos

Saidas (pasta saida_piloto/): uma figura por folha + resumo_piloto.csv
"""
import argparse
import glob
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from skimage.segmentation import mark_boundaries, slic
from sklearn.cluster import KMeans

COR_CLASSE = {
    "saudavel": (0, 160, 0),
    "clorose": (255, 215, 0),
    "necrose": (139, 69, 19),
    "ignorar": (255, 255, 255),
}

# Limiares (ajustados olhando ~10 folhas; valide com a sua anotacao do Labelme)
LIMIAR_BRANCO = 235  # pixel com min(R,G,B) >= isso e fundo branco
HUE_NECROSE = 80.0  # matiz Lab (graus) abaixo disso -> necrose (marrom)
HUE_CLOROSE = 100.0  # matiz abaixo disso (e acima do anterior) -> clorose (amarelo)
L_NECROSE = 55.0  # L* (escala OpenCV 0-255) abaixo disso -> tecido muito escuro = necrose
T_LOCAL = 12.0  # queda de matiz (graus) em relacao aos vizinhos -> lesao pequena
RAIO_LOCAL = 60  # raio (px) da vizinhanca para o contraste local


# --------------------------------------------------------------------------------------
# Imagem e mascara da folha
# --------------------------------------------------------------------------------------
def redimensionar(img, lado_max=900):
    """Reduz a imagem (acelera o SLIC). Mantem a proporcao."""
    h, w = img.shape[:2]
    f = lado_max / max(h, w)
    if f >= 1:
        return img
    return cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)


def _elipse(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def mascara_folha(img_bgr, frac_caule=0.022, frac_borda=0.007):
    """Fundo branco -> folha. Depois:
      1) abertura grande remove o CAULE (pecíolo), que e fino; cantos arredondados da
         folha que a abertura comeu sao recuperados perto do corpo da folha;
      2) erosao pequena tira a BORDA (mistura com o branco e sombra do recorte).
    Retorna a mascara booleana da folha interna."""
    s = max(img_bgr.shape[:2])
    nao_branco = (img_bgr.min(axis=2) < LIMIAR_BRANCO).astype(np.uint8)
    nao_branco = cv2.morphologyEx(nao_branco, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    nao_branco = cv2.morphologyEx(nao_branco, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    r = max(int(s * frac_caule), 2)
    aberta = cv2.morphologyEx(nao_branco, cv2.MORPH_OPEN, _elipse(r))
    perto = cv2.dilate(aberta, _elipse(max(r // 2, 1))).astype(bool)
    folha = aberta.astype(bool) | (nao_branco.astype(bool) & perto)

    n, rot, stats, _ = cv2.connectedComponentsWithStats(folha.astype(np.uint8))
    if n > 1:  # fica so com o maior pedaco conectado
        folha = rot == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    b = max(int(s * frac_borda), 2)
    return cv2.erode(folha.astype(np.uint8), _elipse(b)).astype(bool)


# --------------------------------------------------------------------------------------
# Superpixels e features
# --------------------------------------------------------------------------------------
def gerar_superpixels(img_rgb, mascara, n_segmentos=4000, compactness=10):
    """SLIC (skimage converte para Lab internamente). Superpixels so dentro da folha."""
    return slic(
        img_rgb,
        n_segments=n_segmentos,
        compactness=compactness,
        start_label=1,
        mask=mascara,
        convert2lab=True,
    )


def features_superpixel(img_rgb, seg):
    """Por superpixel: media de L*, a*, b* (escala OpenCV 0-255), area e centroide (x, y)."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
    n = int(seg.max()) + 1
    plano = seg.ravel()
    cont = np.bincount(plano, minlength=n).astype(np.float64)
    ys, xs = np.indices(seg.shape)
    ids = np.where(cont > 0)[0]
    ids = ids[ids > 0]
    medias = [np.bincount(plano, weights=lab[..., c].ravel(), minlength=n)[ids] / cont[ids] for c in range(3)]
    cx = np.bincount(plano, weights=xs.ravel(), minlength=n)[ids] / cont[ids]
    cy = np.bincount(plano, weights=ys.ravel(), minlength=n)[ids] / cont[ids]
    return ids, np.stack(medias, axis=1), cont[ids].astype(int), np.stack([cx, cy], axis=1)


def derivar(F, C, raio=RAIO_LOCAL):
    """De L*a*b* medio para: matiz (graus no plano a*-b*), L*, croma, e o contraste local
    (matiz e L* menos a mediana dos superpixels vizinhos dentro do raio)."""
    a, b = F[:, 1] - 128.0, F[:, 2] - 128.0
    hue = np.degrees(np.arctan2(b, a))
    L = F[:, 0]
    croma = np.hypot(a, b)
    arvore = cKDTree(C)
    ref = np.array([np.median(np.c_[hue, L][arvore.query_ball_point(c, raio)], axis=0) for c in C])
    return hue, L, croma, hue - ref[:, 0], L - ref[:, 1]


# --------------------------------------------------------------------------------------
# Classificacao
# --------------------------------------------------------------------------------------
def nomear(h, L, c, dh, dL, hue_necrose=HUE_NECROSE, hue_clorose=HUE_CLOROSE, l_necrose=L_NECROSE, t_local=T_LOCAL):
    """Regra de nomeacao (vale para um superpixel ou para o centro de um grupo do k-means)."""
    if L > 185 and c < 14:  # residuo branco / brilho: ignorar
        return "ignorar"
    if h < hue_necrose or L < l_necrose:  # marrom ou muito escuro
        return "necrose"
    if dh < -t_local:  # matiz caiu em relacao aos vizinhos: lesao pequena
        return "necrose" if dL < 0 else "clorose"  # mais escura = marrom; senao amarela
    if h < hue_clorose:  # amarelado
        return "clorose"
    return "saudavel"


def classificar(F, C, metodo="regra", k=8, raio=RAIO_LOCAL, **lim):
    hue, L, croma, dh, dL = derivar(F, C, raio)
    X = np.stack([hue, L, croma, dh, dL], axis=1)
    if metodo == "regra":
        return [nomear(*x, **lim) for x in X]
    # k-means: agrupa superpixels parecidos e nomeia cada grupo pelo seu centro
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9) * np.array([1.5, 0.7, 0.5, 1.2, 0.7])
    rot = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs).labels_
    nome_grupo = {g: nomear(*X[rot == g].mean(axis=0), **lim) for g in np.unique(rot)}
    return [nome_grupo[g] for g in rot]


# --------------------------------------------------------------------------------------
# Execucao
# --------------------------------------------------------------------------------------
def processar(caminho, saida, n_segmentos, compactness, lado_max, metodo, k, lim):
    img = cv2.imread(caminho)
    if img is None:
        raise FileNotFoundError(f"Nao consegui abrir a imagem: {caminho}")
    bgr = redimensionar(img, lado_max)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mask = mascara_folha(bgr)
    seg = gerar_superpixels(rgb, mask, n_segmentos, compactness)
    ids, F, areas, C = features_superpixel(rgb, seg)
    nomes = classificar(F, C, metodo=metodo, k=k, **lim)

    mapa = np.full(rgb.shape, 255, np.uint8)
    area_cls = {"saudavel": 0, "clorose": 0, "necrose": 0, "ignorar": 0}
    cor_por_id = np.zeros((seg.max() + 1, 3), np.uint8)
    for sid, nome, a in zip(ids, nomes, areas):
        cor_por_id[sid] = COR_CLASSE[nome]
        area_cls[nome] += int(a)
    mapa = np.where((seg > 0)[..., None], cor_por_id[seg], mapa)

    total = max(area_cls["saudavel"] + area_cls["clorose"] + area_cls["necrose"], 1)
    severidade = 100.0 * (area_cls["clorose"] + area_cls["necrose"]) / total

    base = os.path.splitext(os.path.basename(caminho))[0]
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    ax[0].imshow(rgb)
    ax[0].set_title("Original")
    ax[1].imshow(mark_boundaries(rgb, seg, color=(1, 1, 0)))
    ax[1].set_title(f"SLIC ({len(ids)} superpixels)")
    ax[2].imshow(mapa)
    ax[2].set_title(f"{metodo} | severidade={severidade:.1f}%")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(saida, f"{base}.png"), dpi=110)
    plt.close(fig)

    return {
        "imagem": base,
        "superpixels": len(ids),
        "pct_saudavel": 100 * area_cls["saudavel"] / total,
        "pct_clorose": 100 * area_cls["clorose"] / total,
        "pct_necrose": 100 * area_cls["necrose"] / total,
        "severidade_%": severidade,
    }


def listar_imagens(pasta):
    if not os.path.isdir(pasta):
        raise SystemExit(f"ERRO: a pasta '{pasta}' nao existe. Confira o nome (use --pasta) e rode na raiz do projeto.")
    arquivos = []
    for padrao in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        arquivos += glob.glob(os.path.join(pasta, padrao))
    arquivos = sorted(set(arquivos))
    if not arquivos:
        raise SystemExit(f"ERRO: nenhuma imagem (.jpg/.png) encontrada em '{pasta}'.")
    return arquivos


def main():
    ap = argparse.ArgumentParser(description="Piloto: SLIC + CIELab (matiz) para clorose/necrose da ferrugem da soja")
    ap.add_argument("--pasta", default="image_base/detached_leaves")
    ap.add_argument("--imagem", default=None, help="processa so esta imagem")
    ap.add_argument("--n", type=int, default=6, help="quantas folhas da pasta (amostra espalhada)")
    ap.add_argument("--segmentos", type=int, default=4000)
    ap.add_argument("--compactness", type=float, default=10)
    ap.add_argument("--lado_max", type=int, default=900)
    ap.add_argument("--metodo", choices=["regra", "kmeans"], default="regra")
    ap.add_argument("--k", type=int, default=8, help="grupos do k-means (so com --metodo kmeans)")
    ap.add_argument("--hue_necrose", type=float, default=HUE_NECROSE)
    ap.add_argument("--hue_clorose", type=float, default=HUE_CLOROSE)
    ap.add_argument("--l_necrose", type=float, default=L_NECROSE)
    ap.add_argument("--t_local", type=float, default=T_LOCAL)
    ap.add_argument("--saida", default="saida_piloto")
    a = ap.parse_args()

    if a.imagem:
        if not os.path.isfile(a.imagem):
            raise SystemExit(f"ERRO: imagem '{a.imagem}' nao encontrada.")
        arquivos = [a.imagem]
    else:
        todos = listar_imagens(a.pasta)
        passo = max(len(todos) // a.n, 1)
        arquivos = todos[::passo][: a.n]

    os.makedirs(a.saida, exist_ok=True)
    lim = dict(hue_necrose=a.hue_necrose, hue_clorose=a.hue_clorose, l_necrose=a.l_necrose, t_local=a.t_local)
    linhas = []
    for f in arquivos:
        linhas.append(processar(f, a.saida, a.segmentos, a.compactness, a.lado_max, a.metodo, a.k, lim))
        print(f"ok: {os.path.basename(f)}  severidade={linhas[-1]['severidade_%']:.1f}%", flush=True)

    df = pd.DataFrame(linhas).round(2)
    df.to_csv(os.path.join(a.saida, "resumo_piloto.csv"), index=False)
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()