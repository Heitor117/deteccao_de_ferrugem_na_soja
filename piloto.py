"""
PILOTO - Deteccao de ferrugem na soja (clorose / necrose)
Pipeline: mascara da folha -> SLIC -> features CIELab por superpixel -> k-means (k=3)

Uso:
    python piloto.py                                  # 6 folhas de images_base/detached_leaves
    python piloto.py --pasta images_base/detached_leaves --n 8 --segmentos 600
    python piloto.py --imagem images_base/detached_leaves/001_1.jpg

Saidas (pasta saida_piloto/): figura por folha + resumo_piloto.csv
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
from skimage.segmentation import slic, mark_boundaries
from sklearn.cluster import KMeans

COR_CLASSE = {"saudavel": (0, 160, 0), "clorose": (255, 215, 0), "necrose": (139, 69, 19)}


def redimensionar(img, lado_max=900):
    """Reduz a imagem (acelera o SLIC). Mantem proporcao."""
    h, w = img.shape[:2]
    f = lado_max / max(h, w)
    if f >= 1:
        return img
    return cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)


def mascara_folha(img_bgr, limiar_branco=235):
    """Fundo branco -> folha = pixels que nao sao quase brancos.
    Mantem o maior componente conexo e preenche buracos pequenos."""
    nao_branco = (img_bgr.min(axis=2) < limiar_branco).astype(np.uint8)
    nao_branco = cv2.morphologyEx(nao_branco, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    nao_branco = cv2.morphologyEx(nao_branco, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, rot, stats, _ = cv2.connectedComponentsWithStats(nao_branco)
    if n <= 1:
        return nao_branco.astype(bool)
    maior = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return rot == maior


def gerar_superpixels(img_rgb, mascara, n_segmentos=600, compactness=10):
    """SLIC no espaco Lab (skimage converte internamente). Superpixels so dentro da folha."""
    seg = slic(
        img_rgb,
        n_segments=n_segmentos,
        compactness=compactness,
        start_label=1,
        mask=mascara,
        convert2lab=True,
    )
    return seg


def features_lab(img_rgb, seg):
    """Media e desvio de L*, a*, b* por superpixel (Lab do OpenCV, escala 0-255)."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ids = np.unique(seg)
    ids = ids[ids > 0]
    feats = np.zeros((len(ids), 6), np.float32)
    areas = np.zeros(len(ids), np.int64)
    for i, sid in enumerate(ids):
        px = lab[seg == sid]
        feats[i, :3] = px.mean(axis=0)
        feats[i, 3:] = px.std(axis=0)
        areas[i] = len(px)
    return ids, feats, areas


def agrupar_e_nomear(feats, k=6, seed=0, peso_L=0.3, z_lesao=2.0):
    """k-means com k > 3 sobre (L*, a*, b*) dos superpixels; depois cada GRUPO e
    classificado pela distancia ao padrao saudavel da propria folha.
    (k=3 direto falha: o k-means divide a folha sadia, que e o grupo dominante.)

      referencia saudavel = mediana dos superpixels da folha (maioria e sadia)
      escala robusta      = MAD de cada canal (nao e puxada pelas lesoes)
      z = distancia do centro do grupo a referencia, em unidades de MAD
      z <= z_lesao                -> saudavel
      z  > z_lesao e a* - b* > 0  -> necrose (a* sobe, b* cai)
      z  > z_lesao e a* - b* < 0  -> clorose (b* sobe, a* cai)
    Retorna labels por superpixel, dict {cluster: nome}, centros em Lab.
    """
    pesos = np.array([peso_L, 1.0, 1.0], np.float32)
    X = feats[:, :3]
    ref = np.median(X, axis=0)
    mad = np.median(np.abs(X - ref), axis=0) * 1.4826 + 1e-6
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X * pesos)
    cent = km.cluster_centers_ / pesos
    d = (cent - ref) / mad
    z = np.sqrt(((d * pesos) ** 2).sum(axis=1))
    nomes = {}
    for c in range(k):
        if z[c] <= z_lesao:
            nomes[c] = "saudavel"
        elif (cent[c, 1] - ref[1]) - (cent[c, 2] - ref[2]) > 0:
            nomes[c] = "necrose"
        else:
            nomes[c] = "clorose"
    return km.labels_, nomes, cent


def processar(caminho, saida, n_segmentos, compactness, lado_max, z_lesao=2.0, k=6):
    bgr = redimensionar(cv2.imread(caminho), lado_max)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mask = mascara_folha(bgr)
    seg = gerar_superpixels(rgb, mask, n_segmentos, compactness)
    ids, feats, areas = features_lab(rgb, seg)
    labels, nomes, cent = agrupar_e_nomear(feats, k=k, z_lesao=z_lesao)

    mapa = np.zeros(rgb.shape, np.uint8) + 255
    area_cls = {"saudavel": 0, "clorose": 0, "necrose": 0}
    for sid, lab_c, a in zip(ids, labels, areas):
        nome = nomes[lab_c]
        mapa[seg == sid] = COR_CLASSE[nome]
        area_cls[nome] += int(a)
    total = max(sum(area_cls.values()), 1)
    severidade = 100.0 * (area_cls["clorose"] + area_cls["necrose"]) / total

    base = os.path.splitext(os.path.basename(caminho))[0]
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    ax[0].imshow(rgb); ax[0].set_title("Original")
    ax[1].imshow(mark_boundaries(rgb, seg, color=(1, 1, 0))); ax[1].set_title(f"SLIC ({len(ids)} superpixels)")
    ax[2].imshow(mapa); ax[2].set_title(f"k-means (k=6 -> 3 classes) | severidade={severidade:.1f}%")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", default="images_base/detached_leaves")
    ap.add_argument("--imagem", default=None)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--segmentos", type=int, default=4000)
    ap.add_argument("--compactness", type=float, default=10)
    ap.add_argument("--lado_max", type=int, default=900)
    ap.add_argument("--z", type=float, default=2.0, help="limiar de anomalia (em MADs)")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--saida", default="saida_piloto")
    a = ap.parse_args()

    os.makedirs(a.saida, exist_ok=True)
    if a.imagem:
        arquivos = [a.imagem]
    else:
        todos = sorted(glob.glob(os.path.join(a.pasta, "*.jpg")))
        passo = max(len(todos) // a.n, 1)
        arquivos = todos[::passo][: a.n]  # amostra espalhada pela pasta

    linhas = [processar(f, a.saida, a.segmentos, a.compactness, a.lado_max, a.z, a.k) for f in arquivos]
    df = pd.DataFrame(linhas).round(2)
    df.to_csv(os.path.join(a.saida, "resumo_piloto.csv"), index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()