"""
COMPARAR - Experimento B: algoritmo de superpixel x espaco de cor, avaliados contra a anotacao.

Configuracoes testadas (cada uma CALIBRADA nas folhas de treino e AVALIADA nas de teste):
    segmentacao: SLIC | Felzenszwalb | Watershed compacto | Sem superpixel (pixel a pixel)
    espaco de cor: CIELab (matiz no plano a*-b*, L*) | HSV (H, V)
Regra de classificacao igual a do metodo final (contraste local desligado):
    necrose se matiz < limiar_necrose ou brilho < limiar_brilho; clorose se matiz < limiar_clorose; senao saudavel.

Uso (raiz do projeto, venv ativa, depois de labelme_para_mascaras.py):
    python comparar.py
    python comparar.py --teste 001_1,001_2,001_3,001_4,012_3,012_4,016_3,016_6   (este e o padrao)

Saidas (resultados/comparacao/):
    comparacao.csv          uma linha por configuracao: limiares escolhidos, F1/IoU por classe e erro (treino e teste),
                            n medio de segmentos e tempo medio de segmentacao por folha
    fig_comparacao.png      F1 medio e erro da severidade no TESTE, por configuracao
    fig_segmentacoes.png    recorte de uma folha com as fronteiras de cada algoritmo
    feat_*.npz              cache (use --refazer se mudar parametros)
"""
import argparse
import glob
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, mark_boundaries, relabel_sequential, slic, watershed

import piloto as P

METODOS = ["slic", "felzenszwalb", "watershed", "pixel"]
NOME_METODO = {"slic": "SLIC", "felzenszwalb": "Felzenszwalb", "watershed": "Watershed compacto",
               "pixel": "Sem superpixel"}
ESPACOS = ["lab", "hsv"]
NOME_ESPACO = {"lab": "CIELab", "hsv": "HSV"}
# grades de limiares (matiz em graus; brilho na escala 0-255: L* do OpenCV ou V do HSV)
GRADE = {
    "lab": dict(hc=np.arange(86, 108, 2), hn=[60, 65, 70, 75, 80, 85], bn=[45, 55, 65]),
    "hsv": dict(hc=np.arange(44, 92, 4), hn=[16, 20, 24, 28, 32, 36, 40], bn=[45, 55, 65, 80]),
}
TESTE_PADRAO = "001_1,001_2,001_3,001_4,012_3,012_4,016_3,016_6"


def achar_imagem(pasta, base):
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        f = os.path.join(pasta, base + ext)
        if os.path.isfile(f):
            return f
    raise FileNotFoundError(f"imagem '{base}' nao encontrada em {pasta}")


def segmentar(metodo, rgb, mask, cfg):
    if metodo == "slic":
        s = slic(rgb, n_segments=cfg["segmentos"], compactness=10, start_label=1, mask=mask, convert2lab=True)
    elif metodo == "felzenszwalb":
        s = felzenszwalb(rgb, scale=cfg["fz_scale"], sigma=0.5, min_size=cfg["fz_min"]) + 1
        s[~mask] = 0
    elif metodo == "watershed":
        s = watershed(sobel(rgb2gray(rgb)), markers=cfg["segmentos"], compactness=0.001, mask=mask)
    else:  # pixel a pixel
        s = np.zeros(mask.shape, np.int64)
        s[mask] = np.arange(1, int(mask.sum()) + 1)
    return relabel_sequential(s.astype(np.int64))[0]


def extrair(args):
    base, metodo, pasta_img, pasta_mask, saida, cfg = args
    f = os.path.join(saida, f"feat_{metodo}_{base}.npz")
    f_mask = os.path.join(pasta_mask, base + ".png")
    # o cache guarda a contagem dos pixels anotados: se a mascara mudou depois dele, refaz
    if os.path.isfile(f) and not cfg["refazer"] and os.path.getmtime(f) > os.path.getmtime(f_mask):
        return base, metodo
    bgr = P.redimensionar(cv2.imread(achar_imagem(pasta_img, base)), cfg["lado_max"])
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mask = P.mascara_folha(bgr)
    t0 = time.time()
    seg = segmentar(metodo, rgb, mask, cfg)
    t_seg = time.time() - t0
    ids, F, areas, _ = P.features_superpixel(rgb, seg)  # media de L*a*b* (escala OpenCV) por segmento
    a, b = F[:, 1] - 128.0, F[:, 2] - 128.0
    n = int(seg.max()) + 1
    plano = seg.ravel()
    cont_px = np.bincount(plano, minlength=n)[ids].astype(np.float64)
    rgb_med = np.stack([np.bincount(plano, weights=rgb[..., c].ravel().astype(np.float64), minlength=n)[ids] / cont_px
                        for c in range(3)], axis=1)
    hsv = cv2.cvtColor(np.clip(rgb_med, 0, 255).astype(np.uint8)[:, None, :], cv2.COLOR_RGB2HSV)[:, 0, :].astype(np.float64)
    ann = cv2.imread(os.path.join(pasta_mask, base + ".png"), cv2.IMREAD_GRAYSCALE)
    ann = cv2.resize(ann, (seg.shape[1], seg.shape[0]), interpolation=cv2.INTER_NEAREST).ravel()
    cont = np.stack([np.bincount(plano[ann == k], minlength=n)[ids] for k in (1, 2, 3)], axis=1)
    np.savez_compressed(f, lab_h=np.degrees(np.arctan2(b, a)), lab_b=F[:, 0], lab_c=np.hypot(a, b),
                        hsv_h=hsv[:, 0] * 2.0, hsv_b=hsv[:, 2], hsv_c=hsv[:, 1],
                        areas=areas, cont=cont, n_seg=len(ids), t_seg=t_seg)
    return base, metodo


def classificar(d, esp, hc, hn, bn):
    h, br, c = d[f"{esp}_h"], d[f"{esp}_b"], d[f"{esp}_c"]
    cls = np.ones(len(h), np.int8)
    cls[h < hc] = 2
    cls[(h < hn) | (br < bn)] = 3
    cls[(br > 185) & (c < 14)] = 0  # brilho/residuo branco: fora da conta
    return cls


def avaliar(dados, esp, lim):
    conf = np.zeros((3, 3))
    erros = []
    for d in dados:
        cls = classificar(d, esp, **lim)
        for j in (1, 2, 3):
            conf[:, j - 1] += d["cont"][cls == j].sum(axis=0)
        v = cls > 0
        sev_p = 100 * d["areas"][v & (cls > 1)].sum() / max(d["areas"][v].sum(), 1)
        c = d["cont"][v]
        sev_a = 100 * (c[:, 1] + c[:, 2]).sum() / max(c.sum(), 1)
        erros.append(abs(sev_p - sev_a))
    out = {}
    for i, nm in enumerate(("saudavel", "clorose", "necrose")):
        tp, fp, fn = conf[i, i], conf[:, i].sum() - conf[i, i], conf[i, :].sum() - conf[i, i]
        out[f"F1_{nm}"] = 2 * tp / max(2 * tp + fp + fn, 1)
        out[f"IoU_{nm}"] = tp / max(tp + fp + fn, 1)
    out["F1_medio"] = float(np.mean([out["F1_saudavel"], out["F1_clorose"], out["F1_necrose"]]))
    out["MAE_severidade"] = float(np.mean(erros))
    return out


def fig_segmentacoes(base, pasta_img, cfg, destino):
    bgr = P.redimensionar(cv2.imread(achar_imagem(pasta_img, base)), cfg["lado_max"])
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mask = P.mascara_folha(bgr)
    ys, xs = np.where(mask)
    cy, cx = int(ys.mean()), int(xs.mean())
    r = 110  # recorte central para as fronteiras ficarem visiveis
    y0, x0 = max(cy - r, 0), max(cx - r, 0)
    fig, axs = plt.subplots(1, 4, figsize=(6.9, 1.9))
    for ax, m in zip(axs, METODOS):
        if m == "pixel":
            ax.imshow(rgb[y0:y0 + 2 * r, x0:x0 + 2 * r])
        else:
            seg = segmentar(m, rgb, mask, cfg)
            ax.imshow(mark_boundaries(rgb, seg, color=(1, 1, 0))[y0:y0 + 2 * r, x0:x0 + 2 * r])
        ax.set_title(NOME_METODO[m], fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(destino, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Experimento B: superpixel x espaco de cor")
    ap.add_argument("--imagens", default=None)
    ap.add_argument("--mascaras", default="anotacoes/mascaras")
    ap.add_argument("--saida", default="resultados/comparacao")
    ap.add_argument("--teste", default=TESTE_PADRAO)
    ap.add_argument("--segmentos", type=int, default=4000)
    ap.add_argument("--fz_scale", type=float, default=10)  # ~4000 segmentos, como o SLIC
    ap.add_argument("--fz_min", type=int, default=20)
    ap.add_argument("--lado_max", type=int, default=900)
    ap.add_argument("--exemplo", default="016_3", help="folha usada na figura das segmentacoes")
    ap.add_argument("--refazer", action="store_true")
    ap.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    a = ap.parse_args()

    pasta_img = a.imagens
    if not pasta_img:
        for p in ("image_base/detached_leaves", "images_base/detached_leaves", "imagens_base/detached_leaves"):
            if os.path.isdir(p):
                pasta_img = p
                break
    if not pasta_img or not os.path.isdir(pasta_img):
        raise SystemExit("ERRO: pasta das imagens nao encontrada. Use --imagens image_base/detached_leaves")
    bases = sorted(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(a.mascaras, "*.png")))
    if not bases:
        raise SystemExit(f"ERRO: nenhuma mascara em '{a.mascaras}'. Rode labelme_para_mascaras.py antes.")
    teste = [x.strip() for x in a.teste.split(",") if x.strip() and x.strip() in bases]
    treino = [b for b in bases if b not in teste]
    if not teste or not treino:
        raise SystemExit("ERRO: precisa de folhas anotadas no treino e no teste (confira --teste).")

    os.makedirs(a.saida, exist_ok=True)
    cfg = dict(segmentos=a.segmentos, fz_scale=a.fz_scale, fz_min=a.fz_min, lado_max=a.lado_max, refazer=a.refazer)
    tarefas = [(b, m, pasta_img, a.mascaras, a.saida, cfg) for m in METODOS for b in bases]
    print(f"Folhas: treino {len(treino)}, teste {len(teste)}. Segmentando {len(tarefas)} combinacoes folha x metodo ...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (b, m) in enumerate(ex.map(extrair, tarefas), 1):
            if i % 11 == 0 or i == len(tarefas):
                print(f"  {i}/{len(tarefas)} ({time.time() - t0:.0f}s)", flush=True)

    linhas = []
    for m in METODOS:
        def carregar(lista):
            return [dict(np.load(os.path.join(a.saida, f"feat_{m}_{b}.npz"))) for b in lista]
        D_tr, D_te = carregar(treino), carregar(teste)
        n_seg = np.mean([float(d["n_seg"]) for d in D_tr + D_te])
        t_seg = np.mean([float(d["t_seg"]) for d in D_tr + D_te])
        for esp in ESPACOS:
            g = GRADE[esp]
            melhor, melhor_f1 = None, -1
            for hc, hn, bn in itertools.product(g["hc"], g["hn"], g["bn"]):
                if hn >= hc:
                    continue
                lim = dict(hc=float(hc), hn=float(hn), bn=float(bn))
                f1 = avaliar(D_tr, esp, lim)["F1_medio"]
                if f1 > melhor_f1:
                    melhor, melhor_f1 = lim, f1
            tr, te = avaliar(D_tr, esp, melhor), avaliar(D_te, esp, melhor)
            linhas.append({"segmentacao": NOME_METODO[m], "espaco_cor": NOME_ESPACO[esp],
                           "lim_matiz_clorose": melhor["hc"], "lim_matiz_necrose": melhor["hn"], "lim_brilho_necrose": melhor["bn"],
                           "n_segmentos_medio": round(n_seg), "tempo_seg_s": round(t_seg, 2),
                           **{f"treino_{k}": v for k, v in tr.items()}, **{f"teste_{k}": v for k, v in te.items()}})
            print(f"  {NOME_METODO[m]:<19} {NOME_ESPACO[esp]:<6} teste: F1 medio={te['F1_medio']:.3f}  "
                  f"F1 clorose={te['F1_clorose']:.3f}  erro sev={te['MAE_severidade']:.1f}", flush=True)
    res = pd.DataFrame(linhas).sort_values("teste_F1_medio", ascending=False)
    res.round(4).to_csv(os.path.join(a.saida, "comparacao.csv"), index=False)

    # figura: F1 medio e erro no teste
    rot = [f"{r.segmentacao}\n{r.espaco_cor}" for r in res.itertuples()]
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.9), sharey=True)
    y = np.arange(len(res))[::-1]
    axs[0].barh(y, res["teste_F1_medio"], color="#6E370A", zorder=3)
    axs[0].set_xlabel("F1 médio (teste)")
    axs[0].set_xlim(0, 1)
    axs[1].barh(y, res["teste_MAE_severidade"], color="#FF6E00", zorder=3)
    axs[1].set_xlabel("Erro médio da severidade (pontos %, teste)")
    axs[0].set_yticks(y, rot, fontsize=6)
    for ax in axs:
        ax.set_facecolor("#D9D9D9")
        ax.grid(axis="x", color="#7f7f7f", lw=0.6)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(a.saida, "fig_comparacao.png"), dpi=300)
    plt.close(fig)
    try:
        fig_segmentacoes(a.exemplo, pasta_img, cfg,
                         os.path.join(a.saida, "fig_segmentacoes.png"))
    except FileNotFoundError:
        pass

    pd.set_option("display.width", 220)
    cols = ["segmentacao", "espaco_cor", "n_segmentos_medio", "tempo_seg_s", "teste_F1_saudavel", "teste_F1_clorose",
            "teste_F1_necrose", "teste_F1_medio", "teste_MAE_severidade"]
    print("\nRanking no TESTE (calibrado no treino):")
    print(res[cols].round(3).to_string(index=False))
    print(f"\nPronto em {time.time() - t0:.0f}s -> {os.path.abspath(a.saida)}")


if __name__ == "__main__":
    main()