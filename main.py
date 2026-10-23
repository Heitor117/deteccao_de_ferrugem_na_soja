"""
MAIN (v3) - Ferrugem asiatica da soja: evolucao da clorose/necrose POR FOLIOLO, ao longo das capturas.

Nome da imagem = <folha>_<captura>   (ex.: 006_3 = folha 006, 3a captura, ~3 dias entre capturas).
Cada folha e analisada isoladamente: severidade de cada captura -> curva de evolucao -> AUDPC.

Um comando gera tudo (precisa do piloto.py na mesma pasta):
    python main.py
    python main.py --lang en            # rotulos em ingles (modelo do artigo)
    python main.py --folhas 006,016     # so estas folhas (teste rapido)
    python main.py --so_figuras         # nao reprocessa: refaz tabelas/figuras a partir de resultados.csv
    python main.py --refazer            # ignora o cache

Saidas (pasta resultados/):
    resultados.csv                      uma linha por imagem (folha, captura, % classes, severidade)
    tabelas/evolucao_por_folha.csv      por folha: n capturas, severidade inicial/final, AUDPC, captura >50%
    tabelas/severidade_por_captura.csv  media/mediana/desvio por numero da captura
    tabelas/anomalias_temporais.csv     quedas de severidade entre capturas (conferir a imagem!)
    tabelas/faixas_severidade.csv       contagem por faixa (cortes provisorios)
    resumo_estatistico.txt
    figuras/fig_evolucao_folhas.png     curvas de evolucao (uma linha por folha) + media
    figuras/fig_evolucao_paineis.png    um painel por folha, barras empilhadas necrose/clorose
    figuras/fig_barras_imagens.png      modelo 2.2: originais + resultado de UMA folha ao longo do tempo + barras de todas
    figuras/fig_ordenado.png            modelo 2.1: todas as imagens ordenadas por severidade
    figuras/fig_boxplot_captura.png     severidade por numero da captura
    figuras/fig_hist_severidade.png     histograma (faixas de 10 em 10, alinhadas ao eixo)
    figuras/fig_hist_matiz.png          histograma do matiz + limiares
    figuras/(dispersao/confusao)        so se existir anotacao (--anotacoes)
    imagens_processadas/*.png           original | SLIC | classificacao, por imagem
    cache/, parametros.txt

TEMPO: a data de cada captura vem do EXIF da foto (DateTimeOriginal, data em que a foto foi tirada).
'dia' = dias desde a 1a sessao de fotos; 'captura' = numero da sessao (1, 2, 3...). O numero no nome do
arquivo NAO e usado como tempo (nas folhas 010 e 011 ele esta fora da ordem cronologica). Duas fotos da
mesma folha na mesma sessao viram uma media no tempo e entram em tabelas/repeticoes_mesma_sessao.csv.
Se as imagens nao tiverem EXIF, usa o numero do arquivo com --dias entre capturas (com aviso).
"""
import argparse
import glob
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.segmentation import mark_boundaries

import piloto as P

IDX = {"saudavel": 1, "clorose": 2, "necrose": 3}  # mapa de classes: 0 = fundo
COR_RGB = {1: (0, 220, 0), 2: (255, 110, 0), 3: (110, 55, 10)}  # saudavel verde, clorose laranja, necrose marrom
COR_HEX = {"saudavel": "#00DC00", "clorose": "#FF6E00", "necrose": "#6E370A"}
CINZA = "#D9D9D9"

TXT = {
    "pt": dict(necrose="Necrose", clorose="Clorose", saudavel="Saudável", folhas="Folíolos (números = dias após a 1ª captura)",
               sev="Severidade (%)", n="Número de imagens", cap="Captura", dias="Dias após a 1ª captura", dia="Dia", matiz="Ângulo de matiz no plano a*-b* (graus)",
               sp="Superpixels (ponderado pela área)", orig="Original", res="Resultado", pred="Severidade prevista (%)",
               anot="Severidade anotada (%)", real="Anotado", prev="Previsto", media="Média", folha="Folíolo",
               mediana="mediana", ordenadas="Imagens ordenadas por severidade"),
    "en": dict(necrose="Necrosis", clorose="Chlorosis", saudavel="Healthy", folhas="Leaflets (numbers = days after first capture)",
               sev="Severity (%)", n="Number of images", cap="Capture", dias="Days after first capture", dia="Day", matiz="Hue angle in the a*-b* plane (degrees)",
               sp="Superpixels (area-weighted)", orig="Original", res="Result", pred="Predicted severity (%)",
               anot="Annotated severity (%)", real="Annotated", prev="Predicted", media="Mean", folha="Leaflet",
               mediana="median", ordenadas="Images sorted by severity"),
}


# --------------------------------------------------------------------------------------
# Processamento de uma imagem (em paralelo, com cache)
# --------------------------------------------------------------------------------------
def analisar(args):
    caminho, saida, cfg = args
    base = os.path.splitext(os.path.basename(caminho))[0]
    cache = os.path.join(saida, "cache")
    f_cls, f_sp = os.path.join(cache, base + "_cls.png"), os.path.join(cache, base + "_sp.npz")
    if not cfg["refazer"] and os.path.isfile(f_cls) and os.path.isfile(f_sp):
        cls = cv2.imread(f_cls, cv2.IMREAD_GRAYSCALE)
        nsp = len(np.load(f_sp)["hue"])
    else:
        img = cv2.imread(caminho)
        if img is None:
            raise FileNotFoundError(caminho)
        bgr = P.redimensionar(img, cfg["lado_max"])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = P.mascara_folha(bgr)
        if not mask.any():
            raise ValueError(f"mascara da folha vazia em {caminho}")
        seg = P.gerar_superpixels(rgb, mask, cfg["segmentos"], cfg["compactness"])
        ids, F, areas, C = P.features_superpixel(rgb, seg)
        lim = dict(hue_necrose=cfg["hue_necrose"], hue_clorose=cfg["hue_clorose"],
                   l_necrose=cfg["l_necrose"], t_local=cfg["t_local"])
        nomes = P.classificar(F, C, metodo=cfg["metodo"], k=cfg["k"], **lim)
        tab = np.zeros(seg.max() + 1, np.uint8)  # "ignorar" -> 0 (fora da conta)
        for sid, nome in zip(ids, nomes):
            tab[sid] = IDX.get(nome, 0)
        cls = tab[seg]
        nsp = len(ids)
        hue = np.degrees(np.arctan2(F[:, 2] - 128, F[:, 1] - 128))
        cv2.imwrite(f_cls, cls)
        np.savez_compressed(f_sp, hue=hue.astype(np.float32), area=areas.astype(np.int32))
        # painel de conferencia: original | SLIC | classificacao (fundo branco)
        mapa = np.full(rgb.shape, 255, np.uint8)
        for i, c in COR_RGB.items():
            mapa[cls == i] = c
        fig, ax = plt.subplots(1, 3, figsize=(16, 5))
        ax[0].imshow(rgb)
        ax[0].set_title("Original")
        ax[1].imshow(mark_boundaries(rgb, seg, color=(1, 1, 0)))
        ax[1].set_title(f"SLIC ({nsp} superpixels)")
        ax[2].imshow(mapa)
        n3 = {c: int((cls == i).sum()) for c, i in IDX.items()}
        sv = 100 * (n3["clorose"] + n3["necrose"]) / max(sum(n3.values()), 1)
        ax[2].set_title(f"{cfg['metodo']} | severidade={sv:.2f}%")
        for a in ax:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(saida, "imagens_processadas", base + ".png"), dpi=110)
        plt.close(fig)
    n = {c: int((cls == i).sum()) for c, i in IDX.items()}
    tot = max(sum(n.values()), 1)
    return {"imagem": base, "superpixels": nsp, "pixels_analisados": tot,
            "pct_saudavel": 100 * n["saudavel"] / tot, "pct_clorose": 100 * n["clorose"] / tot,
            "pct_necrose": 100 * n["necrose"] / tot, "severidade_%": 100 * (n["clorose"] + n["necrose"]) / tot,
            "metodo": cfg["metodo"]}


def achar_pasta(pasta):
    if pasta:
        return pasta
    for p in ("image_base/detached_leaves", "images_base/detached_leaves", "imagens_base/detached_leaves"):
        if os.path.isdir(p):
            return p
    raise SystemExit("ERRO: nao achei a pasta das imagens. Use --pasta (ex.: image_base/detached_leaves).")


def datas_exif(pasta, imagens):
    """Data/hora em que a foto foi TIRADA (EXIF DateTimeOriginal, tag 36867). NaT se nao houver.
    Obs.: a tag 306 (DateTime) e a data da ultima edicao (aqui, quando o fundo foi removido no GIMP)."""
    from PIL import Image
    out = {}
    for b in imagens:
        d = None
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
            f = os.path.join(pasta, b + ext)
            if os.path.isfile(f):
                try:
                    d = Image.open(f).getexif().get_ifd(0x8769).get(36867)
                except Exception:
                    d = None
                break
        out[b] = pd.to_datetime(d, format="%Y:%m:%d %H:%M:%S", errors="coerce") if d else pd.NaT
    return out


def separar_nome(df, dias, pasta):
    """folha = prefixo do nome; tempo = data EXIF da foto (fallback: numero do arquivo)."""
    df = df.drop(columns=[c for c in ("folha", "n_arquivo", "captura", "dia", "data", "repeticao") if c in df.columns])
    m = df["imagem"].str.extract(r"^(.*?)_(\d+)$")
    df["folha"] = m[0]
    df["n_arquivo"] = pd.to_numeric(m[1])
    if df["n_arquivo"].isna().any():
        raise SystemExit("ERRO: nome de imagem fora do padrao <folha>_<numero> (ex.: 006_3): "
                         + ", ".join(df.loc[df["n_arquivo"].isna(), "imagem"].head(5)))
    datas = datas_exif(pasta, df["imagem"]) if pasta else {}
    df["data"] = df["imagem"].map(datas) if datas else pd.NaT
    if datas and df["data"].notna().all():
        dia = df["data"].dt.normalize()
        df["dia"] = (dia - dia.min()).dt.days
        sessao = {d: i + 1 for i, d in enumerate(sorted(df["dia"].unique()))}
        df["captura"] = df["dia"].map(sessao)
        fonte = "EXIF (data em que a foto foi tirada)"
    else:
        print("  AVISO: sem data EXIF em todas as imagens -> tempo pelo numero do arquivo (--dias entre capturas)")
        df["captura"] = df["n_arquivo"]
        df["dia"] = (df["captura"] - 1) * dias
        fonte = f"numero do arquivo x {dias:g} dias"
    df["repeticao"] = df.duplicated(["folha", "captura"], keep=False)
    df["data"] = pd.to_datetime(df["data"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return df.sort_values(["folha", "dia", "imagem"]).reset_index(drop=True), fonte


def por_sessao(df):
    """Uma linha por folha e sessao (media das fotos repetidas na mesma sessao) - base da analise temporal."""
    num = ["pct_saudavel", "pct_clorose", "pct_necrose", "severidade_%"]
    g = df.groupby(["folha", "captura", "dia"])
    out = g[num].mean().reset_index()
    out["imagem"] = g["imagem"].agg("+".join).values
    out["n_fotos"] = g.size().values
    return out.sort_values(["folha", "dia"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Tabelas
# --------------------------------------------------------------------------------------
def tabela_evolucao(df):
    lin = []
    for f, g in df.groupby("folha"):
        g = g.sort_values("dia")
        t, y = g["dia"].to_numpy(float), g["severidade_%"].to_numpy(float)
        audpc = float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(t))) if len(g) > 1 else np.nan
        dur = t[-1] - t[0] if len(g) > 1 else np.nan
        acima = g.loc[g["severidade_%"] > 50, "dia"]
        lin.append({"folha": f, "n_capturas": len(g), "sev_inicial": y[0], "sev_final": y[-1],
                    "delta": y[-1] - y[0], "sev_max": y.max(), "AUDPC": audpc,
                    "sAUDPC_%": audpc / dur if dur else np.nan,
                    "1o_dia_acima_50%": int(acima.iloc[0]) if len(acima) else np.nan})
    return pd.DataFrame(lin).round(2)


def tabela_anomalias(df, queda):
    lin = []
    for f, g in df.groupby("folha"):
        g = g.sort_values("captura")
        for (_, a), (_, b) in zip(g.iloc[:-1].iterrows(), g.iloc[1:].iterrows()):
            if a["severidade_%"] - b["severidade_%"] >= queda:
                lin.append({"folha": f, "de": a["imagem"], "para": b["imagem"], "sev_de": a["severidade_%"],
                            "sev_para": b["severidade_%"], "queda_pontos": round(a["severidade_%"] - b["severidade_%"], 2)})
    return pd.DataFrame(lin, columns=["folha", "de", "para", "sev_de", "sev_para", "queda_pontos"])


# --------------------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------------------
def estilo():
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "savefig.dpi": 300})


def _cinza(ax, grade="y"):
    ax.set_facecolor(CINZA)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(axis=grade, color="#7f7f7f", lw=0.6)
    ax.set_axisbelow(True)


def _legenda(ax, t):
    from matplotlib.patches import Patch
    h = [Patch(color=COR_HEX["clorose"], label=t["clorose"]), Patch(color=COR_HEX["necrose"], label=t["necrose"])]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)


def _empilhar(ax, x, d, w=0.85):
    ax.bar(x, d["pct_necrose"], width=w, color=COR_HEX["necrose"], zorder=3)
    ax.bar(x, d["pct_clorose"], bottom=d["pct_necrose"], width=w, color=COR_HEX["clorose"], zorder=3)


def fig_evolucao(df, t, destino):
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    for f, g in df.groupby("folha"):
        ax.plot(g["dia"], g["severidade_%"], color="#555555", lw=0.8, alpha=0.55, marker="o", ms=2.5, zorder=3)
    m = df.groupby("dia")["severidade_%"].mean()
    ax.plot(m.index, m.values, color=COR_HEX["necrose"], lw=2.2, marker="s", ms=4, zorder=4, label=t["media"])
    _cinza(ax)
    ax.set_ylim(0, 100)
    ax.set_xticks(sorted(df["dia"].unique()))
    ax.set_xlabel(t["dias"])
    ax.set_ylabel(t["sev"])
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(destino)
    plt.close(fig)


def fig_paineis(df, t, destino, colunas=6):
    folhas = sorted(df["folha"].unique())
    lin = int(np.ceil(len(folhas) / colunas))
    fig, axs = plt.subplots(lin, colunas, figsize=(6.9, 1.15 * lin + 0.5), sharey=True, sharex=True, squeeze=False)
    dias_u = sorted(df["dia"].unique())
    for ax, f in zip(axs.ravel(), folhas):
        g = df[df["folha"] == f]
        _empilhar(ax, g["dia"], g, 2.2)
        _cinza(ax)
        ax.set_title(f, fontsize=7, pad=2)
        ax.set_ylim(0, 100)
        ax.set_xlim(dias_u[0] - 2, dias_u[-1] + 2)
        ax.set_xticks(dias_u)
        ax.tick_params(length=0, labelsize=6)
    for ax in axs.ravel()[len(folhas):]:
        ax.axis("off")
    for ax in axs[:, 0]:
        ax.set_ylabel("%")
    fig.supxlabel(t["dias"], fontsize=8)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=COR_HEX["clorose"], label=t["clorose"]), Patch(color=COR_HEX["necrose"], label=t["necrose"])],
               loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(h_pad=0.4, w_pad=0.3, rect=(0, 0.04, 1, 0.94))
    fig.savefig(destino)
    plt.close(fig)


def _escolher_exemplo(df, forcar):
    if forcar:
        return forcar
    r = df.groupby("folha")["severidade_%"].agg(n="size", amp=lambda s: s.max() - s.min())
    return r.sort_values(["n", "amp"], ascending=False).index[0]


def fig_barras_imagens(df, t, saida, pasta, destino, exemplo=None):
    """Modelo 2.2: linha de originais, linha de resultados (fundo preto) de UMA folha ao longo do tempo,
    e embaixo as barras empilhadas de todas as imagens, agrupadas por folha."""
    cache = os.path.join(saida, "cache")
    fx = _escolher_exemplo(df, exemplo)
    ex = df[df["folha"] == fx].sort_values(["dia", "imagem"])
    tem_cache = bool(pasta) and all(os.path.isfile(os.path.join(cache, i + "_cls.png")) for i in ex["imagem"])
    fig = plt.figure(figsize=(6.9, 5.2 if tem_cache else 2.8))
    gs = fig.add_gridspec(4 if tem_cache else 1, max(len(ex), 1),
                          height_ratios=[0.6, 0.6, 0.3, 1.5] if tem_cache else [1], hspace=0.12, wspace=0.04)
    if tem_cache:
        for j, (_, r) in enumerate(ex.iterrows()):
            cls = cv2.imread(os.path.join(cache, r["imagem"] + "_cls.png"), cv2.IMREAD_GRAYSCALE)
            orig = None
            for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
                orig = cv2.imread(os.path.join(pasta, r["imagem"] + ext))
                if orig is not None:
                    break
            orig = cv2.cvtColor(cv2.resize(orig, (cls.shape[1], cls.shape[0]), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            res = np.zeros(cls.shape + (3,), np.uint8)
            for i, c in COR_RGB.items():
                res[cls == i] = c
            ys, xs = np.where(cls > 0)  # recorta na folha (mesma caixa nas duas linhas)
            y0, y1, x0, x1 = max(ys.min() - 8, 0), ys.max() + 8, max(xs.min() - 8, 0), xs.max() + 8
            orig, res = orig[y0:y1, x0:x1], res[y0:y1, x0:x1]
            a1, a2 = fig.add_subplot(gs[0, j]), fig.add_subplot(gs[1, j])
            a1.imshow(orig)
            a2.imshow(res)
            a1.set_anchor("S")  # linhas coladas uma na outra
            a2.set_anchor("N")
            for a in (a1, a2):
                a.set_xticks([])
                a.set_yticks([])
                for s in a.spines.values():
                    s.set_visible(False)
            a2.set_xlabel(f"{t['dia']} {int(r['dia'])} · {r['severidade_%']:.1f}%", fontsize=7, labelpad=2)
            if j == 0:
                a1.set_ylabel(t["orig"], fontsize=7)
                a2.set_ylabel(t["res"], fontsize=7)
                a1.set_title(f"{t['folha']} {fx}", fontsize=7, loc="left", pad=2)
    ax = fig.add_subplot(gs[-1, :])
    x, pos, centros, rot = 0.0, [], [], []
    for f, g in df.groupby("folha"):
        g = g.sort_values(["dia", "imagem"])
        p = x + np.arange(len(g))
        pos += list(p)
        centros.append((f, p.mean()))
        rot += [int(v) for v in g["dia"]]
        x = p[-1] + 2
    d = df.sort_values(["folha", "dia", "imagem"])
    _empilhar(ax, np.array(pos), d, 0.9)
    _cinza(ax)
    ax.set_ylim(0, 100)
    ax.set_xlim(-1, x - 1)
    ax.set_xticks(pos)
    ax.set_xticklabels(rot, fontsize=5)
    ax.tick_params(axis="x", length=0, pad=1)
    for f, c in centros:
        ax.text(c, -0.13, f, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6)
    ax.set_ylabel("%")
    ax.set_xlabel(t["folhas"], labelpad=24)
    _legenda(ax, t)
    fig.savefig(destino, bbox_inches="tight")
    plt.close(fig)


def fig_ordenado(df, t, destino):
    d = df.sort_values("severidade_%").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.9, 3.2), facecolor=CINZA)
    _empilhar(ax, np.arange(len(d)), d)
    _cinza(ax)
    ax.set_ylim(0, 100)
    ax.set_xlim(-1, len(d))
    ax.set_xticks([])
    ax.set_ylabel("%")
    ax.set_xlabel(t["ordenadas"])
    _legenda(ax, t)
    fig.tight_layout()
    fig.savefig(destino, facecolor=fig.get_facecolor())
    plt.close(fig)


def fig_boxplot(df, t, destino):
    caps = sorted(df["dia"].unique())
    dados = [df.loc[df["dia"] == c, "severidade_%"] for c in caps]
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    bp = ax.boxplot(dados, positions=caps, widths=1.6, patch_artist=True, showfliers=False, zorder=3,
                    medianprops=dict(color="black"))
    for b in bp["boxes"]:
        b.set(facecolor="white", edgecolor="#555555")
    rng = np.random.default_rng(0)
    for c, v in zip(caps, dados):
        ax.scatter(c + rng.uniform(-0.45, 0.45, len(v)), v, s=7, color=COR_HEX["necrose"], zorder=4)
    _cinza(ax)
    ax.set_ylim(0, 100)
    ax.set_xticks(caps)
    ax.set_xticklabels([f"{c}\n(n={len(v)})" for c, v in zip(caps, dados)], fontsize=6)
    ax.set_xlim(caps[0] - 2, caps[-1] + 2)
    ax.set_xlabel(t["dias"])
    ax.set_ylabel(t["sev"])
    fig.tight_layout()
    fig.savefig(destino)
    plt.close(fig)


def fig_hist_sev(df, t, destino):
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    bordas = np.arange(0, 101, 10)  # faixas alinhadas ao eixo (0-10, 10-20, ...)
    c, _, barras = ax.hist(df["severidade_%"], bins=bordas, color="#555555", edgecolor="white", zorder=3)
    for r, v in zip(barras, c):
        ax.text(r.get_x() + r.get_width() / 2, v, int(v), ha="center", va="bottom", fontsize=6)
    _cinza(ax)
    ax.set_xticks(bordas)
    ax.set_xlabel(t["sev"])
    ax.set_ylabel(t["n"])
    ax.set_ylim(0, c.max() * 1.15)
    fig.tight_layout()
    fig.savefig(destino)
    plt.close(fig)


def fig_hist_matiz(cache, t, cfg, destino):
    hs, ws = [], []
    for f in glob.glob(os.path.join(cache, "*_sp.npz")):
        z = np.load(f)
        hs.append(z["hue"])
        ws.append(z["area"])
    if not hs:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ax.hist(np.concatenate(hs), bins=np.arange(40, 140, 2), weights=np.concatenate(ws), color="#555555",
            edgecolor="white", zorder=3)
    ax.axvline(cfg["hue_necrose"], color=COR_HEX["necrose"], lw=1.6, zorder=4)
    ax.axvline(cfg["hue_clorose"], color=COR_HEX["clorose"], lw=1.6, zorder=4)
    _cinza(ax)
    ax.set_xlabel(t["matiz"])
    ax.set_ylabel(t["sp"])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(destino)
    plt.close(fig)


def avaliar(df, cache, pasta_anot, t, d_tab, d_fig):
    linhas, conf = [], np.zeros((3, 3), np.int64)
    for _, r in df.iterrows():
        fa = os.path.join(pasta_anot, r["imagem"] + ".png")
        fp = os.path.join(cache, r["imagem"] + "_cls.png")
        if not (os.path.isfile(fa) and os.path.isfile(fp)):
            continue
        pred = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        ann = cv2.resize(cv2.imread(fa, cv2.IMREAD_GRAYSCALE), (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
        m = (pred > 0) & (ann > 0)
        if not m.any():
            continue
        for i in range(3):
            for j in range(3):
                conf[i, j] += int(((ann[m] == i + 1) & (pred[m] == j + 1)).sum())
        linhas.append((r["imagem"], 100 * np.isin(ann[m], (2, 3)).mean(), r["severidade_%"]))
    if not linhas:
        print("  (anotacoes: nenhuma mascara correspondente - avaliacao ignorada)")
        return
    a = pd.DataFrame(linhas, columns=["imagem", "sev_anotada", "sev_prevista"])
    res = []
    for i, nm in enumerate(["saudavel", "clorose", "necrose"]):
        tp, fp_, fn = conf[i, i], conf[:, i].sum() - conf[i, i], conf[i, :].sum() - conf[i, i]
        res.append({"classe": nm, "F1": 2 * tp / max(2 * tp + fp_ + fn, 1), "IoU": tp / max(tp + fp_ + fn, 1)})
    res.append({"classe": "severidade", "MAE_pontos": float((a.sev_anotada - a.sev_prevista).abs().mean()),
                "r_Pearson": float(np.corrcoef(a.sev_anotada, a.sev_prevista)[0, 1]) if len(a) > 2 else np.nan,
                "n_imagens": len(a)})
    pd.DataFrame(res).round(4).to_csv(os.path.join(d_tab, "avaliacao.csv"), index=False)
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.plot([0, 100], [0, 100], color="#555555", lw=1, ls="--")
    ax.scatter(a.sev_anotada, a.sev_prevista, s=14, color=COR_HEX["necrose"], zorder=3)
    _cinza(ax, "both")
    ax.set_xlabel(t["anot"])
    ax.set_ylabel(t["pred"])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(d_fig, "fig_dispersao_anotado.png"))
    plt.close(fig)
    cn = conf / np.maximum(conf.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.imshow(cn, cmap="Greys", vmin=0, vmax=1)
    rot = [t["saudavel"], t["clorose"], t["necrose"]]
    ax.set_xticks(range(3), rot)
    ax.set_yticks(range(3), rot)
    ax.set_xlabel(t["prev"])
    ax.set_ylabel(t["real"])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{100 * cn[i, j]:.0f}%", ha="center", va="center", color="white" if cn[i, j] > 0.5 else "black")
    fig.tight_layout()
    fig.savefig(os.path.join(d_fig, "fig_matriz_confusao.png"))
    plt.close(fig)
    print(f"  avaliacao: {len(a)} imagens anotadas")


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Evolucao da severidade por folha (SLIC + CIELab)")
    ap.add_argument("--pasta", default=None)
    ap.add_argument("--saida", default="resultados")
    ap.add_argument("--folhas", default=None, help="so estas folhas, ex.: 006,016")
    ap.add_argument("--so_figuras", action="store_true", help="nao reprocessa; usa resultados.csv existente")
    ap.add_argument("--lang", choices=["pt", "en"], default="pt")
    ap.add_argument("--refazer", action="store_true")
    ap.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    ap.add_argument("--dias", type=float, default=3.0, help="dias entre capturas (para AUDPC)")
    ap.add_argument("--queda", type=float, default=5.0, help="queda (pontos %%) entre capturas que vira alerta")
    ap.add_argument("--exemplo", default=None, help="folha usada nas imagens da fig_barras_imagens (auto)")
    ap.add_argument("--anotacoes", default="anotacoes/mascaras")
    ap.add_argument("--segmentos", type=int, default=4000)
    ap.add_argument("--compactness", type=float, default=10)
    ap.add_argument("--lado_max", type=int, default=900)
    ap.add_argument("--metodo", choices=["regra", "kmeans"], default="regra")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--hue_necrose", type=float, default=P.HUE_NECROSE)
    ap.add_argument("--hue_clorose", type=float, default=P.HUE_CLOROSE)
    ap.add_argument("--l_necrose", type=float, default=P.L_NECROSE)
    ap.add_argument("--t_local", type=float, default=P.T_LOCAL)
    ap.add_argument("--faixas", default="10,50", help="cortes provisorios de severidade (%%)")
    a = ap.parse_args()

    t = TXT[a.lang]
    S = a.saida
    d_cache, d_tab, d_fig, d_img = (os.path.join(S, x) for x in ("cache", "tabelas", "figuras", "imagens_processadas"))
    for d in (d_cache, d_tab, d_fig, d_img):
        os.makedirs(d, exist_ok=True)
    cfg = dict(segmentos=a.segmentos, compactness=a.compactness, lado_max=a.lado_max, metodo=a.metodo, k=a.k,
               hue_necrose=a.hue_necrose, hue_clorose=a.hue_clorose, l_necrose=a.l_necrose, t_local=a.t_local)
    f_csv = os.path.join(S, "resultados.csv")
    t0 = time.time()

    if a.so_figuras:
        if not os.path.isfile(f_csv):
            raise SystemExit(f"ERRO: {f_csv} nao existe. Rode sem --so_figuras primeiro.")
        try:
            pasta = achar_pasta(a.pasta)
        except SystemExit:
            pasta = ""
        df = pd.read_csv(f_csv, dtype={"imagem": str})
        print(f"[1/4] --so_figuras: {len(df)} imagens lidas de {f_csv}", flush=True)
    else:
        pasta = achar_pasta(a.pasta)
        arq = P.listar_imagens(pasta)
        if a.folhas:
            quer = {x.strip() for x in a.folhas.split(",")}
            arq = [f for f in arq if re.split(r"_", os.path.basename(f))[0] in quer]
            if not arq:
                raise SystemExit(f"ERRO: nenhuma imagem das folhas {sorted(quer)} em '{pasta}'.")
        f_par = os.path.join(d_cache, "params.json")
        if os.path.isfile(f_par) and json.load(open(f_par)) != cfg:
            a.refazer = True  # parametros mudaram: cache invalido
        json.dump(cfg, open(f_par, "w"))
        cfg["refazer"] = a.refazer
        print(f"[1/4] Processando {len(arq)} imagens ({a.workers} processos) ...", flush=True)
        linhas = []
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, r in enumerate(ex.map(analisar, [(f, S, cfg) for f in arq]), 1):
                linhas.append(r)
                print(f"  {i}/{len(arq)} {r['imagem']}  severidade={r['severidade_%']:.2f}%", flush=True)
        df = pd.DataFrame(linhas).round(2)

    df, fonte_tempo = separar_nome(df, a.dias, pasta)
    df.round(2).to_csv(f_csv, index=False, encoding="utf-8-sig")
    dt = por_sessao(df)  # analise temporal: uma linha por folha x sessao

    print("[2/4] Tabelas ...", flush=True)
    evo = tabela_evolucao(dt)
    evo.to_csv(os.path.join(d_tab, "evolucao_por_folha.csv"), index=False)
    porcap = dt.groupby(["captura", "dia"])["severidade_%"].agg(["count", "mean", "median", "std", "min", "max"]).round(2)
    porcap.to_csv(os.path.join(d_tab, "severidade_por_captura.csv"))
    reps = df[df["repeticao"]][["folha", "captura", "dia", "imagem", "data", "severidade_%"]].copy()
    if len(reps):
        reps["amplitude_na_sessao"] = reps.groupby(["folha", "captura"])["severidade_%"].transform(lambda s: s.max() - s.min())
    reps.round(2).to_csv(os.path.join(d_tab, "repeticoes_mesma_sessao.csv"), index=False)
    anom = tabela_anomalias(dt, a.queda)
    anom.to_csv(os.path.join(d_tab, "anomalias_temporais.csv"), index=False)
    c1, c2 = (float(v) for v in a.faixas.split(","))
    sev = df["severidade_%"]
    faixas = pd.DataFrame({"faixa": [f"<{c1:g}%", f"{c1:g}-{c2:g}%", f">{c2:g}%"],
                           "imagens": [int((sev < c1).sum()), int(((sev >= c1) & (sev <= c2)).sum()), int((sev > c2).sum())]})
    faixas["%"] = (100 * faixas["imagens"] / len(df)).round(1)
    faixas.to_csv(os.path.join(d_tab, "faixas_severidade.csv"), index=False)
    with open(os.path.join(S, "resumo_estatistico.txt"), "w", encoding="utf-8") as f:
        f.write("RESUMO ESTATISTICO\n\n")
        f.write(f"Imagens: {len(df)} | Folhas (foliolos): {df['folha'].nunique()}\n")
        f.write(f"Severidade estimada: media {sev.mean():.2f}% | mediana {sev.median():.2f}% | dp {sev.std():.2f} | "
                f"min {sev.min():.2f}% ({df.loc[sev.idxmin(), 'imagem']}) | max {sev.max():.2f}% ({df.loc[sev.idxmax(), 'imagem']})\n")
        f.write("As imagens de uma mesma folha NAO sao independentes (capturas no tempo).\n")
        f.write(f"Tempo: {fonte_tempo}. Dias das sessoes: {sorted(int(v) for v in df['dia'].unique())}\n")
        f.write(f"Fotos repetidas na mesma sessao: {int(df['repeticao'].sum())} (ver tabelas/repeticoes_mesma_sessao.csv)\n")
        f.write(f"Alertas de queda >= {a.queda:g} pontos entre capturas: {len(anom)} (ver tabelas/anomalias_temporais.csv)\n")
        f.write("Valores sao estimativas do algoritmo; acuracia so com anotacao manual (--anotacoes).\n")

    print("[3/4] Figuras ...", flush=True)
    estilo()
    fig_evolucao(dt, t, os.path.join(d_fig, "fig_evolucao_folhas.png"))
    fig_paineis(dt, t, os.path.join(d_fig, "fig_evolucao_paineis.png"))
    fig_barras_imagens(df, t, S, pasta, os.path.join(d_fig, "fig_barras_imagens.png"), a.exemplo)
    fig_ordenado(df, t, os.path.join(d_fig, "fig_ordenado.png"))
    fig_boxplot(dt, t, os.path.join(d_fig, "fig_boxplot_captura.png"))
    fig_hist_sev(df, t, os.path.join(d_fig, "fig_hist_severidade.png"))
    fig_hist_matiz(d_cache, t, cfg, os.path.join(d_fig, "fig_hist_matiz.png"))

    print("[4/4] Avaliacao ...", flush=True)
    if os.path.isdir(a.anotacoes):
        avaliar(df, d_cache, a.anotacoes, t, d_tab, d_fig)
    else:
        print(f"  (sem anotacoes em '{a.anotacoes}': F1/IoU/dispersao/confusao saem depois da anotacao)")

    with open(os.path.join(S, "parametros.txt"), "w", encoding="utf-8") as f:
        for k, v in {**cfg, "faixas": a.faixas, "tempo": fonte_tempo, "lang": a.lang}.items():
            f.write(f"{k} = {v}\n")
    print(f"\nPronto em {time.time() - t0:.0f}s -> {os.path.abspath(S)}")
    print(f"n = {len(df)} | folhas = {df['folha'].nunique()} | severidade media {sev.mean():.2f}% | mediana {sev.median():.2f}%")
    if len(anom):
        print("ATENCAO - quedas de severidade entre capturas (confira as imagens):")
        print(anom.to_string(index=False))


if __name__ == "__main__":
    main()