"""
CALIBRAR - testa combinacoes de limiares do algoritmo contra as suas anotacoes (Labelme).

Ideia: o algoritmo decide saudavel / clorose / necrose com limiares fixos (matiz, luminosidade,
contraste local). Aqui as folhas anotadas sao processadas UMA vez (superpixels + cores), e depois
centenas de combinacoes de limiares sao testadas em segundos, sem reprocessar as imagens.

Uso (na raiz do projeto, com a venv ativa; depois de rodar labelme_para_mascaras.py):
    python calibrar.py
    python calibrar.py --teste 003_2,006_3,016_1     # protocolo correto: calibra no resto, avalia nestas
    python calibrar.py --criterio mae                # escolhe pelo menor erro da severidade (padrao: F1 medio)

Saidas (resultados/calibracao/):
    grade_limiares.csv         todas as combinacoes com F1/IoU por classe e erro da severidade
    melhor_vs_padrao.csv       limiares padrao x melhores (treino e, se houver, teste)
    fig_calibracao.png         F1 e erro da severidade em funcao do limiar de clorose
    feat_*.npz                 cache das folhas (use --refazer se mudar --segmentos etc.)

IMPORTANTE: ajustar os limiares e medir o acerto nas MESMAS folhas da resultado otimista.
Calibre com um grupo (treino) e confirme em folhas que nao participaram (teste, --teste).
"""
import argparse
import glob
import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import piloto as P

PADRAO = dict(hue_clorose=P.HUE_CLOROSE, hue_necrose=P.HUE_NECROSE, l_necrose=P.L_NECROSE, t_local=P.T_LOCAL)


def faixa(txt):
    """'86:108:2' -> [86, 88, ..., 106]   |   '8,12,16' -> [8, 12, 16]"""
    if ":" in txt:
        a, b, c = (float(x) for x in txt.split(":"))
        return list(np.arange(a, b, c))
    return [float(x) for x in txt.split(",")]


def achar_imagem(pasta, base):
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        f = os.path.join(pasta, base + ext)
        if os.path.isfile(f):
            return f
    raise FileNotFoundError(f"imagem '{base}' nao encontrada em {pasta}")


def extrair(args):
    """Processa uma folha anotada e guarda, por superpixel: cores derivadas e contagem de pixels anotados."""
    base, pasta_img, pasta_mask, saida, cfg = args
    f = os.path.join(saida, f"feat_{base}.npz")
    f_mask = os.path.join(pasta_mask, base + ".png")
    # o cache guarda a contagem dos pixels anotados: se a mascara mudou depois dele, refaz
    if os.path.isfile(f) and not cfg["refazer"] and os.path.getmtime(f) > os.path.getmtime(f_mask):
        return base
    img = cv2.imread(achar_imagem(pasta_img, base))
    bgr = P.redimensionar(img, cfg["lado_max"])
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    seg = P.gerar_superpixels(rgb, P.mascara_folha(bgr), cfg["segmentos"], cfg["compactness"])
    ids, F, areas, C = P.features_superpixel(rgb, seg)
    hue, L, croma, dh, dL = P.derivar(F, C)
    ann = cv2.imread(os.path.join(pasta_mask, base + ".png"), cv2.IMREAD_GRAYSCALE)
    ann = cv2.resize(ann, (seg.shape[1], seg.shape[0]), interpolation=cv2.INTER_NEAREST)
    n = int(seg.max()) + 1
    plano, a = seg.ravel(), ann.ravel()
    cont = np.stack([np.bincount(plano[a == k], minlength=n)[ids] for k in (1, 2, 3)], axis=1)
    np.savez_compressed(f, hue=hue, L=L, croma=croma, dh=dh, dL=dL, areas=areas, cont=cont)
    return base


def classificar(d, hue_clorose, hue_necrose, l_necrose, t_local):
    """Mesma regra do piloto/main, vetorizada. 0 = ignorar, 1 = saudavel, 2 = clorose, 3 = necrose."""
    h, L, c, dh, dL = d["hue"], d["L"], d["croma"], d["dh"], d["dL"]
    cls = np.ones(len(h), np.int8)
    cls[h < hue_clorose] = 2
    loc = dh < -t_local
    cls[loc] = np.where(dL[loc] < 0, 3, 2)
    cls[(h < hue_necrose) | (L < l_necrose)] = 3
    cls[(L > 185) & (c < 14)] = 0
    return cls


def avaliar(dados, lim):
    """Confusao (linhas = anotado, colunas = previsto) somada nas folhas + erro absoluto da severidade."""
    conf = np.zeros((3, 3))
    erros = []
    for d in dados:
        cls = classificar(d, **lim)
        for j in (1, 2, 3):
            conf[:, j - 1] += d["cont"][cls == j].sum(axis=0)
        v = cls > 0
        tot_p = d["areas"][v].sum()
        sev_p = 100 * d["areas"][v & (cls > 1)].sum() / max(tot_p, 1)
        c = d["cont"][v]
        sev_a = 100 * (c[:, 1] + c[:, 2]).sum() / max(c.sum(), 1)
        erros.append(abs(sev_p - sev_a))
    out = {}
    for i, nm in enumerate(("saudavel", "clorose", "necrose")):
        tp, fp, fn = conf[i, i], conf[:, i].sum() - conf[i, i], conf[i, :].sum() - conf[i, i]
        out[f"F1_{nm}"] = 2 * tp / max(2 * tp + fp + fn, 1)
        out[f"IoU_{nm}"] = tp / max(tp + fp + fn, 1)
    out["F1_medio"] = np.mean([out["F1_saudavel"], out["F1_clorose"], out["F1_necrose"]])
    out["MAE_severidade"] = float(np.mean(erros))
    return out


def main():
    ap = argparse.ArgumentParser(description="Calibra os limiares contra as anotacoes")
    ap.add_argument("--imagens", default=None)
    ap.add_argument("--mascaras", default="anotacoes/mascaras")
    ap.add_argument("--saida", default="resultados/calibracao")
    ap.add_argument("--teste", default="", help="folhas anotadas reservadas so para o teste final, ex.: 003_2,006_3")
    ap.add_argument("--criterio", choices=["f1", "mae"], default="f1")
    ap.add_argument("--hue_clorose", default="86:108:2")
    ap.add_argument("--hue_necrose", default="60,65,70,75,80,85")
    ap.add_argument("--l_necrose", default="45,55,65")
    ap.add_argument("--t_local", default="8,12,16,99", help="99 = desliga o contraste local")
    ap.add_argument("--segmentos", type=int, default=4000)
    ap.add_argument("--compactness", type=float, default=10)
    ap.add_argument("--lado_max", type=int, default=900)
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
    teste = [x.strip() for x in a.teste.split(",") if x.strip()]
    faltam = [x for x in teste if x not in bases]
    if faltam:
        raise SystemExit(f"ERRO: folhas de teste sem mascara: {faltam}")
    treino = [b for b in bases if b not in teste]
    if not treino:
        raise SystemExit("ERRO: sobrou nenhuma folha para treino.")

    os.makedirs(a.saida, exist_ok=True)
    cfg = dict(segmentos=a.segmentos, compactness=a.compactness, lado_max=a.lado_max, refazer=a.refazer)
    print(f"Folhas anotadas: {len(bases)} (treino {len(treino)}, teste {len(teste)}). Extraindo cores ...", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for b in ex.map(extrair, [(b, pasta_img, a.mascaras, a.saida, cfg) for b in bases]):
            print(f"  ok {b}", flush=True)

    def carregar(lista):
        return [dict(np.load(os.path.join(a.saida, f"feat_{b}.npz"))) for b in lista]

    D_tr, D_te = carregar(treino), carregar(teste)

    nomes = ("hue_clorose", "hue_necrose", "l_necrose", "t_local")
    valores = [faixa(getattr(a, n)) for n in nomes]
    linhas = []
    for comb in itertools.product(*valores):
        lim = dict(zip(nomes, comb))
        linhas.append({**lim, **avaliar(D_tr, lim)})
    grade = pd.DataFrame(linhas)
    chave, asc = ("F1_medio", False) if a.criterio == "f1" else ("MAE_severidade", True)
    grade = grade.sort_values(chave, ascending=asc).reset_index(drop=True)
    grade.round(4).to_csv(os.path.join(a.saida, "grade_limiares.csv"), index=False)

    melhor = {n: float(grade.loc[0, n]) for n in nomes}
    comp = [{"conjunto": "treino", "limiares": "padrao", **PADRAO, **avaliar(D_tr, PADRAO)},
            {"conjunto": "treino", "limiares": "melhor", **melhor, **avaliar(D_tr, melhor)}]
    if D_te:
        comp += [{"conjunto": "teste", "limiares": "padrao", **PADRAO, **avaliar(D_te, PADRAO)},
                 {"conjunto": "teste", "limiares": "melhor", **melhor, **avaliar(D_te, melhor)}]
    comp = pd.DataFrame(comp)
    comp.round(4).to_csv(os.path.join(a.saida, "melhor_vs_padrao.csv"), index=False)

    # figura: efeito do limiar de clorose (demais limiares nos valores melhores)
    xs = sorted(set(valores[0]))
    ys = [avaliar(D_tr, {**melhor, "hue_clorose": x}) for x in xs]
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.plot(xs, [y["F1_medio"] for y in ys], color="#6E370A", marker="o", ms=3, label="F1 medio")
    ax.set_xlabel("Limiar de matiz da clorose (graus)")
    ax.set_ylabel("F1 medio")
    ax2 = ax.twinx()
    ax2.plot(xs, [y["MAE_severidade"] for y in ys], color="#FF6E00", marker="s", ms=3, label="Erro da severidade")
    ax2.set_ylabel("Erro medio da severidade (pontos %)")
    ax.axvline(PADRAO["hue_clorose"], color="gray", ls="--", lw=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.saida, "fig_calibracao.png"), dpi=300)
    plt.close(fig)

    pd.set_option("display.width", 200)
    cols = ["hue_clorose", "hue_necrose", "l_necrose", "t_local", "F1_saudavel", "F1_clorose", "F1_necrose", "F1_medio", "MAE_severidade"]
    print("\n5 melhores combinacoes (treino):")
    print(grade[cols].head(5).round(3).to_string(index=False))
    print("\nPadrao x melhor:")
    print(comp[["conjunto", "limiares"] + cols].round(3).to_string(index=False))
    print(f"\nTabelas e figura em: {os.path.abspath(a.saida)}")
    if not D_te:
        print("AVISO: sem --teste, o 'melhor' foi medido nas mesmas folhas em que foi escolhido (resultado otimista).")


if __name__ == "__main__":
    main()