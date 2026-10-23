"""
Converte os .json do Labelme em mascaras de classe (PNG).

Valores na mascara:  0 = fundo | 1 = saudavel | 2 = clorose | 3 = necrose
Regra: dentro da folha (pixel nao-branco), tudo que NAO foi anotado e saudavel;
os poligonos 'clorose' e 'necrose' sobrescrevem. Poligonos maiores sao pintados antes dos menores
(pode contornar a folha toda como necrose e marcar clorose por cima). Fora da folha = fundo.

Uso:
    python labelme_para_mascaras.py --json dados/anotacoes --imagens images_base/detached_leaves --saida dados/mascaras
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

CLASSES = {"clorose": 2, "necrose": 3}


def converter(caminho_json, pasta_imagens, saida):
    with open(caminho_json, encoding="utf-8") as f:
        d = json.load(f)
    nome_img = os.path.basename(d["imagePath"].replace("\\", "/"))
    img = cv2.imread(os.path.join(pasta_imagens, nome_img))
    if img is None:
        raise FileNotFoundError(f"Imagem nao encontrada: {nome_img}")
    h, w = img.shape[:2]
    if (h, w) != (d["imageHeight"], d["imageWidth"]):
        raise ValueError(f"{nome_img}: tamanho da imagem difere do JSON")

    folha = img.min(axis=2) < 235
    mascara = np.zeros((h, w), np.uint8)
    mascara[folha] = 1  # saudavel por padrao

    # poligonos MAIORES primeiro, MENORES por cima: assim vale tanto uma necrose grande envolvendo
    # manchas de clorose quanto uma necrose pequena dentro de uma clorose, em qualquer ordem de desenho
    def area(f):
        return cv2.contourArea(np.round(np.array(f["points"])).astype(np.float32)) if len(f["points"]) >= 3 else 0
    formas = sorted(d["shapes"], key=area, reverse=True)
    for forma in formas:
        rotulo = forma["label"].strip().lower()
        if rotulo not in CLASSES:
            print(f"  aviso: rotulo desconhecido '{rotulo}' em {nome_img} (ignorado)")
            continue
        pts = np.round(np.array(forma["points"])).astype(np.int32)
        if forma.get("shape_type", "polygon") != "polygon" or len(pts) < 3:
            continue
        cv2.fillPoly(mascara, [pts], CLASSES[rotulo])

    mascara[~folha] = 0  # poligono que passa do contorno da folha nao conta no fundo
    os.makedirs(saida, exist_ok=True)
    base = os.path.splitext(nome_img)[0]
    cv2.imwrite(os.path.join(saida, base + ".png"), mascara)
    return base, [(mascara == v).sum() for v in (1, 2, 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--imagens", required=True)
    ap.add_argument("--saida", required=True)
    a = ap.parse_args()
    for j in sorted(glob.glob(os.path.join(a.json, "*.json"))):
        base, (s, c, n) = converter(j, a.imagens, a.saida)
        tot = s + c + n
        print(f"{base}: saudavel={s} clorose={c} necrose={n} | severidade={100*(c+n)/tot:.1f}%")


if __name__ == "__main__":
    main()