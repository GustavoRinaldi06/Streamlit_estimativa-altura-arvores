import cv2
import numpy as np
import math
from ultralytics import YOLO
import os

# --- FUNÇÕES AUXILIARES ---
def dms_para_decimal(graus, minutos, segundos, direcao):
    decimal = graus + (minutos / 60) + (segundos / 3600)
    if direcao.upper() in ['S', 'W']:
        decimal = -decimal
    return decimal

def calcular_gsd_sasplanet(zoom_level, lat_dec):
    equator_resolution = 156543.03392
    lat_rad = math.radians(lat_dec)
    gsd = (equator_resolution * math.cos(lat_rad)) / (2 ** zoom_level)
    return gsd

# ===============================================================================
# MÓDULO 1: INTERFACE PARA SELEÇÃO 
# ===============================================================================
class SeletorSombraReferencia:
    def __init__(self, image_path):
        self.img_original = cv2.imread(image_path)
        if self.img_original is None:
            raise ValueError(f"Erro ao abrir imagem: {image_path}")
        self.img_display = self.img_original.copy()
        self.pontos = []
        self.janela_nome = "ETAPA 1: Clique na Base e depois na Ponta da Sombra do Poste"
        self.concluido = False
        self.distancia_px = 0

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.pontos) < 2:
                self.pontos.append((x, y))
                cv2.circle(self.img_display, (x, y), 5, (0, 255, 0), -1)
                
                if len(self.pontos) == 2:
                    cv2.line(self.img_display, self.pontos[0], self.pontos[1], (0, 255, 0), 2)
                    p1, p2 = np.array(self.pontos[0]), np.array(self.pontos[1])
                    self.distancia_px = np.linalg.norm(p1 - p2)
                    self.concluido = True

        elif event == cv2.EVENT_MOUSEMOVE and len(self.pontos) == 1:
            temp = self.img_display.copy()
            cv2.line(temp, self.pontos[0], (x, y), (0, 255, 255), 1)
            cv2.imshow(self.janela_nome, temp)
            return

        cv2.imshow(self.janela_nome, self.img_display)

    def obter_dados_referencia(self):
        cv2.namedWindow(self.janela_nome)
        cv2.setMouseCallback(self.janela_nome, self._mouse_callback)
        cv2.imshow(self.janela_nome, self.img_display)
        cv2.waitKey(0)
        cv2.destroyWindow(self.janela_nome)
        
        if not self.concluido or self.distancia_px == 0:
            raise Exception("Seleção cancelada.")
        return self.distancia_px, self.pontos

# ===============================================================================
# MÓDULO 2: ANÁLISE HÍBRIDA
# ===============================================================================
def analisar_arvores_hibrido(
    model_path, 
    img_original, 
    azimute_sombra_graus, 
    fator_altura_por_pixel,    
    gsd_real_mapa=None,        
    pontos_referencia=None,
    output_path="resultado_validacao.jpg"
):
    print("\n--- Etapa 2: Análise Automática ---")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        return

    # Garante que img_original é uma matriz do OpenCV
    if isinstance(img_original, str):
        img_original = cv2.imread(img_original)

    if img_original is None:
        print("Erro ao ler matriz da imagem.")
        return

    h, w = img_original.shape[:2]
    vis_img = img_original.copy()

    # Desenhar linha do poste se existir
    if pontos_referencia and len(pontos_referencia) == 2:
        cv2.line(vis_img, pontos_referencia[0], pontos_referencia[1], (0, 255, 0), 3)
        cv2.putText(vis_img, "REF POSTE", (pontos_referencia[0][0], pontos_referencia[0][1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Inferência
    results = model.predict(img_original, conf=0.25, imgsz=640, verbose=False)[0]
    if results.masks is None:
        print("Nenhuma árvore detectada.")
        cv2.imwrite(output_path, vis_img)
        return

    CLASS_ID_SOMBRA, CLASS_ID_ARVORE = 0, 1
    arvores, sombras = [], []
    
    masks = results.masks.xy 
    classes = results.boxes.cls.cpu().numpy()
    
    for i, poly in enumerate(masks):
        if len(poly) == 0: continue
        poly_int = np.array(poly, dtype=np.int32)
        cls_id = int(classes[i])
        
        if cls_id == CLASS_ID_ARVORE:
            (x, y), r = cv2.minEnclosingCircle(poly_int)
            obj = {'center': (int(x), int(y)), 'radius': r, 'poly': poly_int}
            arvores.append(obj)
        elif cls_id == CLASS_ID_SOMBRA:
            obj = {'poly': poly_int, 'mask_img': None}
            sombras.append(obj)

    # Renderizar sombras
    for s in sombras:
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(m, [s['poly']], 255)
        s['mask_img'] = m

    # Geometria
    rad = math.radians(azimute_sombra_graus)
    vec = (math.sin(rad), -math.cos(rad))
    raio_busca = 3000
    
    candidatos = []
    for i_arv, arv in enumerate(arvores):
        cx, cy = arv['center']
        ponta = (int(cx + vec[0]*raio_busca), int(cy + vec[1]*raio_busca))
        
        mask_line = np.zeros((h, w), dtype=np.uint8)
        cv2.line(mask_line, (cx, cy), ponta, 255, thickness=2)

        for i_somb, s in enumerate(sombras):
            inter = cv2.bitwise_and(mask_line, s['mask_img'])
            if cv2.countNonZero(inter) > 0:
                pts = cv2.findNonZero(inter).reshape(-1, 2)
                dists = np.linalg.norm(pts - [cx, cy], axis=1)
                
                candidatos.append({
                    'dist_inicio': np.min(dists),
                    'dist_final': np.max(dists),
                    'arvore_idx': i_arv,
                    'ponto_final': pts[np.argmax(dists)]
                })

    candidatos.sort(key=lambda x: x['dist_inicio'])
    processados = set()

    for c in candidatos:
        if c['arvore_idx'] in processados: continue
        processados.add(c['arvore_idx'])
        
        arv = arvores[c['arvore_idx']]
        cx, cy = arv['center']
        px, py = tuple(c['ponto_final'])
        
        # --- CÁLCULOS HÍBRIDOS ---
        altura_m = c['dist_final'] * fator_altura_por_pixel
        raio_m = (arv['radius'] * gsd_real_mapa) if gsd_real_mapa is not None and gsd_real_mapa > 0 else None

        # Desenho
        cv2.circle(vis_img, (cx, cy), int(arv['radius']), (0, 255, 255), 2)
        cv2.line(vis_img, (cx, cy), (px, py), (0, 0, 255), 2)
        cv2.circle(vis_img, (px, py), 4, (0, 0, 255), -1)
        
        texto = (f"R:{raio_m:.1f}m | H:{altura_m:.1f}m" if raio_m is not None else f"H:{altura_m:.1f}m")
        
        text_y = cy - 10 if cy - 10 > 10 else cy + 20
        (wt, ht), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis_img, (cx-20, text_y-ht-4), (cx-20+wt, text_y+4), (0,0,0), -1)
        cv2.putText(vis_img, texto, (cx-20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # Salva diretamente a imagem no destino sem abrir cv2.imshow
    cv2.imwrite(output_path, vis_img)
    print(f"Resultado salvo em: {output_path}")

# ===============================================================================
# Função Principal para Interfaces
# ===============================================================================
def VerificarPorReferência(
    path_imagem, 
    path_modelo, 
    h_referencia, 
    zoom, 
    latitude_dec, 
    output_path="output/resultado_validacao.jpg",
    sombra_poste_px=80.0,
    azimute_manual=45.0,
    gsd_manual=None
):
    if not os.path.exists(path_imagem) or not os.path.exists(path_modelo):
        return False

    img_mat = cv2.imread(path_imagem)
    if img_mat is None:
        return False

    # Define GSD
    if gsd_manual is not None and gsd_manual > 0:
        gsd_mapa = gsd_manual
    else:
        gsd_mapa = calcular_gsd_sasplanet(zoom, latitude_dec)

    fator_h = h_referencia / sombra_poste_px if sombra_poste_px > 0 else 0.05

    # Executa a análise e salva diretamente no output_path
    analisar_arvores_hibrido(
        model_path=path_modelo,
        img_original=img_mat,
        azimute_sombra_graus=azimute_manual,
        fator_altura_por_pixel=fator_h,
        gsd_real_mapa=gsd_mapa,
        pontos_referencia=None,
        output_path=output_path
    )
    return True