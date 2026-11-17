import cv2
import numpy as np
import matplotlib.pyplot as plt

# Funcion auxiliar para mostrar imagenes
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:
        plt.show(block=blocking)


def rellenar_objetos(mask, k_close, k_open):
    """Clausura, relleno de huecos (contornos externos) y apertura."""
    se_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    se_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se_c)
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    relleno = np.zeros_like(mask)
    cv2.drawContours(relleno, contornos, -1, 255, thickness=cv2.FILLED)
    return cv2.morphologyEx(relleno, cv2.MORPH_OPEN, se_o)


# A. Segmentacion de monedas y dados
print("=" * 70)
print("A. SEGMENTACION")
print("=" * 70)

# Etapa A.1: Cargamos la imagen
img_bgr = cv2.imread('./Imagenes/monedas.jpg')
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
H, W = gray.shape
print(f"[A.1] Imagen cargada: {W}x{H} px")
imshow(img, title="A.1 - Imagen original", color_img=True)

# Etapa A.2: Estimamos el fondo (intensidad no uniforme)
# Mediana con kernel grande sobre la imagen reducida: los objetos desaparecen y queda solamente el degrade del fondo.
chica = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
fondo = cv2.medianBlur(chica, 101)
fondo = cv2.resize(fondo, (W, H), interpolation=cv2.INTER_LINEAR)
dif = gray.astype(np.int16) - fondo.astype(np.int16)
print(f"[A.2] Fondo estimado: min={fondo.min()}  max={fondo.max()}")
plt.figure()
ax1 = plt.subplot(121); imshow(fondo, new_fig=False, title="A.2 - Fondo estimado")
plt.subplot(122, sharex=ax1, sharey=ax1)
imshow(np.clip(dif, 0, 255).astype(np.uint8), new_fig=False, title="A.2 - Gris menos fondo (parte positiva)")
plt.show(block=False)

# Etapa A.3: Mascara por bordes (gradiente morfologico)
# Las monedas tienen bordes fuertes sobre un fondo casi liso.
blur = cv2.GaussianBlur(gray, (5, 5), 0)
se_g = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, se_g)
_, bordes = cv2.threshold(grad, 30, 255, cv2.THRESH_BINARY)
mask_bordes = rellenar_objetos(bordes, k_close=15, k_open=31)
print("[A.3] Mascara por gradiente morfologico (umbral 30) generada")
plt.figure()
ax1 = plt.subplot(131); imshow(grad, new_fig=False, title="A.3 - Gradiente morfologico")
plt.subplot(132, sharex=ax1, sharey=ax1); imshow(bordes, new_fig=False, title="A.3 - Umbral")
plt.subplot(133, sharex=ax1, sharey=ax1); imshow(mask_bordes, new_fig=False, title="A.3 - Cierre + relleno + apertura")
plt.show(block=False)

# Etapa A.4: Mascara por color (monedas doradas)
# Una moneda dorada tiene poco contraste en gris con el fondo pero mucha saturacion.
sat = cv2.GaussianBlur(hsv[:, :, 1], (7, 7), 0)
_, dorado = cv2.threshold(sat, 70, 255, cv2.THRESH_BINARY)
mask_dorado = rellenar_objetos(dorado, k_close=15, k_open=15)
print("[A.4] Mascara por saturacion (monedas doradas) generada")
plt.figure()
ax1 = plt.subplot(121); imshow(sat, new_fig=False, title="A.4 - Saturacion (S)")
plt.subplot(122, sharex=ax1, sharey=ax1); imshow(mask_dorado, new_fig=False, title="A.4 - Mascara dorados")
plt.show(block=False)

# Etapa A.5: Mascara por brillo (dados blancos)
# Los dados son mucho mas claros que el fondo y casi sin saturacion. Los puntos
# negros quedan como huecos que se rellenan con el cierre.
blanco = ((dif > 25) & (hsv[:, :, 1] < 40)).astype(np.uint8) * 255
mask_blanco = rellenar_objetos(blanco, k_close=25, k_open=41)

# El fondo claro de la derecha genera manchas irregulares: nos quedamos solo con los
# objetos convexos y grandes (solidez alta), que son los dados y las monedas claras.
n_b, lab_b, st_b, _ = cv2.connectedComponentsWithStats(mask_blanco)
mask_blanco_f = np.zeros_like(mask_blanco)
for i in range(1, n_b):
    obj_b = (lab_b == i).astype(np.uint8)
    cnt_b, _ = cv2.findContours(obj_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solidez = cv2.contourArea(cnt_b[0]) / cv2.contourArea(cv2.convexHull(cnt_b[0]))
    if st_b[i, cv2.CC_STAT_AREA] > 15000 and solidez > 0.93:
        mask_blanco_f[obj_b == 1] = 255
mask_blanco = mask_blanco_f
print("[A.5] Mascara por brillo (dados blancos) generada")
plt.figure()
ax1 = plt.subplot(121); imshow(blanco, new_fig=False, title="A.5 - Claro y poco saturado")
plt.subplot(122, sharex=ax1, sharey=ax1); imshow(mask_blanco, new_fig=False, title="A.5 - Filtrado por solidez")
plt.show(block=False)

# Etapa A.6: Union de las mascaras y etiquetado
mask = cv2.bitwise_or(cv2.bitwise_or(mask_bordes, mask_dorado), mask_blanco)
mask = rellenar_objetos(mask, k_close=5, k_open=31)

AREA_MIN = 15000     # descarta restos de ruido
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
objetos = []
for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] < AREA_MIN:
        continue
    obj = (labels == i).astype(np.uint8)
    cnt, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = cv2.contourArea(cnt[0])
    perim = cv2.arcLength(cnt[0], True)
    rho = 4 * np.pi * area / perim ** 2
    objetos.append(dict(id=i, area=stats[i, cv2.CC_STAT_AREA], rho=rho, cx=centroids[i][0], cy=centroids[i][1],
                        bbox=(stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                              stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])))
mask_final = np.isin(labels, [o['id'] for o in objetos]).astype(np.uint8) * 255
print(f"[A.6] Objetos segmentados: {len(objetos)}")

segmentada = img.copy()
segmentada[mask_final == 0] = 0
plt.figure()
ax1 = plt.subplot(121); imshow(mask_final, new_fig=False, title="A.6 - Mascara final")
plt.subplot(122, sharex=ax1, sharey=ax1); imshow(segmentada, new_fig=False, title="A.6 - Objetos segmentados", color_img=True)
plt.show(block=False)


# B. Clasificacion y conteo de monedas

print("\n" + "=" * 70)
print("B. CLASIFICACION DE MONEDAS")
print("=" * 70)

# Etapa B.1: Separo dados de monedas por factor de forma
# Las monedas son casi circulares (rho alto); los dados, cuadrados redondeados.
RHO_TH = 0.80
dados = [o for o in objetos if o['rho'] < RHO_TH]
monedas = [o for o in objetos if o['rho'] >= RHO_TH]
for o in objetos:
    tipo = 'moneda' if o['rho'] >= RHO_TH else 'dado'
    print(f"[B.1] Objeto {o['id']:2d}  area={o['area']:6d}  rho={o['rho']:.2f}  clasificado como: {tipo}")
print(f"[B.1] Monedas: {len(monedas)}  |  Dados: {len(dados)}")

# Etapa B.2: Clasifico las monedas por tamaño (area)
# Todas las monedas estan en el mismo plano, asi que el area es un buen descriptor.
# Se confirma con el color (saturacion media).
AREA_10C_MAX = 75000     # por debajo: 10 centavos
AREA_1P_MAX = 100000      # entre ambos: 1 peso / por encima: 50 centavos
etiquetas = {'10 centavos': (0, 255, 255), '1 peso': (255, 0, 0), '50 centavos': (0, 255, 0)}
conteo = {k: 0 for k in etiquetas}
for o in monedas:
    px = labels == o['id']
    o['sat'] = hsv[:, :, 1][px].mean()
    if o['area'] < AREA_10C_MAX:
        o['clase'] = '10 centavos'
    elif o['area'] < AREA_1P_MAX:
        o['clase'] = '1 peso'
    else:
        o['clase'] = '50 centavos'
    conteo[o['clase']] += 1
    print(f"[B.2] Moneda {o['id']:2d}  area={o['area']:6d}  S medio={o['sat']:5.1f}  clase: {o["clase"]}")

print("\n[B.2] Conteo por tipo de moneda:")
for k, v in conteo.items():
    print(f"      {k:12s}: {v}")
print(f"      TOTAL       : {sum(conteo.values())}")

res_b = img.copy()
for o in monedas:
    color = etiquetas[o['clase']]
    cnt, _ = cv2.findContours((labels == o['id']).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(res_b, cnt, -1, color, 8)
    cv2.putText(res_b, o['clase'], (int(o['cx']) - 90, int(o['cy'])), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 4)
imshow(res_b, title="B.2 - Monedas clasificadas", color_img=True)


# C. Valor de la cara superior de cada dado

print("\n" + "=" * 70)
print("C. LECTURA DE DADOS")
print("=" * 70)

res_c = img.copy()
total_dados = 0
for n_d, o in enumerate(dados, start=1):
    # Etapa C.1: Recorto el dado
    x, y, w, h = o['bbox']
    obj = ((labels == o['id']).astype(np.uint8) * 255)[y:y + h, x:x + w]
    gris_d = gray[y:y + h, x:x + w]

    # Etapa C.2: Busco los puntos oscuros dentro del dado
    # Erosionamos la mascara para no tomar el borde del dado ni las caras laterales.
    se_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    interior = cv2.erode(obj, se_e)
    _, oscuros = cv2.threshold(gris_d, 90, 255, cv2.THRESH_BINARY_INV)
    oscuros = cv2.bitwise_and(oscuros, interior)

    # Etapa C.3: Filtro por area y circularidad (puntos de la cara superior)
    n_p, lab_p, st_p, cen_p = cv2.connectedComponentsWithStats(oscuros)
    puntos = []
    for j in range(1, n_p):
        a = st_p[j, cv2.CC_STAT_AREA]
        if a < 150:
            continue
        p_obj = (lab_p == j).astype(np.uint8)
        cnt, _ = cv2.findContours(p_obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        per = cv2.arcLength(cnt[0], True)
        rho_p = 4 * np.pi * cv2.contourArea(cnt[0]) / per ** 2
        print(f"[C.3] Dado {n_d}  candidato {j:2d}  area={a:5d}  rho={rho_p:.2f}")
        if rho_p > 0.75:
            puntos.append((cen_p[j][0] + x, cen_p[j][1] + y))
    valor = len(puntos)
    total_dados += valor
    print(f"[C.3] Dado {n_d}: cara superior = {valor}")

    for (px_, py_) in puntos:
        cv2.circle(res_c, (int(px_), int(py_)), 18, (255, 0, 0), 6)
    cv2.rectangle(res_c, (x, y), (x + w, y + h), (0, 255, 0), 6)
    cv2.putText(res_c, f"Dado {n_d}: {valor}", (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 4)

    plt.figure()
    ax1 = plt.subplot(131); imshow(img[y:y + h, x:x + w], new_fig=False, title=f"C.1 - Dado {n_d}", color_img=True)
    plt.subplot(132, sharex=ax1, sharey=ax1); imshow(interior, new_fig=False, title="C.2 - Interior (erosion)")
    plt.subplot(133, sharex=ax1, sharey=ax1); imshow(oscuros, new_fig=False, title="C.2 - Puntos oscuros")
    plt.show(block=False)

print(f"\n[C.3] Cantidad de dados: {len(dados)}  |  Suma de caras superiores: {total_dados}")
imshow(res_c, title="C.3 - Dados leidos", color_img=True)

plt.show()