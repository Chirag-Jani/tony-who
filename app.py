import cv2
import mediapipe as mp
import numpy as np
import colorsys
import random
import time
import math

# ── MediaPipe ────────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
)
CONNECTIONS = list(mp_hands.HAND_CONNECTIONS)
FINGER_TIPS = [4, 8, 12, 16, 20]

# ── Camera ───────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ── Colors (Iron Man palette) ─────────────────────────────────────────────────
CYAN       = (255, 220, 0)      # BGR cyan
CYAN_DIM   = (120, 100, 0)
BLUE       = (200, 80,  0)
WHITE      = (255, 255, 255)
GOLD       = (0,   180, 255)
RED_ALERT  = (0,   0,   255)

# ── State ────────────────────────────────────────────────────────────────────
nodes      = []          # list of (x, y)
beams      = []          # list of (i, j) undirected segment between two anchors
rects      = []          # list of (i, j) diagonal corners → axis-aligned square/rectangle
circles    = []          # list of (i_center, i_rim) — radius = dist between anchors
shape_mode = 0           # 0 = line, 1 = rect (AABB from two anchors), 2 = circle
MODE_NAMES = ("LINE", "RECT", "CIRCLE")
globe_mode = False       # Iron Man 2–style HUD wireframe sphere (G to toggle)
globe_rx = 0.35
globe_ry = -0.6
globe_cx = None          # screen center (set on first G)
globe_cy = None
globe_scale = 1.0
globe_two_prev = None    # (mid_xy, pinch_dist) for two-hand move + zoom
globe_prev_pinch = {}    # hand_idx -> last pinch screen pos for drag-to-spin
pinch_cooldown   = {}    # hand_idx -> frame count
prev_pinch       = {}    # hand_idx -> bool
both_pinch_nodes = {}    # hand_idx -> node_idx for beam building
pinch_smooth     = {}    # hand_idx -> (x, y) smoothed pinch for stable placement/drag
drag_node        = {}    # hand_idx -> node index while grabbing, or None
frame_count = 0
pulse = 0.0

GRAB_RADIUS = 48         # px: pinch within this of a node grabs it instead of new node
PINCH_SMOOTH = 0.42      # EMA toward raw pinch while pinching (higher = snappier)

# Beam link: edge-trigger so one clean link per gesture; preview drawn while aiming
prev_both_pinch = False

QUOTES = [
    "Pinch empty space to drop an anchor — pinch near one to grab.",
    "M cycles LINE / RECT / CIRCLE — both pinches commit the current mode.",
    "Rect uses two anchors as diagonal corners (axis-aligned). Circle: center vs rim anchor.",
    "Open palm deletes a node; B/R/O pop last line, rect, circle; Backspace = last node.",
    "C clears everything. Space anchors on the grid read cleaner.",
]

# ── Helpers ──────────────────────────────────────────────────────────────────
def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def is_pinching(lm, w, h, threshold=40):
    thumb  = (int(lm[4].x*w),  int(lm[4].y*h))
    index  = (int(lm[8].x*w),  int(lm[8].y*h))
    return dist(thumb, index) < threshold, ((thumb[0]+index[0])//2, (thumb[1]+index[1])//2)

def is_open_palm(lm, w, h):
    tips = [8, 12, 16, 20]
    bases = [6, 10, 14, 18]
    extended = sum(1 for t, b in zip(tips, bases)
                   if lm[t].y < lm[b].y)
    return extended >= 4

def draw_glowing_line(img, p1, p2, color, thickness=1, alpha=0.6):
    overlay = img.copy()
    cv2.line(overlay, p1, p2, color, thickness + 6)
    cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
    cv2.line(img, p1, p2, color, thickness + 1)
    bright = tuple(min(255, c+100) for c in color)
    cv2.line(img, p1, p2, bright, max(1, thickness-1))

def draw_corner_bracket(img, pt, size=18, color=CYAN, thickness=2):
    x, y = pt
    s = size
    # top-left
    cv2.line(img, (x-s, y-s), (x-s+8, y-s), color, thickness)
    cv2.line(img, (x-s, y-s), (x-s, y-s+8), color, thickness)
    # top-right
    cv2.line(img, (x+s, y-s), (x+s-8, y-s), color, thickness)
    cv2.line(img, (x+s, y-s), (x+s, y-s+8), color, thickness)
    # bottom-left
    cv2.line(img, (x-s, y+s), (x-s+8, y+s), color, thickness)
    cv2.line(img, (x-s, y+s), (x-s, y+s-8), color, thickness)
    # bottom-right
    cv2.line(img, (x+s, y+s), (x+s-8, y+s), color, thickness)
    cv2.line(img, (x+s, y+s), (x+s, y+s-8), color, thickness)

def draw_reticle(img, pt, radius=14, color=CYAN):
    cv2.circle(img, pt, radius, color, 1)
    cv2.circle(img, pt, 3, color, -1)
    cv2.line(img, (pt[0]-radius-4, pt[1]), (pt[0]-radius+4, pt[1]), color, 1)
    cv2.line(img, (pt[0]+radius-4, pt[1]), (pt[0]+radius+4, pt[1]), color, 1)
    cv2.line(img, (pt[0], pt[1]-radius-4), (pt[0], pt[1]-radius+4), color, 1)
    cv2.line(img, (pt[0], pt[1]+radius-4), (pt[0], pt[1]+radius+4), color, 1)

def draw_node(img, pt, pulse):
    p = abs(math.sin(pulse))
    r = int(10 + 5 * p)
    glow_color = tuple(int(c * (0.4 + 0.6*p)) for c in CYAN)
    overlay = img.copy()
    cv2.circle(overlay, pt, r+8, glow_color, -1)
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
    cv2.circle(img, pt, r, CYAN, 2)
    cv2.circle(img, pt, 4, WHITE, -1)
    draw_corner_bracket(img, pt, size=r+6, color=CYAN_DIM, thickness=1)

def draw_beam(img, p1, p2, pulse):
    p = abs(math.sin(pulse * 2))
    color = tuple(int(c * (0.6 + 0.4*p)) for c in CYAN)
    draw_glowing_line(img, p1, p2, color, thickness=2)
    # midpoint label
    mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
    d = int(dist(p1, p2))
    cv2.putText(img, f"{d}px", (mid[0]+6, mid[1]-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, CYAN_DIM, 1)

def draw_dashed_line(img, p1, p2, color, thickness=2, dash_len=14, gap_len=8, phase=0.0):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    L = math.hypot(dx, dy) + 1e-6
    ux, uy = dx / L, dy / L
    period = dash_len + gap_len
    t = (-phase) % period
    while t < L:
        seg_end = min(t + dash_len, L)
        a = (int(p1[0] + ux * t), int(p1[1] + uy * t))
        b = (int(p1[0] + ux * seg_end), int(p1[1] + uy * seg_end))
        cv2.line(img, a, b, color, thickness)
        t = seg_end + gap_len

def aabb_from_diagonal(p1, p2):
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    left, right = min(x1, x2), max(x1, x2)
    top, bottom = min(y1, y2), max(y1, y2)
    return (left, top), (right, bottom)

def draw_beam_preview(img, p1, p2, pulse, already_linked):
    ph = (pulse * 40.0) % (14 + 8)
    base = GOLD if not already_linked else CYAN_DIM
    col = tuple(int(c * (0.55 if already_linked else 0.85)) for c in base)
    overlay = img.copy()
    draw_dashed_line(overlay, p1, p2, col, thickness=3, dash_len=16, gap_len=10, phase=ph)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
    draw_dashed_line(img, p1, p2, col, thickness=1, dash_len=16, gap_len=10, phase=ph)
    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    label = "LINKED" if already_linked else "LINE PREVIEW"
    cv2.putText(img, label, (mid[0] - 52, mid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

def draw_rect_preview(img, p1, p2, pulse, already_linked):
    tl, br = aabb_from_diagonal(p1, p2)
    corners = [(tl[0], tl[1]), (br[0], tl[1]), (br[0], br[1]), (tl[0], br[1])]
    ph = (pulse * 40.0) % 22
    base = GOLD if not already_linked else CYAN_DIM
    col = tuple(int(c * (0.5 if already_linked else 0.82)) for c in base)
    overlay = img.copy()
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        draw_dashed_line(overlay, a, b, col, thickness=3, dash_len=14, gap_len=8, phase=ph + i * 6)
    cv2.addWeighted(overlay, 0.32, img, 0.68, 0, img)
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        draw_dashed_line(img, a, b, col, thickness=1, dash_len=14, gap_len=8, phase=ph + i * 6)
    cx, cy = (tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2
    label = "LINKED" if already_linked else "RECT PREVIEW"
    cv2.putText(img, label, (cx - 58, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

def draw_circle_preview(img, center, rim_pt, pulse, already_linked):
    r = max(3, int(dist(center, rim_pt)))
    ph = pulse * 3.0
    base = BLUE if not already_linked else CYAN_DIM
    col = tuple(int(c * (0.5 if already_linked else 0.8)) for c in base)
    nseg = 48
    for k in range(nseg):
        a1 = (k / nseg) * 2 * math.pi + ph
        a2 = ((k + 1) / nseg) * 2 * math.pi + ph
        if (k + int(ph * 10)) % 3 == 0:
            continue
        sa = (int(center[0] + r * math.cos(a1)), int(center[1] + r * math.sin(a1)))
        sb = (int(center[0] + r * math.cos(a2)), int(center[1] + r * math.sin(a2)))
        cv2.line(img, sa, sb, col, 2)
    overlay = img.copy()
    cv2.circle(overlay, center, r, col, 1)
    cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
    label = "LINKED" if already_linked else "CIRCLE PREVIEW"
    cv2.putText(
        img,
        label,
        (center[0] - 62, center[1] - r - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        col,
        1,
    )

def draw_alignment_grid(img, step=72, dot_r=1):
    c = tuple(int(x * 0.35) for x in CYAN_DIM)
    for x in range(step // 2, W, step):
        for y in range(step // 2, H, step):
            cv2.circle(img, (x, y), dot_r, c, -1)

def draw_anchor_highlight(img, pt, pulse, color):
    r = int(22 + 6 * abs(math.sin(pulse * 3)))
    cv2.circle(img, pt, r, color, 2)
    cv2.circle(img, pt, 4, color, -1)

def draw_rect_shape(img, p1, p2, pulse, color_base=GOLD):
    p = abs(math.sin(pulse * 2))
    color = tuple(int(c * (0.55 + 0.35 * p)) for c in color_base)
    tl, br = aabb_from_diagonal(p1, p2)
    pts = np.array([[tl[0], tl[1]], [br[0], tl[1]], [br[0], br[1]], [tl[0], br[1]]], np.int32)
    overlay = img.copy()
    cv2.polylines(overlay, [pts], True, color, 3)
    cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)
    cv2.polylines(img, [pts], True, color, 2)
    bright = tuple(min(255, c + 80) for c in color)
    cv2.polylines(img, [pts], True, bright, 1)
    w, h = br[0] - tl[0], br[1] - tl[1]
    cv2.putText(
        img,
        f"{w}x{h}",
        (tl[0] + 4, tl[1] - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        CYAN_DIM,
        1,
    )

def draw_circle_shape(img, center, rim_pt, pulse, color_base=BLUE):
    p = abs(math.sin(pulse * 2))
    r = int(dist(center, rim_pt))
    r = max(2, r)
    color = tuple(int(c * (0.55 + 0.35 * p)) for c in color_base)
    overlay = img.copy()
    cv2.circle(overlay, center, r + 2, color, 4)
    cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
    cv2.circle(img, center, r, color, 2)
    cv2.circle(img, center, max(1, r - 1), tuple(min(255, c + 90) for c in color), 1)
    cv2.putText(
        img,
        f"r={r}",
        (center[0] + r // 3, center[1] - r // 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        CYAN_DIM,
        1,
    )

def remove_node_index(nearest):
    """Remove nodes[nearest] and fix beams, rects, circles + drag indices."""
    global beams, rects, circles
    beams[:] = [(a, b) for a, b in beams if a != nearest and b != nearest]
    beams[:] = [
        (a if a < nearest else a - 1, b if b < nearest else b - 1) for a, b in beams
    ]
    rects[:] = [(a, b) for a, b in rects if a != nearest and b != nearest]
    rects[:] = [
        (a if a < nearest else a - 1, b if b < nearest else b - 1) for a, b in rects
    ]
    circles[:] = [(a, b) for a, b in circles if a != nearest and b != nearest]
    circles[:] = [
        (a if a < nearest else a - 1, b if b < nearest else b - 1) for a, b in circles
    ]
    nodes.pop(nearest)
    for hi in list(drag_node.keys()):
        dni = drag_node[hi]
        if dni == nearest:
            del drag_node[hi]
        elif dni > nearest:
            drag_node[hi] = dni - 1

def draw_hud_frame(img, pulse):
    p = abs(math.sin(pulse * 0.5))
    c = tuple(int(x * (0.3 + 0.2*p)) for x in CYAN)
    s = 60  # corner size
    t = 2
    # corners
    cv2.line(img, (0,0), (s,0), c, t)
    cv2.line(img, (0,0), (0,s), c, t)
    cv2.line(img, (W,0), (W-s,0), c, t)
    cv2.line(img, (W,0), (W,s), c, t)
    cv2.line(img, (0,H), (s,H), c, t)
    cv2.line(img, (0,H), (0,H-s), c, t)
    cv2.line(img, (W,H), (W-s,H), c, t)
    cv2.line(img, (W,H), (W,H-s), c, t)
    # top center bracket
    mid = W//2
    cv2.line(img, (mid-40,0), (mid+40,0), c, t)
    cv2.line(img, (mid-40,0), (mid-40,8), c, t)
    cv2.line(img, (mid+40,0), (mid+40,8), c, t)

def draw_scanlines(img, alpha=0.04):
    overlay = img.copy()
    for y in range(0, H, 4):
        cv2.line(overlay, (0,y), (W,y), (0,0,0), 1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

def draw_hud_text(img, nodes, beams, rects, circles, fps, pulse, quote_idx, mode_idx, mode_name, globe_on):
    p = abs(math.sin(pulse))
    c = tuple(int(x*(0.5+0.5*p)) for x in CYAN)

    def txt(s, pos, scale=0.45, color=None):
        cv2.putText(img, s, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color or CYAN_DIM, 1)

    # top left
    txt("STARK INDUSTRIES", (12, 20), 0.5, CYAN)
    txt(f"JARVIS v2.0 // ACTIVE", (12, 38), 0.38)
    mode_line = (
        f"GLOBE // G to sketch"
        if globe_on
        else f"MODE: {mode_name} (M)   N:{len(nodes):02d}  L:{len(beams):02d}  R:{len(rects):02d}  O:{len(circles):02d}"
    )
    txt(mode_line, (12, 54), 0.34)

    # center tip (rotating quote)
    tip = (
        "Globe: one-hand pinch = spin | two-hand pinch = drag to move, spread/pinch to zoom."
        if globe_on
        else QUOTES[quote_idx % len(QUOTES)]
    )
    tw, _ = cv2.getTextSize(tip, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
    tx = max(8, (W - tw) // 2)
    txt(tip, (tx, 78), 0.38, tuple(int(x * 0.75) for x in CYAN))

    # top right
    txt(f"FPS {int(fps):02d}", (W-70, 20), 0.45, CYAN)
    txt("SYS OK", (W-70, 38), 0.38)

    # bottom left
    txt("PINCH place | near node = grab | PALM = del node | BS = del last node", (12, H-52), 0.32)
    txt("BOTH PINCH = commit LINE / RECT / CIRCLE (see MODE) — preview while holding", (12, H-38), 0.31)
    txt("B=undo line  R=undo rect  O=undo circle  M=next mode  C=clear all", (12, H-24), 0.32)
    txt("RECT: diagonal anchors (axis box)  CIRCLE: hand0=center hand1=rim", (12, H-10), 0.30)

    # bottom right — fake vitals
    txt(f"ARC REACTOR: {int(95+5*p)}%", (W-180, H-24), 0.38, c)
    txt(f"SUIT PWR: NOMINAL", (W-180, H-10), 0.38)

def vignette(img, strength=0.22):
    rows, cols = img.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols*0.6)
    kernel_y = cv2.getGaussianKernel(rows, rows*0.6)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    tinted = np.zeros_like(img)
    tinted[:,:,0] = (mask * 45).astype(np.uint8)
    cv2.addWeighted(tinted, strength, img, 1.0 - strength, 0, img)

def _rot_y_3d(p, ang):
    c, s = math.cos(ang), math.sin(ang)
    x, y, z = p
    return (x * c + z * s, y, -x * s + z * c)

def _rot_x_3d(p, ang):
    c, s = math.cos(ang), math.sin(ang)
    x, y, z = p
    return (x, y * c - z * s, y * s + z * c)

def _globe_transform(p, rx, ry):
    return _rot_x_3d(_rot_y_3d(p, ry), rx)

def _globe_project(p, cx, cy, f, dist):
    x, y, z = p
    denom = dist + z
    if denom < 0.25:
        return None
    return (int(cx + f * x / denom), int(cy - f * y / denom)), z

def _build_globe_polylines(n_merid=18, n_para=17, R=1.0):
    polys = []
    for k in range(n_merid):
        phi = 2 * math.pi * k / n_merid
        pts = []
        for j in range(n_para + 1):
            theta = -math.pi / 2 + math.pi * j / n_para
            x = R * math.cos(theta) * math.cos(phi)
            y = R * math.sin(theta)
            z = R * math.cos(theta) * math.sin(phi)
            pts.append((x, y, z))
        polys.append(("meridian", k, pts))
    n_para_rings = 8
    for j in range(1, n_para_rings):
        theta = -math.pi / 2 + math.pi * j / n_para_rings
        pts = []
        for k in range(n_merid + 1):
            phi = 2 * math.pi * k / n_merid
            x = R * math.cos(theta) * math.cos(phi)
            y = R * math.sin(theta)
            z = R * math.cos(theta) * math.sin(phi)
            pts.append((x, y, z))
        kind = "equator" if abs(theta) < 0.04 else "parallel"
        polys.append((kind, j, pts))
    return polys

GLOBE_POLYLINES = _build_globe_polylines()

def globe_pinch_smooth_update(lm, hand_idx, w, h):
    pinching, pinch_pt = is_pinching(lm, w, h, threshold=38)
    if not pinching:
        pinch_smooth.pop(hand_idx, None)
        return False, pinch_pt
    if hand_idx not in pinch_smooth:
        pinch_smooth[hand_idx] = pinch_pt
        sp = pinch_pt
    else:
        ox, oy = pinch_smooth[hand_idx]
        a = PINCH_SMOOTH
        sp = (
            int(a * pinch_pt[0] + (1.0 - a) * ox),
            int(a * pinch_pt[1] + (1.0 - a) * oy),
        )
        pinch_smooth[hand_idx] = sp
    return True, sp

def draw_hologram_globe(img, rx, ry, pulse, cx, cy, scale=1.0):
    Rpx = min(W, H) * 0.24 * scale
    f = Rpx * 1.35
    dist_cam = 3.8
    overlay = img.copy()
    for kind, _tag, pts in GLOBE_POLYLINES:
        projected = []
        zs = []
        for p in pts:
            tp = _globe_transform(p, rx, ry)
            out = _globe_project(tp, cx, cy, f, dist_cam)
            if out is None:
                projected.append(None)
                zs.append(-999)
            else:
                projected.append(out[0])
                zs.append(out[1])
        is_gold = kind == "equator"
        for i in range(len(projected) - 1):
            a, b = projected[i], projected[i + 1]
            if a is None or b is None:
                continue
            zf = (zs[i] + zs[i + 1]) * 0.5
            alpha = 0.25 + 0.55 * max(0.0, min(1.0, (zf + 1.2) / 2.4))
            if is_gold:
                col = tuple(int(c * (0.5 + 0.45 * alpha)) for c in GOLD)
                t = 2
            else:
                col = tuple(int(c * (0.35 + 0.5 * alpha)) for c in CYAN)
                t = 1
            cv2.line(overlay, a, b, col, t + 1)
            br = tuple(min(255, c + 70) for c in col)
            cv2.line(img, a, b, br, max(1, t - 1))
    cv2.addWeighted(overlay, 0.22, img, 0.78, 0, img)
    pr = int(Rpx * 1.15)
    cv2.circle(img, (cx, cy), pr, tuple(int(c * 0.25) for c in CYAN), 1)
    cv2.putText(
        img,
        "TACTICAL SPHERE // G exit | 1H spin | 2H move + spread/pinch zoom",
        (max(8, cx - 280), cy + pr + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        CYAN_DIM,
        1,
    )
    sc = f"{scale:.2f}x"
    tw, _ = cv2.getTextSize(sc, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
    cv2.putText(
        img,
        sc,
        (cx - tw // 2, cy - pr - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        CYAN_DIM,
        1,
    )

# ── Main loop ─────────────────────────────────────────────────────────────────
prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    frame_count += 1
    pulse += 0.08

    # Readable camera + light HUD wash (was ~7% feed; face was invisible)
    blurred = cv2.GaussianBlur(frame, (0,0), 11)
    dark = cv2.addWeighted(frame, 0.66, blurred, 0.34, 0)
    dark = cv2.addWeighted(dark, 0.84, np.zeros((H, W, 3), dtype=np.uint8), 0.16, 0)
    dark[:, :, 0] = np.clip(dark[:, :, 0].astype(np.int16) + 10, 0, 255).astype(np.uint8)

    draw_alignment_grid(dark)

    # Draw built structures (lines/rects/circles under nodes)
    for i, j in rects:
        if i < len(nodes) and j < len(nodes):
            draw_rect_shape(dark, nodes[i], nodes[j], pulse)
    for ci, ri in circles:
        if ci < len(nodes) and ri < len(nodes):
            draw_circle_shape(dark, nodes[ci], nodes[ri], pulse)
    for i, j in beams:
        if i < len(nodes) and j < len(nodes):
            draw_beam(dark, nodes[i], nodes[j], pulse)
    for idx, node in enumerate(nodes):
        draw_node(dark, node, pulse + idx)

    if globe_mode:
        if globe_cx is None:
            globe_cx = W // 2
            globe_cy = H // 2 - 28
        draw_hologram_globe(dark, globe_rx, globe_ry, pulse, int(globe_cx), int(globe_cy), globe_scale)

    # Hand detection
    pinching_this_frame = {}
    globe_did_two_hand = False

    if result.multi_hand_landmarks:
        if globe_mode and len(result.multi_hand_landmarks) == 2:
            lm0 = result.multi_hand_landmarks[0].landmark
            lm1 = result.multi_hand_landmarks[1].landmark
            ok0, sp0 = globe_pinch_smooth_update(lm0, 0, W, H)
            ok1, sp1 = globe_pinch_smooth_update(lm1, 1, W, H)
            if ok0 and ok1:
                mid = ((sp0[0] + sp1[0]) // 2, (sp0[1] + sp1[1]) // 2)
                d12 = dist(sp0, sp1)
                if globe_two_prev is not None:
                    pmid, pdist = globe_two_prev
                    globe_cx += int((mid[0] - pmid[0]) * 0.92)
                    globe_cy += int((mid[1] - pmid[1]) * 0.92)
                    if pdist > 20:
                        globe_scale *= d12 / pdist
                globe_two_prev = (mid, d12)
                globe_did_two_hand = True
                globe_prev_pinch.clear()
                globe_cx = max(60.0, min(float(W - 60), float(globe_cx)))
                globe_cy = max(60.0, min(float(H - 60), float(globe_cy)))
            else:
                globe_two_prev = None

        for hand_idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            lm = hand_landmarks.landmark
            points = [(int(l.x*W), int(l.y*H)) for l in lm]

            if globe_mode:
                if globe_did_two_hand:
                    conn_offset = 0
                    for conn in CONNECTIONS:
                        p1 = points[conn[0]]
                        p2 = points[conn[1]]
                        alpha = 0.5 + 0.5 * abs(math.sin(pulse + conn_offset))
                        c = tuple(int(x * alpha) for x in GOLD)
                        draw_glowing_line(dark, p1, p2, c, thickness=1)
                        conn_offset += 0.15
                    for tip_idx in FINGER_TIPS:
                        draw_reticle(dark, points[tip_idx], radius=12, color=GOLD)
                    if hand_idx == 0:
                        pr = int(min(W, H) * 0.24 * globe_scale * 1.15)
                        cv2.putText(
                            dark,
                            "MOVE / ZOOM",
                            (int(globe_cx) - 58, int(globe_cy) - pr - 28),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.42,
                            GOLD,
                            1,
                        )
                    continue
                pinching, sp = globe_pinch_smooth_update(lm, hand_idx, W, H)
                pinch_pt = sp if pinching else sp
                if pinching:
                    if hand_idx in globe_prev_pinch:
                        ox, oy = globe_prev_pinch[hand_idx]
                        globe_ry += (sp[0] - ox) * 0.014
                        globe_rx += (oy - sp[1]) * 0.014
                    globe_prev_pinch[hand_idx] = sp
                    globe_rx = max(-1.35, min(1.35, globe_rx))
                else:
                    globe_prev_pinch.pop(hand_idx, None)
                sp = pinch_smooth.get(hand_idx, pinch_pt)
                conn_offset = 0
                for conn in CONNECTIONS:
                    p1 = points[conn[0]]
                    p2 = points[conn[1]]
                    alpha = 0.5 + 0.5 * abs(math.sin(pulse + conn_offset))
                    c = tuple(int(x * alpha) for x in GOLD)
                    draw_glowing_line(dark, p1, p2, c, thickness=1)
                    conn_offset += 0.15
                for tip_idx in FINGER_TIPS:
                    draw_reticle(dark, points[tip_idx], radius=12, color=GOLD)
                if pinching:
                    cv2.putText(
                        dark,
                        "SPIN",
                        (sp[0] + 18, sp[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        GOLD,
                        1,
                    )
                continue

            # Draw hand skeleton in cyan
            conn_offset = 0
            for conn in CONNECTIONS:
                p1 = points[conn[0]]
                p2 = points[conn[1]]
                alpha = 0.5 + 0.5 * abs(math.sin(pulse + conn_offset))
                c = tuple(int(x*alpha) for x in CYAN_DIM)
                draw_glowing_line(dark, p1, p2, c, thickness=1)
                conn_offset += 0.15

            # Fingertip reticles
            for tip_idx in FINGER_TIPS:
                pt = points[tip_idx]
                draw_reticle(dark, pt, radius=12, color=CYAN_DIM)

            # Index tip coords — bottom strip so face stays clear
            index_tip = points[8]
            cv2.putText(
                dark,
                f"IDX {hand_idx}  X:{index_tip[0]} Y:{index_tip[1]}",
                (12, H - 56 - 14 * hand_idx),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                CYAN_DIM,
                1,
            )

            # Pinch detection
            pinching, pinch_pt = is_pinching(lm, W, H, threshold=38)
            sp = pinch_pt
            if pinching:
                if hand_idx not in pinch_smooth:
                    pinch_smooth[hand_idx] = pinch_pt
                else:
                    ox, oy = pinch_smooth[hand_idx]
                    a = PINCH_SMOOTH
                    sp = (
                        int(a * pinch_pt[0] + (1.0 - a) * ox),
                        int(a * pinch_pt[1] + (1.0 - a) * oy),
                    )
                    pinch_smooth[hand_idx] = sp
            else:
                pinch_smooth.pop(hand_idx, None)
                drag_node.pop(hand_idx, None)

            pinching_this_frame[hand_idx] = (pinching, sp if pinching else pinch_pt)

            cooldown = pinch_cooldown.get(hand_idx, 0)
            was_pinching = prev_pinch.get(hand_idx, False)

            # Pinch rising edge first → grab existing or spawn node (then drag same gesture)
            if pinching and not was_pinching:
                nearest_idx = None
                best_d = GRAB_RADIUS + 1
                for ni, n in enumerate(nodes):
                    d = dist(n, sp)
                    if d < best_d:
                        best_d, nearest_idx = d, ni
                if nearest_idx is not None and best_d <= GRAB_RADIUS:
                    drag_node[hand_idx] = nearest_idx
                elif cooldown == 0:
                    nodes.append(sp)
                    pinch_cooldown[hand_idx] = 14
                    drag_node[hand_idx] = len(nodes) - 1

            if pinching:
                di = drag_node.get(hand_idx)
                if di is not None and 0 <= di < len(nodes):
                    nodes[di] = sp
                label = "GRAB" if drag_node.get(hand_idx) is not None else "PINCH"
                col = GOLD if label == "GRAB" else CYAN
                cv2.circle(dark, sp, 18, col, 2)
                cv2.circle(dark, sp, 5, col, -1)
                cv2.putText(dark, label, (sp[0] + 20, sp[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

            # Open palm → delete nearest node
            if is_open_palm(lm, W, H) and len(nodes) > 0:
                palm_pt = points[0]
                nearest = min(range(len(nodes)), key=lambda i: dist(nodes[i], palm_pt))
                if dist(nodes[nearest], palm_pt) < 80:
                    remove_node_index(nearest)

            prev_pinch[hand_idx] = pinching
            if cooldown > 0:
                pinch_cooldown[hand_idx] = cooldown - 1

        # Both pinches → shape commit (disabled in globe mode)
        if not globe_mode:
            both_pinch_now = False
            if len(result.multi_hand_landmarks) == 2 and len(nodes) >= 2:
                p0 = pinching_this_frame.get(0, (False, None))
                p1 = pinching_this_frame.get(1, (False, None))
                if p0[0] and p1[0]:
                    both_pinch_now = True
                    n0 = min(range(len(nodes)), key=lambda x: dist(nodes[x], p0[1]))
                    n1 = min(range(len(nodes)), key=lambda x: dist(nodes[x], p1[1]))
                    if n0 != n1 and both_pinch_now and not prev_both_pinch:
                        if shape_mode == 0:
                            if (n0, n1) not in beams and (n1, n0) not in beams:
                                beams.append((n0, n1))
                        elif shape_mode == 1:
                            t = tuple(sorted((n0, n1)))
                            if t not in rects:
                                rects.append(t)
                        else:
                            if (n0, n1) not in circles:
                                circles.append((n0, n1))
            prev_both_pinch = both_pinch_now
        else:
            prev_both_pinch = False
    else:
        prev_both_pinch = False
        globe_prev_pinch.clear()

    # Live beam preview + anchor highlights (after physics, before final HUD)
    if (
        not globe_mode
        and result.multi_hand_landmarks
        and len(result.multi_hand_landmarks) == 2
        and len(nodes) >= 2
    ):
        p0 = pinching_this_frame.get(0, (False, None))
        p1 = pinching_this_frame.get(1, (False, None))
        if p0[0] and p1[0]:
            n0 = min(range(len(nodes)), key=lambda x: dist(nodes[x], p0[1]))
            n1 = min(range(len(nodes)), key=lambda x: dist(nodes[x], p1[1]))
            if n0 != n1:
                a, b = nodes[n0], nodes[n1]
                if shape_mode == 0:
                    linked = (n0, n1) in beams or (n1, n0) in beams
                    draw_beam_preview(dark, a, b, pulse, linked)
                    ah = GOLD if not linked else CYAN_DIM
                elif shape_mode == 1:
                    linked = tuple(sorted((n0, n1))) in rects
                    draw_rect_preview(dark, a, b, pulse, linked)
                    ah = GOLD if not linked else CYAN_DIM
                else:
                    linked = (n0, n1) in circles
                    draw_circle_preview(dark, a, b, pulse, linked)
                    ah = BLUE if not linked else CYAN_DIM
                draw_anchor_highlight(dark, a, pulse + n0, ah)
                draw_anchor_highlight(dark, b, pulse + n1, ah)

    quote_idx = (frame_count // 420) % len(QUOTES)

    # HUD overlays
    draw_scanlines(dark)
    draw_hud_frame(dark, pulse)
    vignette(dark)
    now = time.time()
    fps = 1.0 / (now - prev_time + 1e-9)
    prev_time = now
    draw_hud_text(
        dark,
        nodes,
        beams,
        rects,
        circles,
        fps,
        pulse,
        quote_idx,
        shape_mode,
        MODE_NAMES[shape_mode],
        globe_mode,
    )

    cv2.imshow("JARVIS", dark)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('g'):
        globe_mode = not globe_mode
        globe_prev_pinch.clear()
        globe_two_prev = None
        prev_both_pinch = False
        if globe_mode:
            pinch_smooth.clear()
            globe_cx = float(W // 2)
            globe_cy = float(H // 2 - 28)
            globe_scale = 1.0
    elif key == ord('m'):
        shape_mode = (shape_mode + 1) % 3
    elif key == ord('b') and beams:
        beams.pop()
    elif key == ord('r') and rects:
        rects.pop()
    elif key == ord('o') and circles:
        circles.pop()
    elif key in (8, 127) and nodes:
        remove_node_index(len(nodes) - 1)
    elif key == ord('c'):
        nodes.clear()
        beams.clear()
        rects.clear()
        circles.clear()
        drag_node.clear()
        pinch_smooth.clear()

cap.release()
cv2.destroyAllWindows()