"""
DEEPFAKE DETECTION — Final Year Project
========================================
Training runs in terminal:
    python deepfake_detector.py --train

UI (predict only):
    streamlit run deepfake_detector.py
"""

import sys, os, gc, json, random, argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from pathlib import Path
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

try:
    import librosa
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "librosa", "-q"])
    import librosa

# ═══════════════════════════════════════════════════════════
# CONFIG — edit paths here before running
# ═══════════════════════════════════════════════════════════
CFG = {
    "video_real":  r"C:\Users\HP\Desktop\python\Deepfake_detect\real",
    "video_fake":  r"C:\Users\HP\Desktop\python\Deepfake_detect\fake",
    "image_real":  r"C:\Users\HP\Desktop\python\archive (2)\real",
    "image_fake":  r"C:\Users\HP\Desktop\python\archive (2)\fake",
    "audio_real":  r"C:\Users\HP\Desktop\python\Deepfake_detect\audio_real",
    "audio_fake":  r"C:\Users\HP\Desktop\python\Deepfake_detect\audio_fake",
    "cache_dir":   r"C:\Users\HP\Desktop\python\Deepfake_detect\cache",
    "model_video": r"C:\Users\HP\Desktop\python\Deepfake_detect\model_video.pth",
    "model_image": r"C:\Users\HP\Desktop\python\Deepfake_detect\model_image.pth",
    "model_audio": r"C:\Users\HP\Desktop\python\Deepfake_detect\model_audio.pth",
    "batch_size":  8,
    "epochs":      10,
    "lr":          1e-4,
    "max_videos":  1000,
    "max_images":  1000,
    "frames_each": 3,
}

IMG_SIZE    = 224
AUDIO_FEAT  = 164
VISUAL_FEAT = 1280
FUSED_DIM   = VISUAL_FEAT + AUDIO_FEAT
TEMPERATURE = 1.5
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ═══════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════

class VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        for p in base.features.parameters():
            p.requires_grad = False
        self.features = base.features
        self.pool     = base.avgpool
        self.flat     = nn.Flatten()
    def forward(self, x):
        return self.flat(self.pool(self.features(x)))


class VideoFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_enc = VisualEncoder()
        self.fusion = nn.Sequential(
            nn.Linear(FUSED_DIM, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 2),
        )
    def forward(self, img, aud):
        return self.fusion(torch.cat([self.visual_enc(img), aud], dim=1))


class ImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        for i, blk in enumerate(base.features):
            for p in blk.parameters():
                p.requires_grad = (i >= 5)
        in_f = base.classifier[1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(in_f, 256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, 2),
        )
        self.net = base
    def forward(self, x):
        return self.net(x)


class AudioModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(AUDIO_FEAT, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )
    def forward(self, x):
        return self.net(x)


# ═══════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════

def preprocess(frames):
    arr = np.stack(frames).astype(np.float32) / 255.0
    return (torch.FloatTensor(arr).permute(0,3,1,2) - MEAN) / STD


def extract_frames(path, num=None):
    num = num or CFG['frames_each']
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < num: cap.release(); return []
    frames = []
    for i in np.linspace(0, total-1, num, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok:
            frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE)))
    cap.release()
    return frames


def load_image_file(path):
    img = cv2.imread(str(path))
    if img is None: return None
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE))


def extract_audio_features(path):
    try:
        y, sr = librosa.load(path, sr=16000, mono=True, duration=30)
        if len(y) < sr: return None
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        feat = np.concatenate([np.mean(mfcc,1), np.std(mfcc,1),
                               np.max(mfcc,1), np.min(mfcc,1)])
        extras = [
            np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
            np.mean(librosa.feature.zero_crossing_rate(y)),
            np.mean(librosa.feature.rms(y=y)),
        ]
        feat = np.append(feat, extras).astype(np.float32)
        return feat / (np.abs(feat).max() + 1e-6)
    except:
        return None


# ═══════════════════════════════════════════════════════════
# GRAD-CAM
# ═══════════════════════════════════════════════════════════

class GradCAM:
    def __init__(self, model_net):
        self.grads = self.acts = None
        model_net.features[-1].register_forward_hook(
            lambda m,i,o: setattr(self, 'acts', o.detach()))
        model_net.features[-1].register_full_backward_hook(
            lambda m,i,o: setattr(self, 'grads', o[0].detach()))
        self.model = model_net

    def generate(self, t, class_idx=1):
        self.model.zero_grad()
        self.model(t)[0, class_idx].backward()
        w   = self.grads.mean(dim=[2,3], keepdim=True)
        cam = F.relu((w * self.acts).sum(dim=1).squeeze()).cpu().numpy()
        cam = cam / (cam.max() + 1e-6)
        return cv2.resize(cam, (IMG_SIZE, IMG_SIZE))


def overlay_heatmap(img, cam):
    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return (0.5*img + 0.5*heat).astype(np.uint8)


# ═══════════════════════════════════════════════════════════
# TEMPORAL INCONSISTENCY
# ═══════════════════════════════════════════════════════════

def temporal_inconsistency(frames, encoder):
    if len(frames) < 2: return 0.0
    encoder.eval()
    with torch.no_grad():
        embs = encoder(preprocess(frames).to(device)).cpu().numpy()
    diffs = []
    for i in range(len(embs)-1):
        a, b = embs[i], embs[i+1]
        cos  = np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-6)
        diffs.append(1.0 - cos)
    return min(float(np.mean(diffs)) / 0.3, 1.0)


# ═══════════════════════════════════════════════════════════
# DISK CACHE
# ═══════════════════════════════════════════════════════════

def _save(arr, path): np.save(path, arr)


def build_video_cache():
    os.makedirs(CFG['cache_dir'], exist_ok=True)
    idx_path = os.path.join(CFG['cache_dir'], "video_index.json")
    r_vids = list(Path(CFG['video_real']).glob('*.mp4')) + list(Path(CFG['video_real']).glob('*.avi'))
    f_vids = list(Path(CFG['video_fake']).glob('*.mp4')) + list(Path(CFG['video_fake']).glob('*.avi'))
    random.shuffle(r_vids); random.shuffle(f_vids)
    lim = min(len(r_vids), len(f_vids), CFG['max_videos'])
    r_vids, f_vids = r_vids[:lim], f_vids[:lim]
    index = []; done = 0; total = len(r_vids)+len(f_vids)
    for label, vids in [(0,r_vids),(1,f_vids)]:
        for v in vids:
            stem = f"vid_{label}_{v.stem}"
            fp   = os.path.join(CFG['cache_dir'], f"{stem}_frames.npy")
            ap   = os.path.join(CFG['cache_dir'], f"{stem}_audio.npy")
            if not os.path.exists(fp):
                frs = extract_frames(str(v))
                if frs: _save(np.stack(frs).astype(np.uint8), fp)
            if not os.path.exists(ap):
                af = extract_audio_features(str(v))
                if af is not None: _save(af, ap)
            has_f = os.path.exists(fp); has_a = os.path.exists(ap)
            if has_f:
    # If no audio, save a zero vector so video model can still train
                if not has_a:
                    _save(np.zeros(AUDIO_FEAT, dtype=np.float32), ap)
                    has_a = True
            index.append({'stem':stem,'label':label,'has_frames':has_f,'has_audio':has_a})
            done+=1; print(f"  Video cache {done}/{total}: {v.name}", flush=True)
    with open(idx_path,'w') as f: json.dump(index, f)
    print(f"  Video cache complete: {len(index)} videos"); return index


def build_image_cache():
    os.makedirs(CFG['cache_dir'], exist_ok=True)
    idx_path = os.path.join(CFG['cache_dir'], "image_index.json")
    exts = ['*.jpg','*.jpeg','*.png','*.bmp']
    index = []
    for label, folder in [(0,CFG['image_real']),(1,CFG['image_fake'])]:
        if not os.path.exists(folder):
            print(f"  Image folder not found: {folder}"); continue
        files = [f for ext in exts for f in Path(folder).glob(ext)]
        random.shuffle(files); files = files[:CFG['max_images']]
        print(f"  Image cache: {len(files)} {'real' if label==0 else 'fake'} images")
        for i, f in enumerate(files):
            stem  = f"img_{label}_{f.stem}_{i}"
            fpath = os.path.join(CFG['cache_dir'], f"{stem}_frames.npy")
            if not os.path.exists(fpath):
                img = load_image_file(str(f))
                if img is not None: _save(np.stack([img]).astype(np.uint8), fpath)
            if os.path.exists(fpath):
                index.append({'stem':stem,'label':label,'has_frames':True,'has_audio':False})
            if (i+1)%100==0: print(f"    {i+1}/{len(files)}", flush=True)
    # Also add video frames
    vid_idx = os.path.join(CFG['cache_dir'], "video_index.json")
    if os.path.exists(vid_idx):
        with open(vid_idx) as f: vi = json.load(f)
        for e in vi:
            if e['has_frames']:
                index.append({'stem':e['stem'],'label':e['label'],'has_frames':True,'has_audio':False})
        print(f"  Added {len(vi)} video frame sets to image training")
    with open(idx_path,'w') as f: json.dump(index, f)
    print(f"  Image cache complete: {len(index)} entries"); return index


def build_audio_cache():
    os.makedirs(CFG['cache_dir'], exist_ok=True)
    idx_path = os.path.join(CFG['cache_dir'], "audio_index.json")
    exts = ['*.wav','*.flac','*.mp3']
    index = []
    for label, folder in [(0,CFG['audio_real']),(1,CFG['audio_fake'])]:
        if not os.path.exists(folder):
            print(f"  Audio folder not found: {folder}"); continue
        files = [f for ext in exts for f in Path(folder).glob(ext)]
        random.shuffle(files); files = files[:500]
        print(f"  Audio cache: {len(files)} {'real' if label==0 else 'fake'} audio files")
        for i, f in enumerate(files):
            stem  = f"aud_{label}_{f.stem}_{i}"
            apath = os.path.join(CFG['cache_dir'], f"{stem}_audio.npy")
            if not os.path.exists(apath):
                af = extract_audio_features(str(f))
                if af is not None: _save(af, apath)
            if os.path.exists(apath):
                index.append({'stem':stem,'label':label,'has_frames':False,'has_audio':True})
            if (i+1)%100==0: print(f"    {i+1}/{len(files)}", flush=True)
    # Fallback to video audio
    vid_idx = os.path.join(CFG['cache_dir'], "video_index.json")
    if os.path.exists(vid_idx) and len(index) < 20:
        with open(vid_idx) as f: vi = json.load(f)
        for e in vi:
            if e.get('has_audio'):
                index.append({'stem':e['stem'],'label':e['label'],'has_frames':False,'has_audio':True})
        print(f"  Added video audio as fallback: {len(index)} total")
    with open(idx_path,'w') as f: json.dump(index, f)
    print(f"  Audio cache complete: {len(index)} entries"); return index


# ═══════════════════════════════════════════════════════════
# LOAD TENSORS FROM CACHE
# ═══════════════════════════════════════════════════════════

def augment_image(img):
    """Applied only during training to stop the model latching onto
    compression/pipeline shortcuts instead of real facial artifacts."""
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    if random.random() < 0.4:
        quality = random.randint(50, 95)
        _, enc = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, quality])
        img = cv2.cvtColor(cv2.imdecode(enc, 1), cv2.COLOR_BGR2RGB)
    if random.random() < 0.3:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if random.random() < 0.3:
        alpha = random.uniform(0.85, 1.15)   # brightness/contrast jitter
        img = np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    if random.random() < 0.2:
        noise = np.random.normal(0, random.uniform(2, 8), img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img

def load_tensors(index, mode):
    """Loads data in small chunks to avoid RAM spikes."""
    fl, al, ll = [], [], []
    CHUNK = 2000

    if mode == 'image':
        by_label = {}
        for e in index:
            by_label.setdefault(e['label'], []).append(e)
        for lbl in by_label:
            random.shuffle(by_label[lbl])
        per_class = CHUNK // max(len(by_label), 1)
        working_index = []
        for lbl, items in by_label.items():
            working_index.extend(items[:per_class])
        random.shuffle(working_index)
    else:
        working_index = index

    for e in working_index:
        stem = e['stem']; cd = CFG['cache_dir']
        fp = os.path.join(cd, f"{stem}_frames.npy")
        ap = os.path.join(cd, f"{stem}_audio.npy")
        frs = np.load(fp) if e.get('has_frames') and os.path.exists(fp) else None
        af  = np.load(ap) if e.get('has_audio')  and os.path.exists(ap) else None

        if mode == 'video' and frs is not None and af is not None:
            fl.append(frs[0]); al.append(af); ll.append(e['label'])
        elif mode == 'image' and frs is not None:
            for fr in frs[:1]:
                fl.append(augment_image(fr))   # <-- augmentation applied here
                ll.append(e['label'])
        elif mode == 'audio' and af is not None:
            al.append(af); ll.append(e['label'])

    if mode == 'image':
        from collections import Counter
        print(f"  [DEBUG] image class balance: {Counter(ll)}")

    def to_t(f):
        arr = np.stack(f).astype(np.float32) / 255.0
        return (torch.FloatTensor(arr).permute(0,3,1,2) - MEAN) / STD

    if mode == 'video' and fl:
        return (to_t(fl), torch.FloatTensor(np.stack(al)), torch.LongTensor(ll))
    if mode == 'image' and fl:
        return (to_t(fl), torch.LongTensor(ll))
    if mode == 'audio' and al:
        return (torch.FloatTensor(np.stack(al)), torch.LongTensor(ll))
    return None

# ═══════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════

def make_loaders(tensors):
    y = tensors[-1].numpy(); idx = np.arange(len(y))
    tr, vl = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
    def loader(i, sh):
        return DataLoader(TensorDataset(*[t[i] for t in tensors]),
                          batch_size=CFG['batch_size'], shuffle=sh)
    return loader(tr, True), loader(vl, False)


def train_model(model, loaders, save_path, name):
    tr_l, vl_l = loaders
    opt   = optim.Adam([p for p in model.parameters() if p.requires_grad],
                       lr=CFG['lr'], weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG['epochs'])
    crit  = nn.CrossEntropyLoss()
    best  = 0.0
    print(f"\n{'='*50}\nTraining: {name}\n{'='*50}")
    for ep in range(CFG['epochs']):
        model.train(); tc=tt=0
        for batch in tr_l:
            *inp, tgt = [b.to(device) for b in batch]
            opt.zero_grad()
            out = model(*inp); crit(out, tgt).backward(); opt.step()
            tc += out.max(1)[1].eq(tgt).sum().item(); tt += tgt.size(0)
        model.eval(); vc=vt=0
        with torch.no_grad():
            for batch in vl_l:
                *inp, tgt = [b.to(device) for b in batch]
                vc += model(*inp).max(1)[1].eq(tgt).sum().item(); vt += tgt.size(0)
        ta=100.*tc/max(tt,1); va=100.*vc/max(vt,1)
        sched.step()
        print(f"  Epoch {ep+1:02d}/{CFG['epochs']} | Train {ta:.1f}% | Val {va:.1f}%", flush=True)
        if va > best: best=va; torch.save(model.state_dict(), save_path)
    print(f"  Best val: {best:.1f}% — saved to {save_path}")
    return best


def run_training():
    print(f"\nDevice: {device}")
    # ── NEW: fix silent videos with zero audio vectors ──
    vid_idx_path = os.path.join(CFG['cache_dir'], "video_index.json")
    if os.path.exists(vid_idx_path):
        with open(vid_idx_path) as f:
            idx = json.load(f)
        fixed = 0
        for e in idx:
            ap = os.path.join(CFG['cache_dir'], f"{e['stem']}_audio.npy")
            if not os.path.exists(ap):
                np.save(ap, np.zeros(AUDIO_FEAT, dtype=np.float32))
                e['has_audio'] = True
                fixed += 1
        if fixed:
            with open(vid_idx_path, 'w') as f:
                json.dump(idx, f)
            print(f"  Fixed {fixed} silent videos with zero audio vectors")

    # --- Build caches ---
    print("\n[1/6] Video cache")
    vid_idx_path = os.path.join(CFG['cache_dir'], "video_index.json")
    if os.path.exists(vid_idx_path):
        print("  Already exists — skipping (delete video_index.json to rebuild)")
        with open(vid_idx_path) as f: vid_index = json.load(f)
    else:
        vid_index = build_video_cache()

    print("\n[2/6] Image cache")
    img_idx_path = os.path.join(CFG['cache_dir'], "image_index.json")
    if os.path.exists(img_idx_path):
        print("  Already exists — skipping")
        with open(img_idx_path) as f: img_index = json.load(f)
    else:
        img_index = build_image_cache()

    print("\n[3/6] Audio cache")
    aud_idx_path = os.path.join(CFG['cache_dir'], "audio_index.json")
    if os.path.exists(aud_idx_path):
        print("  Already exists — skipping")
        with open(aud_idx_path) as f: aud_index = json.load(f)
    else:
        aud_index = build_audio_cache()

    # --- Train models ---
    print("\n[4/6] Video model")
    if os.path.exists(CFG['model_video']):
        print("  Already trained — skipping")
        data=None
    else:
        data = load_tensors(vid_index, 'video')
    if data:
        m = VideoFusionModel().to(device)
        train_model(m, make_loaders(data), CFG['model_video'], "Video Fusion")
        del m, data; gc.collect()

    print("\n[5/6] Image model")
    if os.path.exists(CFG['model_image']):
        print("  Already trained — skipping")
        data = None
    else:
        data = load_tensors(img_index, 'image')
    if data:
        m = ImageModel().to(device)
        orig_epochs = CFG['epochs']
        CFG['epochs'] = 16          # <-- image needs a few more epochs with augmentation
        train_model(m, make_loaders(data), CFG['model_image'], "Image (Grad-CAM)")
        CFG['epochs'] = orig_epochs
        del m, data; gc.collect()
    else:
        print("  No image data — skipping")
    data = None
    

    print("\n[6/6] Audio model")
    data = load_tensors(aud_index, 'audio')
    if data:
        y = data[-1].numpy()
        if (y==0).sum() >= 4 and (y==1).sum() >= 4:
            m = AudioModel().to(device)
            train_model(m, make_loaders(data), CFG['model_audio'], "Audio MFCC")
            del m, data; gc.collect()
        else:
            print(f"  Not enough audio samples — skipping")
    else:
        print("  No audio data — skipping")

    print("\n✓ Training complete.\n")


# ═══════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════

def load_model(cls, path):
    if not os.path.exists(path): return None
    m = cls().to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    m.eval(); return m


def scaled_softmax(logits):
    return torch.softmax(logits / TEMPERATURE, dim=1).cpu().numpy()


def _result(probs, mode, extra=None):
    fp=float(probs[1]); rp=float(probs[0])
    v="FAKE" if fp>=0.5 else "REAL"
    d={'fake_prob':fp,'real_prob':rp,'verdict':v,
       'confidence':fp if v=="FAKE" else rp,'mode':mode}
    if extra: d.update(extra)
    return d


def infer_image(frame_rgb, model):
    t = preprocess([frame_rgb]).to(device)
    cam_gen = GradCAM(model.net)
    model.eval()
    with torch.enable_grad():
        logits  = model(t)
        probs   = scaled_softmax(logits.detach())
        heatmap = cam_gen.generate(t, class_idx=1)
    return _result(probs[0], "image"), overlay_heatmap(frame_rgb, heatmap)

def infer_video(frames, af, model):
    t_f = preprocess(frames).to(device)
    N   = t_f.shape[0]
    t_a = torch.FloatTensor(af).unsqueeze(0).expand(N,-1).to(device)
    with torch.no_grad():
        probs = scaled_softmax(model(t_f, t_a))
    frame_fakes = [float(p[1]) for p in probs]
    avg = probs.mean(axis=0)
    ti  = temporal_inconsistency(frames, model.visual_enc)
    fp  = 0.75*float(avg[1]) + 0.25*ti
    return _result(np.array([1-fp, fp]), "video",
                   extra={'frame_fakes':frame_fakes,'temporal_score':ti})


def infer_audio(af, model):
    if af is None: return None
    t = torch.FloatTensor(af).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = scaled_softmax(model(t))[0]
    return _result(probs, "audio")


# ═══════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════
# REPLACE the entire run_ui() function in your vis_aud2.py
# with this one. Everything else stays the same.
# ═══════════════════════════════════════════════════════════

def run_ui():
    import streamlit as st
    import tempfile

    st.set_page_config(page_title="Deepfake Detector", page_icon="🔍", layout="wide")
    st.sidebar.markdown("## 🔍 Deepfake Detector")

    st.markdown("""
    <style>
    /* ── Sidebar ── */
    [data-testid="stSidebar"] { background:#111317; }
    [data-testid="stSidebar"] * { color:#e0e0e0; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { gap:0; border-bottom:1px solid #2a2d3e; }
    .stTabs [data-baseweb="tab"] {
        background:transparent; border-radius:0;
        color:#888; padding:10px 24px; font-size:14px; font-weight:500;
        border-bottom:2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        color:#fff !important; border-bottom:2px solid #2563eb !important;
        background:transparent !important;
    }

    /* ── Upload box ── */
    [data-testid="stFileUploader"] {
        background:#1a1d27; border:1px solid #2a2d3e;
        border-radius:10px; padding:8px 16px;
    }
    [data-testid="stFileUploader"] label { display:none; }

    /* ── Preview + Result cards ── */
    .preview-box {
        background:#1a1d27; border:1px solid #2a2d3e;
        border-radius:12px; padding:16px;
    }
    .result-box-real {
        background:#0e1a0e; border:1px solid #27ae60;
        border-radius:12px; padding:20px 24px;
    }
    .result-box-fake {
        background:#1a0e0e; border:1px solid #c0392b;
        border-radius:12px; padding:20px 24px;
    }
    .verdict-real { color:#2ecc71; font-size:2rem; font-weight:700; }
    .verdict-fake { color:#e74c3c; font-size:2rem; font-weight:700; }
    .conf-label { color:#aaa; font-size:.9rem; margin-bottom:12px; }
    .bar-row { margin-bottom:6px; }
    .bar-label { color:#ccc; font-size:13px; display:flex; justify-content:space-between; margin-bottom:3px; }
    .bar-bg { background:#2a2d3e; border-radius:4px; height:8px; width:100%; }
    .bar-real { background:#2ecc71; border-radius:4px; height:8px; }
    .bar-fake { background:#e74c3c; border-radius:4px; height:8px; }
    .prediction-msg-real {
        margin-top:14px; padding:10px 14px;
        background:#0d2010; border:1px solid #27ae60; border-radius:8px;
        color:#2ecc71; font-size:13px;
    }
    .prediction-msg-fake {
        margin-top:14px; padding:10px 14px;
        background:#200d0d; border:1px solid #c0392b; border-radius:8px;
        color:#e74c3c; font-size:13px;
    }

    /* ── Tip box ── */
    .tip-box {
        background:#1a1d27; border:1px solid #2a2d3e; border-radius:10px;
        padding:14px 16px; margin-top:16px;
    }
    .tip-title { color:#60a5fa; font-size:13px; font-weight:600; margin-bottom:4px; }
    .tip-body  { color:#888; font-size:12px; }

    /* ── Metric cards ── */
    .metric-box {
        background:#1a1d27; border:1px solid #2a2d3e;
        border-radius:10px; padding:14px; text-align:center; margin-bottom:10px;
    }
    .metric-label { color:#888; font-size:11px; text-transform:uppercase; letter-spacing:1px; }
    .metric-val   { color:#e0e0e0; font-size:1.4rem; font-weight:700; margin-top:4px; }

    footer { visibility:hidden; }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ────────────────────────────────────────────
    with st.sidebar:
        st.divider()

        st.markdown("**DEVICE**")
        st.success(f"✅ Using device: {device}")

        st.divider()
        st.markdown("**MENU**")
        menu = st.radio("Navigation", ["Home", "Predict"], label_visibility="collapsed")

        st.divider()
        st.markdown("""
        <div class="tip-box">
            <div class="tip-title">💡 Tip</div>
            <div class="tip-body">Upload an image, video or audio file to check if it is Real or Fake.</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Helpers ────────────────────────────────────────────
    def result_card(r):
        if not r:
            st.warning("Could not process the file."); return

        v        = r['verdict']
        rp       = r['real_prob'] * 100
        fp       = r['fake_prob'] * 100
        box_cls  = "result-box-real" if v == "REAL" else "result-box-fake"
        v_cls    = "verdict-real"    if v == "REAL" else "verdict-fake"
        icon     = "✅" if v == "REAL" else "🔴"
        msg_cls  = "prediction-msg-real" if v == "REAL" else "prediction-msg-fake"
        msg      = f"This {'image' if r['mode']=='image' else 'file'} is predicted to be {v.capitalize()}."
        rw       = int(rp); fw = int(fp)

        st.markdown(f"""
        <div class="{box_cls}">
            <div style="font-size:12px;color:#888;margin-bottom:6px;">Prediction Result</div>
            <div class="{v_cls}">{icon} {v.capitalize()}</div>
            <div class="conf-label">Confidence: <strong style="color:#{'2ecc71' if v=='REAL' else 'e74c3c'}">{r['confidence']*100:.2f}%</strong></div>
            <div class="bar-row">
                <div class="bar-label"><span>Real</span><span>{rp:.2f}%</span></div>
                <div class="bar-bg"><div class="bar-real" style="width:{rw}%"></div></div>
            </div>
            <div class="bar-row" style="margin-top:8px">
                <div class="bar-label"><span>Fake</span><span>{fp:.2f}%</span></div>
                <div class="bar-bg"><div class="bar-fake" style="width:{fw}%"></div></div>
            </div>
            <div class="{msg_cls}">✅ {msg}</div>
        </div>
        """, unsafe_allow_html=True)

        # Extra info for video
        if 'temporal_score' in r:
            ts = r['temporal_score']
            st.write("")
            st.markdown(f"""<div class="metric-box">
                <div class="metric-label">⏱ Temporal Inconsistency</div>
                <div class="metric-val">{ts*100:.1f}%</div>
                <div style="color:#555;font-size:11px;margin-top:4px">
                    High = face flickers between frames
                </div>
            </div>""", unsafe_allow_html=True)

        if 'frame_fakes' in r:
            st.write("")
            st.markdown("**Per-frame Fake Probability**")
            fakes = r['frame_fakes']
            bars  = ""
            for i, fp2 in enumerate(fakes):
                col = "#e74c3c" if fp2 >= 0.5 else "#2ecc71"
                h   = max(4, int(fp2 * 70))
                bars += (f'<div style="display:inline-block;margin:0 3px;text-align:center">'
                         f'<div style="background:{col};width:26px;height:{h}px;border-radius:4px 4px 0 0"></div>'
                         f'<div style="font-size:9px;color:#666">F{i+1}</div>'
                         f'<div style="font-size:9px;color:{col}">{fp2*100:.0f}%</div></div>')
            st.markdown(
                f'<div style="background:#1a1d27;padding:14px;border-radius:10px;'
                f'display:flex;align-items:flex-end">{bars}</div>',
                unsafe_allow_html=True)

    # ── HOME ───────────────────────────────────────────────
    if menu == "Home":
        st.title("🔍 Deepfake Detection")
        st.markdown("Upload an image, video, or audio file to check whether it is **Real** or **Fake**.")
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.markdown("🖼️ **Image**\n\nAnalyses face texture and boundary artefacts. Shows a Grad-CAM heatmap of which region triggered the result.")
        c2.markdown("🎬 **Video**\n\nCombines visual + audio analysis. Shows per-frame fake probability and temporal inconsistency score.")
        c3.markdown("🎵 **Audio**\n\nAnalyses MFCC spectral features to detect voice cloning and TTS synthesis artefacts.")

        st.divider()
        st.markdown("**Model Status**")
        mc1, mc2, mc3 = st.columns(3)
        for col, path, name in [
            (mc1, CFG['model_image'], "Image"),
            (mc2, CFG['model_video'], "Video"),
            (mc3, CFG['model_audio'], "Audio"),
        ]:
            if os.path.exists(path): col.success(f"✅ {name} ready")
            else:                    col.warning(f"⚠️ {name} not trained")

    # ── PREDICT ────────────────────────────────────────────
    elif menu == "Predict":
        st.title("🔍 Deepfake Detection")
        st.markdown("Upload an image or video to detect whether it is **Real** or **Fake**.")

        tab_img, tab_vid, tab_aud = st.tabs(["🖼️  Image", "🎬  Video", "🎵  Audio"])

        # ── IMAGE ──────────────────────────────────────────
        with tab_img:
            img_model = load_model(ImageModel, CFG['model_image'])
            if img_model is None:
                st.error("Image model not found. Run:  `python vis_aud2.py --train`")
            else:
                st.markdown("**Upload Image**")
                up = st.file_uploader("Upload Image", type=["jpg","jpeg","png"],
                                      key="img_up", label_visibility="collapsed")
                if up is not None:
                    tmp = None
                    try:
                        suf = "." + up.name.rsplit('.',1)[-1].lower()
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suf) as t:
                            t.write(up.read()); tmp = t.name

                        img = cv2.cvtColor(cv2.imread(tmp), cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                        with st.spinner("Analysing…"):
                            res, overlay = infer_image(img, img_model)

                        left, right = st.columns(2)
                        with left:
                            st.markdown('<div class="preview-box">', unsafe_allow_html=True)
                            st.markdown("**Preview**")
                            st.image(img, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        with right:
                            result_card(res)

                        st.write("")
                        st.markdown("**🔥 Grad-CAM — regions the model focused on**")
                        gc1, gc2 = st.columns(2)
                        gc1.image(img,     caption="Original",    use_container_width=True)
                        gc2.image(overlay, caption="Grad-CAM heatmap", use_container_width=True)

                    finally:
                        if tmp and os.path.exists(tmp): os.unlink(tmp)

        # ── VIDEO ──────────────────────────────────────────
        with tab_vid:
            vid_model = load_model(VideoFusionModel, CFG['model_video'])
            if vid_model is None:
                st.error("Video model not found. Run training first.")
            else:
                st.markdown("**Upload Video**")
                up = st.file_uploader("Upload Video", type=["mp4","avi"],
                                      key="vid_up", label_visibility="collapsed")
                if up is not None:
                    tmp = None
                    try:
                        suf = "." + up.name.rsplit('.',1)[-1].lower()
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suf) as t:
                            t.write(up.read()); tmp = t.name

                        left, right = st.columns(2)
                        with left:
                            st.markdown('<div class="preview-box">', unsafe_allow_html=True)
                            st.markdown("**Preview**")
                            st.video(tmp)
                            st.markdown('</div>', unsafe_allow_html=True)
                        with right:
                            with st.spinner("Extracting frames and audio…"):
                                frames = extract_frames(tmp, num=12)
                                af     = extract_audio_features(tmp)
                                if af is None:
                                    af = np.zeros(AUDIO_FEAT, dtype=np.float32)
                                res = infer_video(frames, af, vid_model)
                            result_card(res)

                    finally:
                        if tmp and os.path.exists(tmp): os.unlink(tmp)

        # ── AUDIO ──────────────────────────────────────────
        with tab_aud:
            aud_model = load_model(AudioModel, CFG['model_audio'])
            if aud_model is None:
                st.error("Audio model not found. Run training first.")
            else:
                st.markdown("**Upload Audio**")
                up = st.file_uploader("Upload Audio", type=["mp3","wav"],
                                      key="aud_up", label_visibility="collapsed")
                if up is not None:
                    tmp = None
                    try:
                        suf = "." + up.name.rsplit('.',1)[-1].lower()
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suf) as t:
                            t.write(up.read()); tmp = t.name

                        left, right = st.columns(2)
                        with left:
                            st.markdown('<div class="preview-box">', unsafe_allow_html=True)
                            st.markdown("**Preview**")
                            st.audio(tmp)
                            st.markdown('</div>', unsafe_allow_html=True)
                        with right:
                            with st.spinner("Extracting audio features…"):
                                af  = extract_audio_features(tmp)
                                res = infer_audio(af, aud_model)
                            result_card(res)

                    finally:
                        if tmp and os.path.exists(tmp): os.unlink(tmp)

# ═══════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    if "--train" in sys.argv:
        run_training()
    else:
        run_ui()