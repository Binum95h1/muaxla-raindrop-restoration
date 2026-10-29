import csv,hashlib,json,math,os,random
from pathlib import Path
from collections import defaultdict
from functools import lru_cache
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
REFERENCE=json.loads((Path(__file__).parent/"reference_split.json").read_text())
def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def save_json(value,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp'); temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8'); os.replace(temp,path)

def save_torch(value,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp'); torch.save(value,temp); os.replace(temp,path)

def rgb(path):
    with Image.open(path) as im:
        if im.getexif().get(274,1)!=1: raise RuntimeError('EXIF rotation cần chuẩn hóa: '+str(path))
        return np.array(im.convert('RGB'),dtype=np.uint8)

def mask(path):
    with Image.open(path) as im:
        if im.getexif().get(274,1)!=1: raise RuntimeError('Mask có EXIF rotation: '+str(path))
        a=np.array(im.convert('L'),dtype=np.uint8)
    if not np.isin(a,[0,1,255]).all(): raise RuntimeError('Mask phải là nhị phân 0/1 hoặc 0/255: '+str(path))
    return (a>0).astype(np.uint8)

def find_root(base='/kaggle/input',explicit=''):
    if explicit:
        roots=[Path(explicit)]
    else:
        roots=sorted({p.parent for p in Path(base).rglob('drop') if p.is_dir() and (p.parent/'clean'/'gt').is_dir() and (p.parent/'mask_drop').is_dir()})
    if len(roots)!=1: raise RuntimeError('Cần đúng một bộ Muaxla có drop, clean/gt, mask_drop. Tìm thấy: '+str(roots)+'. Điền DATA_ROOT nếu cần.')
    return roots[0]

def index_folder(folder):
    index={}
    for p in sorted(Path(folder).rglob('*')):
        if p.is_file() and p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}:
            key=p.parent.name.lower()+'/'+p.stem.lower()
            if key in index: raise RuntimeError('Trùng mã ảnh: '+key+' ở '+str(folder))
            index[key]=p
    return index

def audit(root,out):
    root=Path(root); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    indices=[index_folder(root/f) for f in ['drop','clean/gt','mask_drop']]
    expected={r['key'] for r in REFERENCE}
    for label,index in zip(['Drop','Clean','Mask'],indices):
        if set(index)!=expected:
            raise RuntimeError(f'{label}: thiếu {len(expected-set(index))}, dư {len(set(index)-expected)} ảnh; không đúng phiên bản Muaxla đã chia nhóm.')
    file_stamp=digest([(k,str(p.resolve()),p.stat().st_size,p.stat().st_mtime_ns) for index in indices for k,p in sorted(index.items())])
    cache=out/'data_audit.json'; ref_hash=digest(REFERENCE)
    if cache.is_file():
        old=json.loads(cache.read_text(encoding='utf-8'))
        if old.get('file_stamp')==file_stamp and old.get('reference_hash')==ref_hash:
            print('Dùng lại kiểm tra dữ liệu của cùng phiên/filesystem.',flush=True)
            rows=[dict(r,**{name:str(index[r['key']]) for name,index in zip(['drop_path','clean_path','mask_path'],indices)}) for r in old['rows']]
            return rows,old['data_signature']
    representatives=set()
    for group in sorted({r['scene_group'] for r in REFERENCE}):
        rs=sorted([r for r in REFERENCE if r['scene_group']==group],key=lambda r:r['key'])
        representatives.update(rs[i]['key'] for i in sorted({0,len(rs)//2,len(rs)-1}))
    rows=[]; alignment=[]; seen={}
    for n,ref in enumerate(REFERENCE,1):
        rain,clean,m=rgb(indices[0][ref['key']]),rgb(indices[1][ref['key']]),mask(indices[2][ref['key']])
        h,w=clean.shape[:2]
        if rain.shape!=clean.shape or m.shape!=(h,w) or (w,h)!=(int(ref['width']),int(ref['height'])):
            raise RuntimeError('Bộ ba khác kích thước: '+ref['key'])
        ch=hashlib.sha256(str((w,h)).encode()+clean.tobytes()).hexdigest()
        mh=hashlib.sha256(m.tobytes()).hexdigest()
        if ch!=ref['clean_hash'] or mh!=ref['mask_hash']: raise RuntimeError('Clean/Mask khác dữ liệu nguồn: '+ref['key'])
        if ch in seen and seen[ch]!=ref['split'] and clean.mean(2).std()>=3:
            raise RuntimeError('Clean trùng giữa các split: '+ref['key'])
        seen[ch]=ref['split']
        rh=hashlib.sha256(str((w,h)).encode()+rain.tobytes()).hexdigest()
        row=dict(ref,drop_hash=rh,mask_fraction=float(m.mean()),width=w,height=h)
        rows.append(row)
        if ref['key'] in representatives and (m==0).any():
            error=np.abs(rain.astype(np.float32)-clean.astype(np.float32)).mean(2)/255
            bg=error[m==0]; flag=bool(bg.mean()>.08 and (bg>5/255).mean()>.25)
            alignment.append(dict(key=ref['key'],background_mae=float(bg.mean()),flag=flag))
        if n%100==0 or n==len(REFERENCE): print(f'Kiểm tra bộ ba {n}/{len(REFERENCE)}',flush=True)
    if sum(a['flag'] for a in alignment)>max(3,int(.05*len(alignment))):
        save_json(alignment,out/'alignment_flagged.json')
        raise RuntimeError('Nhiều bộ Drop/Clean lệch ngoài mask. Xem alignment_flagged.json trước khi train.')
    data_signature=digest(rows)
    save_json(dict(file_stamp=file_stamp,reference_hash=ref_hash,data_signature=data_signature,rows=rows,alignment=alignment,
        note='Scene-folder groups inherited from audited mask split; timestamps/FPS not inferred.'),cache)
    for row in rows:
        row.update({name:str(index[row['key']]) for name,index in zip(['drop_path','clean_path','mask_path'],indices)})
    return rows,data_signature

def tensor(a):
    return torch.from_numpy(np.ascontiguousarray(a.transpose(2,0,1))).float()/255

class Patches(Dataset):
    def __init__(self,rows,cfg,start,count):
        self.cfg=cfg; self.start=start; self.count=count; self.groups=defaultdict(list)
        for r in rows: self.groups[r['scene_group']].append(r)
        self.names=sorted(self.groups)
    @lru_cache(maxsize=2)
    def load(self,key,drop,clean,mp): return rgb(drop),rgb(clean),mask(mp)
    def __len__(self): return self.count
    def __getitem__(self,index):
        rng=np.random.default_rng(self.cfg['seed']+self.start+index)
        row=rng.choice(self.groups[rng.choice(self.names)])
        rain,clean,m=self.load(row['key'],row['drop_path'],row['clean_path'],row['mask_path'])
        h,w=m.shape; crop=self.cfg['crop']; mode=rng.random()
        if mode<self.cfg['clean_fraction']: rain=clean; m=np.zeros_like(m)
        elif mode<self.cfg['clean_fraction']+self.cfg['centered_drop_fraction'] and m.any():
            ys,xs=np.nonzero(m); i=rng.integers(len(ys)); cy,cx=int(ys[i]),int(xs[i])
            y=int(np.clip(cy-rng.integers(crop),0,max(0,h-crop))); x=int(np.clip(cx-rng.integers(crop),0,max(0,w-crop)))
        if mode<self.cfg['clean_fraction'] or mode>=self.cfg['clean_fraction']+self.cfg['centered_drop_fraction'] or not m.any():
            y=int(rng.integers(max(1,h-crop+1))); x=int(rng.integers(max(1,w-crop+1)))
        arrays=[rain[y:y+crop,x:x+crop],clean[y:y+crop,x:x+crop],m[y:y+crop,x:x+crop]]
        if h<crop or w<crop:
            raise RuntimeError('Ảnh train nhỏ hơn crop; đổi crop trước khi train, không resize nhãn.')
        k=int(rng.integers(4)); flip=bool(rng.integers(2))
        arrays=[np.rot90(a,k).copy() for a in arrays]
        if flip: arrays=[np.flip(a,axis=1).copy() for a in arrays]
        return tensor(arrays[0]),tensor(arrays[1]),torch.from_numpy(arrays[2].copy()).float()[None]

def ssim(a,b):
    # RGB, range=1, 11x11 Gaussian sigma=1.5, population covariance, valid interior.
    a=a.float(); b=b.float(); size=min(11,a.shape[-2],a.shape[-1]); size-=1-size%2
    coords=torch.arange(size,device=a.device,dtype=a.dtype)-(size-1)/2
    g=torch.exp(-coords.square()/(2*1.5**2)); g=g/g.sum(); channels=a.shape[1]
    ky=g.view(1,1,size,1).expand(channels,1,size,1); kx=g.view(1,1,1,size).expand(channels,1,1,size)
    def blur(x): return F.conv2d(F.conv2d(x,ky,groups=channels),kx,groups=channels)
    ma,mb=blur(a),blur(b); va=(blur(a*a)-ma*ma).clamp_min(0); vb=(blur(b*b)-mb*mb).clamp_min(0); cov=blur(a*b)-ma*mb
    score=((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))
    return score.mean((1,2,3))

def masked_mean(error,m):
    count=m.sum((1,2,3))*error.shape[1]; valid=count>0
    value=(error*m).sum((1,2,3))/count.clamp_min(1)
    return value[valid].mean() if valid.any() else error.sum()*0

def starts(length,tile,overlap):
    if length<=tile: return [0]
    return sorted(set(list(range(0,length-tile+1,tile-overlap))+[length-tile]))

def write_csv(rows,path):
    if not rows: return
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
