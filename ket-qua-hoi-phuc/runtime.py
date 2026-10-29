"""Muaxla Restormer experiments. Adapted Restormer (MIT), not STRRNet/HFNet reproduction.
The detector is frozen and predicted at inference; ground-truth masks are loss/metric labels only.
"""
import argparse,copy,csv,hashlib,json,math,os,shutil,time,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import data_utils as du
from models import RestorationSystem

DEFAULTS=dict(seed=20261013,crop=384,overlap=128,batch=1,accumulation=4,workers=2,
    architecture=dict(dim=32,blocks=[2,3,3,4],refinement=2,heads=[1,2,4,8],expansion=2.66,detail_width=24),
    baseline_steps=12000,guided_steps=6000,hybrid_steps=4500,refiner_only_steps=1500,
    lr=2e-4,guided_lr=5e-5,hybrid_lr=3e-5,weight_decay=1e-4,warmup=200,
    clean_fraction=.2,centered_drop_fraction=.6,roi_weight=2.,background_weight=.25,ssim_weight=.1,
    edge_weight=.05,frequency_weight=.01,coarse_weight=.2,identity_guard=.01,background_margin=.003,
    save_every=250,evaluate_every=500,validation_per_group=2,final_val_per_group=12,
    test_per_group=12,session_hours=3.,allow_cpu=False,detail_losses=True)
FORMAT='muaxla_restormer_v2';CODE_FILES=['runtime.py','models.py','restormer_arch.py','data_utils.py','reference_split.json','Restormer_LICENSE.txt','README_VI.md','restore.py']
STAGES={'baseline':'rgb','rgb_control':'rgb','guided':'guided','guided_control':'guided','hybrid':'hybrid'}

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def extract(path,dest):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path) as z:
        if sum(i.file_size for i in z.infolist())>2*1024**3:raise RuntimeError('ZIP lớn hơn 2 GiB.')
        for item in z.infolist():
            if item.is_dir():continue
            p=(dest/item.filename).resolve()
            if not p.is_relative_to(dest.resolve()):raise RuntimeError('ZIP path traversal.')
            p.parent.mkdir(parents=True,exist_ok=True)
            with z.open(item) as a,open(p,'wb') as b:shutil.copyfileobj(a,b)
    return dest

def unique(paths,label):
    paths=sorted(set(Path(p).resolve() for p in paths))
    if len(paths)!=1:raise RuntimeError(f'{label}: cần đúng một, có {len(paths)}: {paths}. Điền đường dẫn cụ thể trong Cell cấu hình.')
    return paths[0]

def discover(base,filename,explicit=''):
    return Path(explicit) if explicit else unique(Path(base).rglob(filename),filename)

def resolve_parent(base,stage,explicit=''):
    if explicit:return Path(explicit)
    archives=list(Path(base).rglob(f'muaxla_v2_{stage}_model.zip'))
    if archives:return unique(archives,'Parent ZIP')
    candidates=[]
    for p in Path(base).rglob('best.pt'):
        c=p.parent/'experiment_config.json'
        if c.exists():
            try:
                if json.loads(c.read_text()).get('stage')==stage:candidates.append(p.parent)
            except (ValueError,OSError):pass
    return unique(candidates,'Parent model directory')

def resolve_mask(base,explicit=''):
    if explicit:return Path(explicit)
    archives=list(Path(base).rglob('muaxla_mask_model.zip'))
    return unique(archives,'Muaxla mask ZIP') if archives else unique(Path(base).rglob('mask_best.pt'),'Muaxla mask_best.pt')

def resolve_resume(base,stage,explicit=''):
    if explicit:return Path(explicit)
    archives=list(Path(base).rglob(f'muaxla_v2_{stage}_resume.zip'))
    if archives:return unique(archives,'Resume ZIP')
    candidates=[]
    for p in Path(base).rglob('last.pt'):
        c=p.parent/'experiment_config.json'
        if c.exists():
            try:
                if json.loads(c.read_text()).get('stage')==stage:candidates.append(p.parent)
            except (ValueError,OSError):pass
    return unique(candidates,'Resume folder') if candidates else None

def restore_resume(source,out):
    source,out=Path(source),Path(out)
    if source.is_file():return extract(source,out)
    out.mkdir(parents=True,exist_ok=True)
    for n in ['last.pt','best.pt','experiment_config.json','history.json','selected.json','training_done.json','parent.pt','mask_source.pt','data_audit.json']:
        if (source/n).exists():shutil.copy2(source/n,out/n)
    return out

def subset(rows,split,per_group):
    groups={}
    for r in rows:
        if r['split']==split:groups.setdefault(r['scene_group'],[]).append(r)
    result=[]
    for g in sorted(groups):
        rs=sorted(groups[g],key=lambda r:r['key'])
        indices=range(len(rs)) if not per_group else sorted(set(np.linspace(0,len(rs)-1,min(per_group,len(rs)),dtype=int)))
        result.extend(rs[i] for i in indices)
    return result

def load_best(path,device='cpu'):
    c=torch.load(path,map_location='cpu',weights_only=True)
    if c.get('format')!=FORMAT:raise RuntimeError('Không phải checkpoint Restormer V2 của bộ này.')
    net=RestorationSystem(c['config'],c['mode'],c.get('mask_base',32));net.load_state_dict(c['state_dict'],strict=True)
    return net.to(device).eval(),c

def initialize(cfg,stage,parent='',mask_path=''):
    mode=STAGES[stage];base=32;mask_checkpoint=None
    if mode!='rgb':
        if stage=='guided':
            mask_checkpoint=torch.load(mask_path,map_location='cpu',weights_only=True)
            if mask_checkpoint.get('format')!='muaxla_mask_unet':raise RuntimeError('Cần mask_best.pt của model Muaxla U-Net, không dùng bản fine-tune Windshield.')
            base=mask_checkpoint['base_channels']
        else:
            pc=torch.load(parent,map_location='cpu',weights_only=True);base=pc.get('mask_base',32)
    net=RestorationSystem(cfg,mode,base)
    if parent:
        pc=torch.load(parent,map_location='cpu',weights_only=True)
        if pc.get('format')!=FORMAT or pc['config']['architecture']!=cfg['architecture']:raise RuntimeError('Parent khác kiến trúc/format.')
        expected='rgb' if stage in {'guided','rgb_control'} else 'guided'
        if pc['mode']!=expected or pc['selected']['source'] not in {'trained','inherited'}:raise RuntimeError('Parent chưa có model trained hợp lệ hoặc sai giai đoạn.')
        state=net.state_dict()
        for k,v in pc['state_dict'].items():
            if k=='backbone.patch_embed.proj.weight' and stage=='guided':
                state[k].zero_();state[k][:,:3]=v
            elif k in state:
                if state[k].shape!=v.shape:raise RuntimeError('Sai shape: '+k)
                state[k]=v
        net.load_state_dict(state,strict=True)
    if mask_checkpoint is not None:net.detector.load_state_dict(mask_checkpoint['model'],strict=True)
    net.freeze_detector();return net,base

def prepare(data_root,out,stage='baseline',config=None,parent='',mask_path='',resume='',input_root='/kaggle/input'):
    if stage not in STAGES:raise ValueError(stage)
    cfg=copy.deepcopy(DEFAULTS);cfg.update(config or {});out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if resume:restore_resume(resume,out)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cpu' and not cfg['allow_cpu']:raise RuntimeError('Bật GPU T4/P100 trong Kaggle.')
    if cfg['crop']%16 or not 0<=cfg['overlap']<cfg['crop']:raise RuntimeError('Crop phải chia hết cho 16; overlap < crop.')
    if cfg['clean_fraction']+cfg['centered_drop_fraction']>1:raise RuntimeError('Tỉ lệ sampling vượt 1.')
    root=du.find_root(input_root,data_root);rows,data_sig=du.audit(root,out)
    ref_hash=du.digest(du.REFERENCE);code_sig=du.digest({n:sha(Path(__file__).parent/n) for n in CODE_FILES if n.endswith('.py')})
    if stage!='baseline':
        if not (out/'parent.pt').exists():
            if not parent:raise RuntimeError('Cần checkpoint best.pt của giai đoạn trước.')
            shutil.copy2(parent,out/'parent.pt')
        parent=str(out/'parent.pt');pc=torch.load(parent,map_location='cpu',weights_only=True)
        if pc.get('split_hash')!=ref_hash or pc.get('data_signature')!=data_sig:raise RuntimeError('Parent và dữ liệu/split hiện tại khác nhau.')
    if stage=='guided' and not (out/'last.pt').exists():
        if not (out/'mask_source.pt').exists():
            if not mask_path:raise RuntimeError('Cần mask_best.pt trong muaxla_mask_model.zip.')
            shutil.copy2(mask_path,out/'mask_source.pt')
        mask_path=str(out/'mask_source.pt')
    effective_protocol={k:v for k,v in cfg.items() if k not in {'workers','session_hours','allow_cpu'}}
    sig=du.digest(dict(stage=stage,config=effective_protocol,data=data_sig,split=ref_hash,code=code_sig,
                       parent=sha(parent) if parent else '',mask=sha(out/'mask_source.pt') if (out/'mask_source.pt').exists() else ''))
    cp=out/'experiment_config.json'
    if cp.exists() and json.loads(cp.read_text())['signature']!=sig:raise RuntimeError('Cấu hình/code/data thay đổi. Dùng OUTPUT_DIR mới; không ghi đè thí nghiệm cũ.')
    du.save_json(dict(signature=sig,stage=stage,config=cfg,split_hash=ref_hash,data_signature=data_sig,code_signature=code_sig,
        implementation='Adapted Restormer; soft predicted mask; small CNN residual refiner. Not a reproduction of STRRNet/HFNet.',
        initialization='random baseline; later stages inherit prior selected weights; no published pretrained Restormer weights',
        comparison='Progressive experiments inherit unequal training budgets; use rgb_control/guided_control for equal extra-update controls.'),cp)
    du.write_csv([{k:v for k,v in r.items() if not k.endswith('_path')} for r in rows],out/'split_manifest.csv')
    for n in CODE_FILES:shutil.copy2(Path(__file__).parent/n,out/n)
    ctx=dict(rows=rows,cfg=cfg,out=out,stage=stage,signature=sig,data_signature=data_sig,split_hash=ref_hash,device=device,parent=parent,mask_path=mask_path)
    print('Stage:',stage,'| Split:',{s:sum(r['split']==s for r in rows) for s in ('train','validation','test')},flush=True)
    return ctx

def create_net(ctx):
    path=ctx['out']/'last.pt'
    if path.exists():
        c=torch.load(path,map_location='cpu',weights_only=True)
        if c['signature']!=ctx['signature']:raise RuntimeError('Resume khác signature.')
        net=RestorationSystem(ctx['cfg'],c['mode'],c['mask_base']);net.load_state_dict(c['state_dict']);base=c['mask_base']
    else:net,base=initialize(ctx['cfg'],ctx['stage'],ctx['parent'],ctx['mask_path'])
    return net.to(ctx['device']),base

def pixel(error,m):return du.masked_mean(error,m)

def edge_terms(pred,target,mask):
    p,t=pred.float(),target.float()
    eh=((p[:,:,:,1:]-p[:,:,:,:-1])-(t[:,:,:,1:]-t[:,:,:,:-1])).abs()
    ev=((p[:,:,1:,:]-p[:,:,:-1,:])-(t[:,:,1:,:]-t[:,:,:-1,:])).abs()
    mh=torch.maximum(mask[:,:,:,1:],mask[:,:,:,:-1]);mv=torch.maximum(mask[:,:,1:,:],mask[:,:,:-1,:])
    return .5*(eh.mean()+ev.mean()),.5*(pixel(eh,mh)+pixel(ev,mv))

def frequency_loss(pred,target):
    diff=torch.fft.rfft2(pred.float(),norm='ortho')-torch.fft.rfft2(target.float(),norm='ortho')
    distance=diff.abs();weight=distance.detach()/(distance.detach().amax((-2,-1),keepdim=True)+1e-6)
    return (weight*distance).mean()

def loss_fn(final,coarse,target,m,cfg,detail=False):
    error=((final.float()-target.float()).square()+1e-6).sqrt()-.001
    loss=error.mean()+cfg['roi_weight']*pixel(error,m)+cfg['background_weight']*pixel(error,1-m)+cfg['ssim_weight']*(1-du.ssim(final,target)).mean()
    if detail and cfg['detail_losses']:
        eg,er=edge_terms(final,target,m)
        loss+=cfg['edge_weight']*(eg+er)+cfg['frequency_weight']*frequency_loss(final,target)
    if detail:loss+=cfg['coarse_weight']*(coarse.float()-target.float()).abs().mean()
    return loss

def ssim_map(a,b):
    k=min(11,a.shape[-2],a.shape[-1]);k-=1-k%2;r=k//2
    v=torch.arange(k,device=a.device,dtype=a.dtype)-r;g=torch.exp(-v.square()/(2*1.5**2));g/=g.sum()
    weight=torch.outer(g,g).expand(a.shape[1],1,k,k)
    def blur(x):return F.conv2d(x,weight,groups=a.shape[1])
    ma,mb=blur(a),blur(b);va=(blur(a*a)-ma*ma).clamp_min(0);vb=(blur(b*b)-mb*mb).clamp_min(0);cov=blur(a*b)-ma*mb
    s=((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))
    return s,r

def metrics(pred,target,m):
    pred=np.clip(pred,0,1);target=target.astype(np.float32);error=np.abs(pred-target)
    a=torch.from_numpy(pred.transpose(2,0,1).copy())[None];b=torch.from_numpy(target.transpose(2,0,1).copy())[None]
    mt=torch.from_numpy(m.astype(np.float32))[None,None];sm,r=ssim_map(a,b)
    mm=mt[...,r:-r,r:-r] if r else mt
    _,eg=edge_terms(a,b,mt)
    mse=float(np.mean((pred-target)**2,dtype=np.float64))
    return dict(psnr=min(100.,-10*math.log10(max(mse,1e-10))),ssim=float(sm.mean()),mae=float(error.mean()),
        roi_mae=float(error[m>0].mean()) if m.any() else None,background_mae=float(error[m==0].mean()) if (m==0).any() else None,
        roi_ssim=float(pixel(sm,mm)) if mm.any() else None,roi_edge_mae=float(eg) if m.any() else None)

def mean_records(records):
    keys=['psnr','ssim','mae','roi_mae','background_mae','roi_ssim','roi_edge_mae','clean_identity_mae','mask_dice','clean_mask_fp']
    groups=sorted({r['scene_group'] for r in records});result={'images':len(records),'groups':len(groups)}
    for k in keys:
        vals=[]
        for g in groups:
            v=[r[k] for r in records if r['scene_group']==g and r.get(k) is not None]
            if v:vals.append(float(np.mean(v)))
        result[k]=float(np.mean(vals)) if vals else None
    return result

def val_examples(ctx):
    cfg=ctx['cfg'];rows=subset(ctx['rows'],'validation',cfg['validation_per_group']);examples=[]
    for r in rows:
        a,b,m=du.rgb(r['drop_path']),du.rgb(r['clean_path']),du.mask(r['mask_path']);h,w=m.shape;t=cfg['crop']
        coords=np.argwhere(m>0)
        y,x=coords[len(coords)//2] if len(coords) else (h//2,w//2)
        y=int(np.clip(y-t//2,0,max(0,h-t)));x=int(np.clip(x-t//2,0,max(0,w-t)))
        examples.append((r,a[y:y+t,x:x+t],b[y:y+t,x:x+t],m[y:y+t,x:x+t]))
    return examples

@torch.inference_mode()
def evaluate_patches(net,ctx):
    net.eval();records=[];raw=[];device=ctx['device']
    for r,a,b,m in val_examples(ctx):
        x=du.tensor(a)[None].to(device);gt=b.astype(np.float32)/255
        p,_,pm=net(x,True);pred=p.float().clamp(0,1)[0].cpu().permute(1,2,0).numpy()
        clean=net(du.tensor(b)[None].to(device)).float().clamp(0,1)[0].cpu().permute(1,2,0).numpy()
        rec=metrics(pred,gt,m);rec.update(key=r['key'],scene_group=r['scene_group'],clean_identity_mae=float(np.abs(clean-gt).mean()))
        if net.detector is not None:
            binary=pm[0,0].cpu().numpy()>=.5;rec['mask_dice']=float(2*(binary&(m>0)).sum()/(binary.sum()+m.sum()+1e-6)) if m.any() else None
            cp=net.detector(du.tensor(b)[None].to(device)).sigmoid();rec['clean_mask_fp']=float((cp>=.5).float().mean())
        records.append(rec);raw.append(dict(metrics(a.astype(np.float32)/255,gt,m),key=r['key'],scene_group=r['scene_group'],clean_identity_mae=0.))
    return dict(model=mean_records(records),raw=mean_records(raw),protocol='fixed foreground-centered validation crops; not full-image results')

@torch.inference_mode()
def tiled_restore(net,a,cfg,device,aux=False):
    net.eval();h,w=a.shape[:2];t=cfg['crop'];o=cfg['overlap'];coords=[(y,x) for y in du.starts(h,t,o) for x in du.starts(w,t,o)]
    signal=np.zeros((h,w,3),np.float32);coarse=np.zeros_like(signal);mask=np.zeros((h,w,1),np.float32);weights=np.zeros((h,w,1),np.float32)
    win=np.maximum(np.outer(np.hanning(t),np.hanning(t)),.01).astype(np.float32)[...,None]
    for y,x in coords:
        tile=a[y:y+t,x:x+t];ph,pw=tile.shape[:2];tile=np.pad(tile,((0,t-ph),(0,t-pw),(0,0)),mode='edge')
        final,c,p=net(du.tensor(tile)[None].to(device),True)
        arrays=[v.float().clamp(0,1)[0].cpu().permute(1,2,0).numpy() for v in (final,c,p)]
        for acc,v in zip((signal,coarse,mask),arrays):acc[y:y+ph,x:x+pw]+=v[:ph,:pw]*win[:ph,:pw]
        weights[y:y+ph,x:x+pw]+=win[:ph,:pw]
    return (signal/weights,coarse/weights,mask/weights) if aux else signal/weights

def payload(ctx,net,base,step,selected):
    return dict(format=FORMAT,config=ctx['cfg'],mode=net.mode,mask_base=base,stage=ctx['stage'],signature=ctx['signature'],
        split_hash=ctx['split_hash'],data_signature=ctx['data_signature'],step=step,selected=selected,
        state_dict={k:v.detach().cpu().clone() for k,v in net.state_dict().items()})

def score(m):return m['roi_mae']+.2*m['roi_edge_mae'] if m['roi_mae'] is not None and m['roi_edge_mae'] is not None else m['mae']

def eligible(m,raw,cfg):
    return m['roi_mae'] is not None and m['roi_mae']<raw['roi_mae'] and m['psnr']>raw['psnr'] and m['background_mae']<=raw['background_mae']+cfg['background_margin'] and m['clean_identity_mae']<=cfg['identity_guard']

def steps_for(ctx):
    s=ctx['stage'];c=ctx['cfg']
    return c['baseline_steps'] if s=='baseline' else c['guided_steps'] if s in {'guided','rgb_control'} else c['hybrid_steps']

def set_phase(net,ctx,step):
    net.requires_grad_(True);net.freeze_detector()
    phase='refiner_only' if ctx['stage']=='hybrid' and step<ctx['cfg']['refiner_only_steps'] else 'joint' if ctx['stage']=='hybrid' else 'backbone'
    net.backbone.requires_grad_(phase!='refiner_only');return phase

def preflight(ctx):
    if (ctx['out']/'preflight.json').exists():return json.loads((ctx['out']/'preflight.json').read_text())
    du.seed_all(ctx['cfg']['seed']);net,_=create_net(ctx);set_phase(net,ctx,steps_for(ctx));net.train()
    cfg=dict(ctx['cfg'],crop=min(ctx['cfg']['crop'],128));ds=du.Patches([r for r in ctx['rows'] if r['split']=='train'],cfg,0,4)
    samples=[ds[i] for i in range(4)];x,y,m=[torch.stack([v[k] for v in samples]).to(ctx['device']) for k in range(3)]
    # Batch=1 to keep pilot below the main training memory budget.
    opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=1e-4)
    losses=[]
    for i in range(16):
        k=i%4;opt.zero_grad();p,c,_=net(x[k:k+1],True);loss=loss_fn(p,c,y[k:k+1],m[k:k+1],cfg,ctx['stage'] in {'hybrid','guided_control'})
        if not torch.isfinite(loss):raise RuntimeError('Preflight NaN/Inf.')
        loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();losses.append(float(loss.detach()))
    result=dict(initial_loss=float(np.mean(losses[:4])),final_loss=float(np.mean(losses[-4:])),finite=True,
        loss_decreased=bool(np.mean(losses[-4:])<np.mean(losses[:4])),note='16 pilot updates on 4 fixed TRAIN crops; pilot weights discarded, not a quality benchmark.')
    if not result['loss_decreased'] and ctx['stage']=='baseline':raise RuntimeError('Pilot loss không giảm. Không chạy lượt train dài; kiểm tra dữ liệu/cấu hình.')
    du.save_json(result,ctx['out']/'preflight.json');del net,opt
    if ctx['device'].type=='cuda':torch.cuda.empty_cache()
    print('Preflight:',result,flush=True);return result

def train(ctx,stop_after=None):
    cfg,out,device=ctx['cfg'],ctx['out'],ctx['device'];du.seed_all(cfg['seed']);net,base=create_net(ctx)
    optimizer=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda');step=0;history=[];finished=False
    selected={};last=out/'last.pt';end_total=steps_for(ctx)
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=True);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        step=saved['step'];history=saved['history'];selected=saved['selected'];finished=saved['finished']
        torch.set_rng_state(saved['torch_rng'])
        if device.type=='cuda' and saved['cuda_rng']:torch.cuda.set_rng_state_all(saved['cuda_rng'])
        if not (out/'best.pt').exists():raise RuntimeError('Resume thiếu best.pt.')
        print('Resume:',step,'/',end_total,'finished=',finished,flush=True)
    else:
        initial=evaluate_patches(net,ctx);selected=dict(step=0,source='identity' if ctx['stage']=='baseline' else 'inherited',metrics=initial['model'],score=score(initial['model']))
        du.save_torch(payload(ctx,net,base,0,selected),out/'best.pt');du.save_json(initial,out/'initial_validation.json')
    if finished:return selected
    effective=cfg['batch']*cfg['accumulation'];end=min(end_total,stop_after if stop_after is not None else end_total)
    ds=du.Patches([r for r in ctx['rows'] if r['split']=='train'],cfg,step*effective,(end-step)*effective)
    loader=DataLoader(ds,batch_size=cfg['batch'],shuffle=False,num_workers=cfg['workers'],pin_memory=device.type=='cuda',drop_last=True)
    # Separate generator prevents DataLoader iterator creation from changing model RNG at resume.
    generator=torch.Generator();generator.manual_seed(cfg['seed']+step);loader.generator=generator
    iterator=iter(loader);start=time.monotonic();begin=step;losses=[];pause=False;updates_without_optimizer=0
    def checkpoint(done=False):
        state=payload(ctx,net,base,step,selected);state.update(optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),history=history,finished=done,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],updates_without_optimizer=updates_without_optimizer)
        du.save_torch(state,last);du.save_json(history,out/'history.json');du.save_json(selected,out/'selected.json')
        if done:du.save_json(dict(stage=ctx['stage'],step=step,finished=True,selected=selected),out/'training_done.json')
    try:
        while step<end:
            phase=set_phase(net,ctx,step);net.train();optimizer.zero_grad(set_to_none=True)
            local_step=step+1
            base_lr=cfg['lr'] if ctx['stage']=='baseline' else cfg['guided_lr'] if ctx['stage'] in {'guided','rgb_control'} else cfg['hybrid_lr']
            if phase=='refiner_only':base_lr=cfg['guided_lr']
            if local_step<=cfg['warmup']:lr=base_lr*local_step/max(1,cfg['warmup'])
            else:lr=base_lr*(.1+.9*(1+math.cos(math.pi*(local_step-cfg['warmup'])/max(1,end_total-cfg['warmup'])))/2)
            for group in optimizer.param_groups:group['lr']=lr
            value=0.
            for _ in range(cfg['accumulation']):
                x,y,m=[v.to(device,non_blocking=True) for v in next(iterator)]
                with torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):p,c,_=net(x,True)
                loss=loss_fn(p,c,y,m,cfg,ctx['stage'] in {'hybrid','guided_control'})/cfg['accumulation']
                if not torch.isfinite(loss):raise RuntimeError('Train loss NaN/Inf.')
                scaler.scale(loss).backward();value+=float(loss.detach())
            scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(net.parameters(),1.)
            old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
            if scaler.get_scale()<old_scale:updates_without_optimizer+=1
            step+=1;losses.append(value)
            if step%50==0:
                sec=(time.monotonic()-start)/max(1,step-begin)
                print(f'{ctx["stage"]} {step}/{end_total} {phase} loss={np.mean(losses[-50:]):.5f} LR={lr:.7f} ~{(end_total-step)*sec/60:.1f} min train còn',flush=True)
            if step%cfg['evaluate_every']==0 or step==end:
                val=evaluate_patches(net,ctx);cur=val['model'];good=eligible(cur,val['raw'],cfg)
                improved=good and (selected['source']=='identity' or score(cur)<selected['score']-1e-6)
                if improved:
                    selected=dict(step=step,source='trained',metrics=cur,score=score(cur));du.save_torch(payload(ctx,net,base,step,selected),out/'best.pt')
                history.append(dict(step=step,phase=phase,train_loss=float(np.mean(losses[-cfg['evaluate_every']:])),lr=lr,validation=val,eligible=good,selected=improved))
                print('VAL:',cur,'| best:',selected['step'],selected['source'],flush=True)
            pause=cfg['session_hours']>0 and (time.monotonic()-start)>cfg['session_hours']*3600
            if step%cfg['save_every']==0 or step==end or pause:checkpoint(step>=end_total)
            if pause:print('Tự tạm dừng trước giới hạn session; tiếp tục bằng resume ZIP.',flush=True);break
    except (KeyboardInterrupt,Exception):
        # A partial accumulated update was never applied: restore deterministic RNG for next step.
        checkpoint(False);pack(ctx,resume_only=True);raise
    checkpoint(step>=end_total);pack(ctx,resume_only=step<end_total)
    del iterator,loader,optimizer,net
    if device.type=='cuda':torch.cuda.empty_cache()
    return selected

def zip_atomic(files,path,root):
    path=Path(path);temp=path.with_suffix('.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in files:z.write(p,Path(p).relative_to(root))
    with zipfile.ZipFile(temp) as z:
        if z.testzip() is not None:raise RuntimeError('ZIP CRC lỗi.')
    os.replace(temp,path);return path

def pack(ctx,resume_only=False):
    out=ctx['out'];stage=ctx['stage'];archives=[]
    if (out/'last.pt').exists():
        names=['last.pt','best.pt','experiment_config.json','history.json','selected.json','training_done.json','parent.pt','mask_source.pt','data_audit.json']
        files=[out/n for n in names if (out/n).exists()]
        archives.append(zip_atomic(files,out.parent/f'muaxla_v2_{stage}_resume.zip',out))
    if not resume_only:
        files=[out/'best.pt',out/'experiment_config.json']+[out/n for n in CODE_FILES]
        archives.append(zip_atomic(files,out.parent/f'muaxla_v2_{stage}_model.zip',out))
    reports=[p for p in out.rglob('*') if p.is_file() and p.suffix in {'.json','.csv','.png','.jpg','.md'} and p.name!='reference_split.json']
    archives.append(zip_atomic(reports,out.parent/f'muaxla_v2_{stage}_reports.zip',out))
    print('ZIP:',*[str(p) for p in archives],sep='\n',flush=True);return archives

def preview(a,p,b,m,path,label='Restored'):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);h,w=a.shape[:2];width=480;height=round(h*width/w)
    arrays=[a,np.uint8(np.clip(p*255,0,255).round()),b];canvas=Image.new('RGB',(3*width,height+28),'white');draw=ImageDraw.Draw(canvas)
    for i,(im,text) in enumerate(zip(arrays,['Drop input',label,'Clean target'])):
        draw.text((i*width+8,8),text,fill='black');canvas.paste(Image.fromarray(im).resize((width,height),Image.Resampling.LANCZOS),(i*width,28))
    canvas.save(path,quality=95)
    coords=np.argwhere(m>0)
    if len(coords):
        y,x=coords[len(coords)//2];t=min(256,h,w);y=int(np.clip(y-t//2,0,h-t));x=int(np.clip(x-t//2,0,w-t))
        crop=Image.new('RGB',(3*t,t+28),'white');draw=ImageDraw.Draw(crop)
        for i,(im,text) in enumerate(zip(arrays,['Drop ROI',label+' ROI','Clean ROI'])):
            draw.text((i*t+8,8),text,fill='black');crop.paste(Image.fromarray(im[y:y+t,x:x+t]),(i*t,28))
        crop.save(path.with_name(path.stem+'_ROI.png'))
        overlay=a.copy();overlay[m>0]=(a[m>0].astype(np.float32)*.5+np.array([255,0,0])*.5).astype(np.uint8)
        Image.fromarray(overlay).resize((width,height)).save(path.with_name(path.stem+'_GT_mask.jpg'),quality=95)

@torch.inference_mode()
def evaluate_native(net,ctx,rows,label,checkpoint_sha,previews=True):
    out=ctx['out'];cache=out/(label+'_metrics.json');per_image=out/(label+'_per_image.csv')
    keys=[r['key'] for r in rows];protocol=du.digest(dict(keys=keys,crop=ctx['cfg']['crop'],overlap=ctx['cfg']['overlap'],checkpoint=checkpoint_sha))
    if cache.exists():
        value=json.loads(cache.read_text())
        if value['protocol_signature']!=protocol:raise RuntimeError('Eval cache khác subset/checkpoint. Xóa file eval tương ứng hoặc dùng OUTPUT_DIR mới.')
        return value
    records=[];raw=[];clean_groups=set();preview_groups=set();net.eval()
    for index,r in enumerate(rows,1):
        a,b,m=du.rgb(r['drop_path']),du.rgb(r['clean_path']),du.mask(r['mask_path']);gt=b.astype(np.float32)/255
        p,c,pm=tiled_restore(net,a,ctx['cfg'],ctx['device'],True)
        rec=dict(metrics(p,gt,m),key=r['key'],scene_group=r['scene_group'],split=r['split'],clean_identity_mae=None,mask_dice=None,clean_mask_fp=None)
        if net.detector is not None:
            binary=pm[...,0]>=.5;rec['mask_dice']=float(2*(binary&(m>0)).sum()/(binary.sum()+m.sum()+1e-6)) if m.any() else None
        if r['scene_group'] not in clean_groups:
            cp,_,cm=tiled_restore(net,b,ctx['cfg'],ctx['device'],True);rec['clean_identity_mae']=float(np.abs(cp-gt).mean())
            if net.detector is not None:rec['clean_mask_fp']=float((cm[...,0]>=.5).mean())
            clean_groups.add(r['scene_group'])
        records.append(rec);raw.append(dict(metrics(a.astype(np.float32)/255,gt,m),key=r['key'],scene_group=r['scene_group'],split=r['split']))
        if previews and r['scene_group'] not in preview_groups:
            stem=r['key'].replace('/','_');pr=out/'previews'/label
            preview(a,p,b,m,pr/(stem+'.jpg'),net.mode)
            Image.fromarray(np.uint8(np.clip(p*255,0,255).round())).save(pr/(stem+'_restored.png'))
            if net.mode=='hybrid':preview(a,c,b,m,pr/(stem+'_coarse.jpg'),'Before refiner')
            if net.detector is not None:Image.fromarray(np.uint8(pm[...,0]*255)).save(pr/(stem+'_predicted_mask.png'))
            preview_groups.add(r['scene_group'])
        if index%10==0 or index==len(rows):print(label,index,'/',len(rows),flush=True)
    value=dict(protocol_signature=protocol,checkpoint_sha256=checkpoint_sha,keys=keys,model=mean_records(records),raw=mean_records(raw),
        protocol='native-resolution fixed-size overlapping tiles; per-scene means; clean identity evaluated once per scene',
        subset=len(rows),full_split=len(rows)==sum(r['split']==rows[0]['split'] for r in ctx['rows']))
    du.write_csv(records,per_image);du.write_csv(raw,out/(label+'_raw_drop.csv'));du.save_json(value,cache);return value

def finish(ctx,run_test=False,full_validation=False,full_test=False):
    out=ctx['out'];done=out/'training_done.json'
    if not done.exists():print('Train chưa xong. Đã xuất resume ZIP; không chạy test.',flush=True);return pack(ctx,resume_only=True)
    net,c=load_best(out/'best.pt',ctx['device']);valrows=subset(ctx['rows'],'validation',0 if full_validation else ctx['cfg']['final_val_per_group'])
    extent='full' if full_validation else str(ctx['cfg']['final_val_per_group'])+'pg'
    result=evaluate_native(net,ctx,valrows,'validation_native_'+extent,sha(out/'best.pt'))
    summary=dict(stage=ctx['stage'],selected=c['selected'],native_validation=result,mask_at_inference='predicted only' if net.detector else False,mask_metrics_threshold=.5,
        inherited_training='baseline=12000; guided inherits baseline + additional guided updates; hybrid inherits guided + additional updates. Read config and histories for actual counts.',
        note='Adapted methods; benefits unproven until ablation. Test split was used in previous project experiments; not a new blind benchmark.')
    if ctx['parent']:
        parent,pc=load_best(ctx['parent'],ctx['device']);pr=evaluate_native(parent,ctx,valrows,'parent_validation_native_'+extent,sha(ctx['parent']),False)
        summary['parent_native_validation']=pr
        summary['validation_delta']={k:result['model'][k]-pr['model'][k] if result['model'][k] is not None and pr['model'][k] is not None else None for k in ['psnr','ssim','roi_mae','roi_edge_mae','roi_ssim','background_mae','clean_identity_mae']}
        del parent
    if run_test:
        if not full_validation:raise RuntimeError('Trước test, đặt FULL_VALIDATION=True để đánh giá đủ validation. Chọn cấu hình bằng validation, không bằng test.')
        te='full' if full_test else str(ctx['cfg']['test_per_group'])+'pg'
        summary['test']=evaluate_native(net,ctx,subset(ctx['rows'],'test',0 if full_test else ctx['cfg']['test_per_group']),'test_native_'+te,sha(out/'best.pt'))
    summary['status']='validation_improved_over_raw' if eligible(result['model'],result['raw'],ctx['cfg']) else 'quality_not_established'
    du.save_json(summary,out/'summary.json');print(json.dumps(summary,indent=2),flush=True);return pack(ctx)

def cli():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='best.pt');p.add_argument('--input',required=True);p.add_argument('--output',default='restored');p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args()
    device=torch.device(a.device)
    if a.device=='cpu':torch.set_num_threads(2)
    net,c=load_best(a.checkpoint,device);source=Path(a.input);dest=Path(a.output)
    if source.resolve()==dest.resolve() or source.resolve() in dest.resolve().parents:raise RuntimeError('Output phải nằm ngoài Input.')
    files=sorted(x for x in source.rglob('*') if x.is_file() and x.suffix.lower() in {'.png','.jpg','.jpeg','.bmp'})
    if not files:raise RuntimeError('Không có ảnh Input.')
    for i,f in enumerate(files,1):
        output=tiled_restore(net,du.rgb(f),c['config'],device);target=dest/f.relative_to(source).parent/(f.name+'.restored.png');target.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(np.uint8(np.clip(output*255,0,255).round())).save(target);print(i,'/',len(files),target,flush=True)

if __name__=='__main__':cli()
