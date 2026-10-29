"""Inference-only launcher for the verified original Muaxla Hybrid checkpoint."""
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCE_SHA='cb5f8f3f7d7c1fdd9d74298f000d29850655b89bd98364a6a3d7a426017edda2'
def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr,'reconfigure'):sys.stderr.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description='Phuc hoi anh bang Hybrid goc; khong huan luyen.')
    parser.add_argument('--input',required=True,help='Anh hoac thu muc anh dau vao')
    parser.add_argument('--output',default=str(ROOT.parent/'Anh_da_phuc_hoi'),help='Mac dinh: Anh_da_phuc_hoi trong thu muc ket qua ban giao')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    source=Path(args.input).resolve();dest=Path(args.output).resolve()
    if not source.exists():parser.error('Input khong ton tai')
    if source==dest or (source.is_dir() and source in dest.parents):parser.error('Output phai nam ngoai Input')
    suffixes={'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}
    files=[source] if source.is_file() and source.suffix.lower() in suffixes else sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in suffixes) if source.is_dir() else []
    if not files:parser.error('Khong co anh ho tro trong Input')
    planned=[]
    for f in files:
        relative=Path(f.name) if source.is_file() else f.relative_to(source)
        target=dest/relative.parent/(relative.name+'.restored.png')
        if target.exists():parser.error('Tu choi ghi de ket qua da co: '+str(target))
        planned.append((f,target))
    checkpoint=ROOT/'best.pt'
    if not checkpoint.is_file() or hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=SOURCE_SHA:raise RuntimeError('Checkpoint Hybrid goc bi thieu/doi hash')
    import numpy as np
    import torch
    from PIL import Image
    source_dir=ROOT/'Ma_nguon' if (ROOT/'Ma_nguon').is_dir() else ROOT
    sys.path.insert(0,str(source_dir))
    import runtime as rt
    import data_utils as du
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA khong san sang; chon --device cpu hoac cai PyTorch CUDA phu hop')
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device(args.device);net,_=rt.load_best(checkpoint,device)
    net.float().eval().requires_grad_(False)
    dest.mkdir(parents=True,exist_ok=True);records=[]
    for i,(f,target) in enumerate(planned,1):
        start=time.monotonic();image=du.rgb(f)
        result=rt.tiled_restore(net,image,dict(crop=384,overlap=128),device)
        if result.shape!=image.shape or not np.isfinite(result).all():raise RuntimeError('Dau ra khong hop le: '+str(f))
        target.parent.mkdir(parents=True,exist_ok=True)
        # Exclusive file creation protects any file created after the preflight.
        with target.open('xb') as stream:Image.fromarray(np.uint8(np.clip(result,0,1)*255+.5)).save(stream,format='PNG')
        records.append(dict(input=str(f),output=str(target),seconds=time.monotonic()-start))
        print(f'{i}/{len(planned)}: {target}',flush=True)
    log=dest/('Phuc_hoi_log_'+str(time.time_ns())+'.json')
    log.write_text(json.dumps(dict(model='Original Muaxla Hybrid',checkpoint_sha256=SOURCE_SHA,tile=384,overlap=128,device=args.device,torch=str(torch.__version__),files=records),ensure_ascii=False,indent=2),encoding='utf-8')
    print('Da luu log:',log)
if __name__=='__main__':main()
