"""Build the single proposal catalog and compact method/build/document entrypoints."""
import json
import os
from pathlib import Path
import re

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file())
STATUS={
 'proxy-mutual-distillation':('Pilot có lợi; full đang nối tới r50','P1 và full r7 đã audit; chưa r50/R1/R2'),
 'gaussian-class-moments':('Pilot có lợi; full đang nối tới r50','P1 covariance/quần thể có lợi; full r9–r10 đã audit, chưa r50'),
 'gaussian-proxy-hybrid':('Pilot hai seed PASS; full đang nối tới r50','V8 paired PASS; full r2–r3 có lợi, interaction âm; chưa synergy/R1/R2'),
 'boundary-cell-counts':('Calibration cost PASS; utility pilot đã push','Hai seed direct/teacher P2/P4 cost PASS; chưa đủ controls/r20 để chốt utility'),
 'receiver-response-consistency':('Fidelity probe PASS; chưa train augmentation','Quyết định mới nhất chỉ local augmentation; chưa mở shared-response FL'),
 'ema-deployment-recipe':('Thiết kế đã chốt; chưa triển khai','Cải tiến Gaussian/DML; không đếm thành họ FL mới'),
 'class-prototypes':('Đã kiểm; đối chứng tham chiếu','Mean prototype không chở đủ ranh giới; không ưu tiên biến thể mean'),
 'contact-fragmentation':('Đã kiểm; hạ tầng, còn giới hạn','P2 audit deadline/byte/prior; chưa chứng minh đầy đủ R1'),
 'federated-head-transfer':('Đã kiểm; không đủ lợi để ưu tiên','P1 head-only LG/FedGH không giải quyết backbone học thuộc'),
 'gaussian-logit-distillation':('Đã kiểm; không đủ lợi để ưu tiên','Teacher local yếu; nhãn Gaussian mạnh hơn DS-FL/FedGen adaptation'),
 'fourier-distribution-sketch':('Pilot hai seed FAIL','S-a KME không hơn Gaussian đủ cổng; không mở S-b'),
 'latent-mean-synthetic-books':('Pilot hai seed FAIL','DM tĩnh và mean-latent refreshed không vượt Gaussian liên tục'),
 'proxy-countsketch':('Pilot hai seed FAIL; EF lỗi số','V2-A sketch residual khuếch đại collision; không rerun y nguyên'),
 'proxy-seeded-subspace':('Pilot FAIL: K512 lỗi số, K1024 teacher yếu','V2-B teacher gate FAIL; không KD/ZO/contact'),
 'gaussian-density-ratio':('Pilot hai seed FAIL','V2-C ESS thấp, không hơn diag/own/shuffle theo cổng'),
 'proxy-random-coordinates':('Pilot hai seed FAIL','V3-R small/medium không giữ đủ lợi dense32'),
 'proxy-rolling-subnetwork':('Pilot hai seed FAIL','V3-T hơn communication-only nhưng thua owncodec'),
 'proxy-orthogonal-projection':('Pilot hai seed FAIL','V3-O small/medium không giữ đủ lợi dense32'),
 'proxy-powersgd':('Pilot hai seed FAIL','V3-P adaptive basis có đóng góp nhưng compression utility FAIL'),
 'gaussian-mala-transport':('Pilot hai seed FAIL','V4-M không qua utility; clip floor làm score correction bằng0 ở nhiều lớp'),
 'gaussian-support-constraints':('Pilot hai seed FAIL','V4-S fidelity PASS không dẫn tới utility hơn diag'),
 'kernel-anchor-distillation':('Pilot hai seed FAIL','V5-H coverage cao đi cùng target gần uniform; teacher gate FAIL'),
 'gradient-synthetic-books':('Pilot hai seed quality FAIL; cost PASS','V5 IPC1/4 đạo hàm khả thi nhưng codebook không đủ utility'),
 'geometry-primitive-moments':('Pilot FAIL: G59 lỗi số, G61 quality/byte FAIL','V6 không faithful ở nhánh missing; joint effect có ở control full66'),
 'covariance-nystrom':('Pilot hai seed quality FAIL; compression PASS','V7 rank4/8 giảm byte nhưng không giữ đủ lợi fullcov'),
 'anchor-function-residual':('Pilot hai seed quality FAIL; cost PASS','V9 M64/512 không qua teacher quality; không privateKD/full y nguyên'),
 'dagsnet-joint-energy':('Cost/sampler FAIL; candidate r20 chưa đo','V10 Gaussian-base+MH hợp lệ về base nhưng không qua cổng runtime/sampler'),
}


def relative(file,target):return os.path.relpath(target,file.parent)


def write(path,body):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(body.rstrip()+'\n')


def main():
    record=json.loads((ROOT/'docs/operations/layout-migration-2026-10-14.json').read_text())
    membership={name:[] for name in STATUS}
    for old,build in record['builds'].items():
        for method in [build['family']]+build['related']:
            membership.setdefault(method,[]).append((old,build))
    catalog=[]
    for name,(status,basis) in STATUS.items():
        doc=ROOT/'docs/methods'/name;entry=doc/'README.md'
        old=entry.read_text();title=old.splitlines()[0].removeprefix('# ').strip()
        mechanism=re.search(r'## Cơ chế FL\s+(.+?)(?:\n## |\Z)',old,re.S)
        design=doc/'design.md'
        links=re.findall(r'^- \[.+$',old,re.M)
        if not design.exists():
            write(design,f'# {title} — căn cứ và cơ chế\n\n'
                f'## Căn cứ xây dựng\n\n{basis}. Bằng chứng và giới hạn tại [bài học phương pháp](lessons.md) '
                'và [tổng hợp bằng chứng](../../overview/evidence-summary.md). Không suy codec/probe PASS thành utility PASS.\n\n'
                '## Cách thực hiện\n\n'+(mechanism[1].strip() if mechanism else
                'Cơ chế, cấu hình và cổng cụ thể nằm trong các thiết kế gốc bên dưới; giữ các quyết định đã chốt.')+
                '\n\n## Thiết kế gốc, code và bằng chứng\n\n'+'\n'.join(links)+
                '\n\nKhông raw/perturbed record rời xe. Mọi learned model hiện DAGSNet; R2 khác kiến trúc chỉ ở thiết kế. '
                'Pilot StageU không chứng minh contact hữu hạn, secure aggregation hoặc DP.\n\n[Kết quả và các build](README.md).')
        elif links:
            with design.open('a') as stream:stream.write('\n\n## Đường dẫn triển khai kế thừa\n\n'+'\n'.join(links)+'\n')
        buildlinks=[]
        for old,build in membership[name]:
            bd=ROOT/'docs/methods'/build['family']/'builds'/build['build']/'README.md'
            buildlinks.append(f'- [{build["build"]}]({relative(entry,bd)}) — nhãn cũ `{old}`.')
        if name=='gaussian-class-moments':
            buildlinks.append('- [Input-statistics probe](../../../methods/gaussian-class-moments/builds/input-statistics-probe/README.md).')
        write(entry,f'# {title}\n\n{status}. {basis}.\n\n'
            '[Căn cứ và cơ chế](design.md) · [Bài học riêng](lessons.md) · [Tất cả đề xuất](../../proposals.md)\n\n'
            '## Build, kết quả và artifact\n\n'+('\n'.join(buildlinks) if buildlinks else
            'Chưa có build huấn luyện. Giữ thiết kế/cổng đã chốt; chưa gọi thành công hoặc thất bại utility.')+
            '\n\nMỗi build dẫn tới source, probe/calibration/pilot, kết quả, logs và DVC. Build chung chỉ có một bản artifact.')
        if not (doc/'lessons.md').exists():
            write(doc/'lessons.md',f'# {title} — bài học\n\n{basis}. Chưa có utility mới; không kết luận từ thiết kế hoặc fidelity probe.\n\n[Thiết kế](design.md) · [Kết quả](README.md).')
        catalog.append(dict(method=name,title=title,status=status,basis=basis,recipe=name=='ema-deployment-recipe',
            builds=[v['path'] for _,v in membership[name]]))
    write(ROOT/'docs/methods/catalog.json',json.dumps(catalog,indent=2,ensure_ascii=False))
    for old,build in record['builds'].items():
        family=build['family'];name=build['build'];physical=ROOT/build['path']
        doc=ROOT/'docs/methods'/family/'builds'/name/'README.md'
        folder=doc.parent;results=folder/'results'
        reports=[p for p in sorted(results.glob('*.md')) if p.name!='README.md']
        spec=ROOT/'docs/research'/({'pilot':'pilot-plan.md','p2':'pilot-plan.md','new-fl':'pilot-new-fl.md',
            'v2':'pilot-v2.md','v3':'pilot-v3.md','v4':'pilot-v4-support.md','v5':'pilot-v5-books.md',
            'v6':'pilot-v6-geometry.md','v7':'pilot-v7-nystrom.md','v8':'pilot-v8-hybrid.md',
            'v9':'pilot-v9-residual.md','v10':'pilot-v10-energy.md','full-validation':'full-h1-h2.md',
            'hybrid-full':'full-hybrid.md'}.get(old,'README.md'))
        scientific=next((v for v in catalog if v['method']==family),None)
        status=scientific['status'] if scientific else 'Build so sánh nhiều phương pháp; trạng thái từng arm tại danh mục đề xuất'
        design=ROOT/'docs/methods'/family/'design.md'
        body=f'# Build {name}\n\nNhãn lịch sử `{old}`; {status}.\n\n'
        body+=f'[Căn cứ/cơ chế]({relative(doc,design)}) · [Cấu hình và cổng]({relative(doc,spec)}) · [Danh mục đề xuất]({relative(doc,ROOT/"docs/proposals.md")})\n\n'
        body+='## Giai đoạn trong cùng build\n\n| phần | đường dẫn |\n|---|---|\n'
        for label,path in [('Source đã chạy',physical/'source'),('Probe và proof',physical/'probes'),
                           ('Calibration/pilot/continuation notebooks',physical/'notebooks'),
                           ('Artifact gồm smoke/checkpoint/prediction',physical/'artifacts'),('Scripts riêng',physical/'scripts'),
                           ('Logs',ROOT/'logs'/family/name)]:
            if path.exists():body+=f'| {label} | [{path.name}/]({relative(doc,path)}) |\n'
        body+='\n## Kết quả đã có\n\n'+('\n'.join(f'- [{p.name}](results/{p.name})' for p in reports) or
            'Chưa có utility report. Probe/calibration proof không thay r20/r50 đã audit.')
        metadata=list(physical.glob('*.dvc'))
        body+='\n\n## Khôi phục artifact\n\n'+('\n'.join(f'- [{p.name}]({relative(doc,p)})' for p in metadata) or
            'Chưa có metadata DVC cho artifact build này; không suy đã backup.')
        body+='\n\nDVC có thể chỉ có metadata ở local sau cleanup. Hash remote giữ nguyên; restore đúng target cần dùng. '
        body+='Source/manifest/notebook đã PUSHED bất biến; source snapshot dùng đường dẫn runtime gốc trong notebook. '
        body+='[Quy tắc cấu trúc và alias]('+relative(doc,ROOT/'docs/operations/project-layout.md')+').'
        write(doc,body)
        write(physical/'README.md',f'# {name}\n\nBuild thuộc `{family}`, nhãn cũ `{old}`.\n\n'
            f'[Căn cứ, cơ chế, cấu hình và kết quả]({relative(physical/"README.md",doc)}) · '
            f'[Đề xuất]({relative(physical/"README.md",ROOT/"docs/proposals.md")}).\n\n'
            'Source/notebooks/scripts/probes/artifacts thuộc build này; logs có đường dẫn riêng. Không nhân đôi artifact của build chung.')
        for method in build['related']:
            alias=ROOT/'methods'/method/'builds'/name;alias.parent.mkdir(parents=True,exist_ok=True)
            if not alias.exists():alias.symlink_to(os.path.relpath(physical,alias.parent),target_is_directory=True)
    # Every method root and canonical build subsection has a short entrypoint.
    families=set(membership)|{v['family'] for v in record['builds'].values()}
    for family in sorted(families):
        physical=ROOT/'methods'/family;physical.mkdir(parents=True,exist_ok=True)
        doc=ROOT/'docs/methods'/family/'README.md'
        if not doc.exists():
            items=membership.get(family,[])
            write(doc,f'# {family} — nhóm build so sánh chung\n\n'
                'Thư mục vật lý dùng chung cho đối chứng/so sánh nhiều cơ chế; không đếm là đề xuất mới.\n\n'
                '[Danh mục và trạng thái từng phương pháp](../../proposals.md)\n\n'+
                '\n'.join(f'- [{b["build"]}](builds/{b["build"]}/README.md)' for _,b in items))
        design=doc.parent/'design.md'
        if not design.exists():write(design,f'# {family} — căn cứ build chung\n\n'
            'Giữ cùng source, fixture, seed, lịch LR và compute để so sánh cơ chế. Mỗi cơ chế có thiết kế/cổng riêng '
            'tại [danh mục đề xuất](../../proposals.md); build chung không tự là thuật toán mới.\n\n[Build và kết quả](README.md).')
        write(physical/'README.md',f'# {family}\n\n[Phương pháp, căn cứ và kết quả]({relative(physical/"README.md",doc)}).\n\n'
            'Các bản build nằm trong `builds/`; liên kết build chung không sao chép artifact.')
    # Markdown entrypoints outside immutable DVC contents and private credential folders.
    roots=[ROOT/'methods',ROOT/'docs/methods',ROOT/'logs',ROOT/'scripts']
    for top in roots:
        for directory,subdirs,_ in os.walk(top,followlinks=False):
            path=Path(directory)
            subdirs[:]=[n for n in subdirs if n not in ('__pycache__','private_push') and not (path/n).is_symlink()]
            if path.name=='artifacts':subdirs[:]=[]
            if (path/'README.md').exists():continue
            write(path/'README.md',f'# {path.name}\n\n'
                f'[Mục lục tổng quan]({relative(path/"README.md",ROOT/"docs/README.md")}) · '
                f'[Đề xuất/cơ chế/kết quả]({relative(path/"README.md",ROOT/"docs/proposals.md")}).\n\n'
                'Nội dung thuộc thư mục hiện tại; source, probe/calibration/pilot và artifact giữ trong đúng build. '
                'Không sửa snapshot đã PUSHED hoặc nội dung DVC đã xác minh.')
    page=ROOT/'docs/proposals.md'
    body='# Tất cả đề xuất và trạng thái kiểm nghiệm — 14/10/2026\n\n'
    body+='**26 nhóm đề xuất + 1 recipe EMA.** 3 nhóm có pilot tích cực: Gaussian, DML, Hybrid. '
    body+='Chưa nhóm nào có đủ r50/R1/R2 proof. Không đếm seed/size/controls/build chung thành phương pháp mới.\n\n'
    body+='[Mục lục](README.md) · [Bằng chứng/bài học theo phương pháp](overview/evidence-summary.md) · [Bài học chung](../LESSONS.md)\n\n'
    body+='| phương pháp | đã kiểm nghiệm / thành công hay thất bại | căn cứ | build, kết quả, artifact |\n|---|---|---|---|\n'
    for entry in catalog:
        name=entry['method'];body+=f'| [{entry["title"]}](methods/{name}/design.md) | {entry["status"]} | {entry["basis"]} | [Các build](methods/{name}/README.md) |\n'
    body+='\n## Tiếp tục theo bằng chứng\n\nH1/H2/Hybrid nối checkpoint thực tế tới r50 theo quota. '
    body+='BoundaryCells mở pilot sau cost PASS; chưa đổi cấu hình/gp/cổng theo kết quả ban đầu. '
    body+='ResponseRisk chỉ local augmentation trước, EMA chưa có source/pilot. Các adaptation FAIL không rerun y nguyên.\n\n'
    body+='Pilot PASS chỉ áp dụng adaptation/protocol đã đo; không suy có lợi khi contact hữu hạn hoặc đa kiến trúc. '
    body+='[Notebook mới và quota](operations/launch-batch-2026-10-14.json) · [Sổ lỗi kernel](operations/notebook-errors.md).'
    write(page,body)
    write(ROOT/'docs/methods/README.md','# Hồ sơ phương pháp\n\n[Toàn bộ đề xuất và trạng thái](../proposals.md) · '
        '[Mục lục tài liệu](../README.md).\n\nMỗi phương pháp có README ngắn, design, lessons và các build/kết quả. '
        'Các nhóm build chung phục vụ đối chứng, không đếm là phương pháp mới.')
    print('Navigation:',len(catalog),'method/recipe profiles;',len(record['builds']),'build dossiers; single proposal catalog')


if __name__=='__main__':main()
